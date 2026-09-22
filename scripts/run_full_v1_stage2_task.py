from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/author_core_v1"))

# Reuse the locked Stage 1 neural training implementation without modifying it.
from scripts import run_full_v1_task as stage1
from src.data.formal_data import formal_split, generic_groups, load_formal_batches, rmsse_scales
from src.graphs.diagnostics import graph_statistics

MANIFEST = ROOT / "artifacts/v1/manifests/full_unified_task_manifest_v1.csv"
RUN_ROOT = ROOT / "outputs/full_unified_run_v1/stage2"
EXECUTION_STAGE = "stage_2"
REFERENCE_MODELS = {"seasonal_naive", "global_lightgbm"}
AMENDMENT = ROOT / "configs/v1/FULL_UNIFIED_RUN_V1_STAGE2_EXECUTION_AMENDMENT_001.json"


def read_task(task_id: str) -> dict:
    with MANIFEST.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["full_run_task_id"] == task_id]
    if len(matches) != 1:
        raise KeyError(f"Stage 2 task not found or non-unique: {task_id}")
    return matches[0]


def validate_task(task: dict) -> None:
    if task.get("execution_stage") != EXECUTION_STAGE:
        raise RuntimeError(f"non-stage2 task: {task.get('execution_stage')}")
    for key in ("config_sha256", "data_slice_sha256", "node_ids_sha256", "metric_protocol_sha256", "source_commit_or_lock"):
        if not str(task.get(key, "")).strip() or "RESOLVE" in str(task.get(key, "")).upper():
            raise RuntimeError(f"unresolved required task field: {key}")


def prediction_products(task: dict, outdir: Path, log_prediction: torch.Tensor, formal, inference_seconds: float, learned_graph, diagnostics: dict):
    if list(log_prediction.shape) != [1, 28, 90]:
        raise RuntimeError(f"prediction shape mismatch: {tuple(log_prediction.shape)}")
    prediction = np.clip(np.expm1(log_prediction.detach().cpu().numpy()[0]), 0, None)
    actual = np.clip(np.expm1(formal.target.detach().cpu().numpy()[0]), 0, None)
    dataset = stage1.dataset_key(task["dataset"])
    scales = rmsse_scales(dataset, task["forecast_origin"])
    dates = pd.date_range(task["forecast_origin"], periods=28, freq="D")
    groups = generic_groups(list(map(str, formal.node_ids))).set_index("node_id")
    records = []
    for horizon in range(28):
        for node, node_id in enumerate(map(str, formal.node_ids)):
            records.append({
                "experiment_version": "FULL_UNIFIED_RUN_V1",
                "execution_stage": EXECUTION_STAGE,
                "dataset": task["dataset"],
                "forecast_origin": task["forecast_origin"],
                "seed": int(task["seed"]),
                "model_id": task["model_id"],
                "node_id": node_id,
                "node_index": node,
                "item_block": groups.loc[node_id, "item_block"],
                "store_slot": groups.loc[node_id, "store_slot"],
                "target_date": str(dates[horizon].date()),
                "horizon_step": horizon + 1,
                "actual": float(actual[horizon, node]),
                "prediction": float(prediction[horizon, node]),
                "rmsse_scale_denominator": float(scales[node]),
            })
    frame = pd.DataFrame(records)
    frame.to_csv(outdir / "predictions.csv.gz", index=False, compression="gzip")
    error = frame.actual - frame.prediction
    frame["absolute_error"] = error.abs()
    frame["squared_error"] = error ** 2
    node = frame.groupby(["node_id", "node_index", "item_block", "store_slot"], as_index=False).agg(
        MAE=("absolute_error", "mean"), squared_error=("squared_error", "mean"),
        actual_sum=("actual", "sum"), absolute_error_sum=("absolute_error", "sum"),
        rmsse_scale_denominator=("rmsse_scale_denominator", "first"),
    )
    node["WAPE"] = node.absolute_error_sum / node.actual_sum.replace(0, np.nan)
    node["RMSSE"] = np.sqrt(node.squared_error / node.rmsse_scale_denominator.where(node.rmsse_scale_denominator > 0))
    node.to_csv(outdir / "node_metrics.csv", index=False, encoding="utf-8-sig")
    horizon = frame.groupby("horizon_step", as_index=False).agg(MAE=("absolute_error", "mean"), actual_sum=("actual", "sum"), absolute_error_sum=("absolute_error", "sum"))
    horizon["WAPE"] = horizon.absolute_error_sum / horizon.actual_sum.replace(0, np.nan)
    horizon.to_csv(outdir / "horizon_metrics.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "RMSSE": float(node.RMSSE.mean()), "RMSSE_eligible_nodes": int(node.RMSSE.notna().sum()),
        "WAPE": float(frame.absolute_error.sum() / frame.actual.sum()) if frame.actual.sum() else None,
        "MAE": float(frame.absolute_error.mean()), "prediction_rows": len(frame), "inference_seconds": inference_seconds,
    }
    stage1.atomic_json(outdir / "metric_results.json", metrics)
    stage1.atomic_json(outdir / "graph_diagnostics.json", graph_statistics(learned_graph) if learned_graph is not None else {"not_applicable": True})
    stage1.atomic_json(outdir / "input_usage.json", {"used_fields": diagnostics.get("used_fields", []), "target_exposed_to_forward": False})
    return metrics


def neural_run(task: dict, outdir: Path, test_limits=None, device_name="cuda"):
    dataset = stage1.dataset_key(task["dataset"])
    selection, inner, refit, formal = load_formal_batches(dataset, task["forecast_origin"])
    _, inner_origin, _ = formal_split(dataset, task["forecast_origin"])
    leakage = pd.Timestamp(inner_origin) + pd.Timedelta(days=27) < pd.Timestamp(task["forecast_origin"])
    if not leakage:
        raise RuntimeError("inner validation leakage detected")
    device = torch.device(device_name)
    stage1.seed_all(int(task["seed"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_epoch, selection_logs = stage1.selection_phase(task, outdir, selection, inner, device, test_limits)
    model, refit_logs = stage1.refit_phase(task, outdir, refit, best_epoch, device)
    model.eval()
    start = time.perf_counter()
    output = model.predict(formal.to(device))
    inference_seconds = time.perf_counter() - start
    metrics = prediction_products(task, outdir, output.prediction, formal, inference_seconds, output.learned_graph, output.diagnostics)
    stage1.reload_check(task, outdir, formal, output, device)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
    resource = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds": None, "inference_seconds": inference_seconds,
        "peak_gpu_mib": peak, "within_5_2_gib": peak < 5.2 * 1024, "batch_size": stage1.BATCH_SIZE,
    }
    return {
        "selection_logs": selection_logs, "refit_logs": refit_logs, "best_epoch": best_epoch,
        "inner_origin": inner_origin, "leakage": leakage, "output": output,
        "resource": resource, "metrics": metrics,
    }


def _reference_features(batch) -> np.ndarray:
    return batch.past_target[0, [-1, -7, -14, -28, -56], :].detach().cpu().numpy().T


def _reference_target(batch) -> np.ndarray:
    return np.expm1(batch.target[0].detach().cpu().numpy()).T


def _reference_checkpoint(task: dict, baseline: str, state, selected_trees=None) -> dict:
    return {
        "experiment_version": "FULL_UNIFIED_RUN_V1", "full_run_task_id": task["full_run_task_id"],
        "config_sha256": task["config_sha256"], "data_slice_sha256": task["data_slice_sha256"],
        "node_ids_sha256": task["node_ids_sha256"], "source_lock": task["source_commit_or_lock"],
        "training_phase": "final_refit", "epoch": None, "baseline": baseline,
        "selected_trees": selected_trees, "state": state, "timestamp": stage1.now(),
    }


def seasonal_naive_run(task: dict, outdir: Path):
    from models.baselines import seasonal_naive
    dataset = stage1.dataset_key(task["dataset"])
    _, _, _, formal = load_formal_batches(dataset, task["forecast_origin"])
    start = time.perf_counter()
    raw = seasonal_naive(np.expm1(formal.past_target[0].detach().cpu().numpy()).T, horizon=28, season=7)
    log_prediction = torch.tensor(np.log1p(raw.T), dtype=torch.float32).unsqueeze(0)
    inference_seconds = time.perf_counter() - start
    state = {"season": 7}
    payload = _reference_checkpoint(task, "seasonal_naive", state)
    stage1.atomic_torch(outdir / "checkpoint_best_inner_validation.pt", payload)
    stage1.atomic_torch(outdir / "checkpoint_last.pt", payload)
    stage1.atomic_torch(outdir / "checkpoint_final.pt", payload)
    reloaded = torch.load(outdir / "checkpoint_final.pt", map_location="cpu")
    if reloaded.get("baseline") != "seasonal_naive" or reloaded.get("state", {}).get("season") != 7:
        raise RuntimeError("seasonal naive checkpoint reload failed")
    metrics = prediction_products(task, outdir, log_prediction, formal, inference_seconds, None, {"used_fields": ["past_target"]})
    stage1.atomic_json(outdir / "checkpoint_reload_check.json", {"passed": True, "training_phase": "not_applicable", "epoch": None, "config_sha256": task["config_sha256"]})
    pd.DataFrame([{"training_semantics": "deterministic_seasonal_naive", "season": 7}]).to_csv(outdir / "epoch_selection_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"training_semantics": "deterministic_final_prediction", "season": 7}]).to_csv(outdir / "final_refit_log.csv", index=False, encoding="utf-8-sig")
    return {"best_epoch": None, "inner_origin": None, "leakage": True, "log_prediction": log_prediction,
            "resource": {"parameters": 0, "trainable_parameters": 0, "elapsed_seconds": None, "inference_seconds": inference_seconds, "peak_gpu_mib": 0.0, "within_5_2_gib": True, "batch_size": None}, "metrics": metrics}


def lightgbm_candidates(task: dict) -> list[int]:
    if not AMENDMENT.exists():
        raise RuntimeError("source_blocked: required Stage 2 LightGBM amendment is absent")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8-sig"))
    values = amendment.get("tree_count_candidates")
    if not amendment.get("author_authorization") or not isinstance(values, list):
        raise RuntimeError("source_blocked: invalid Stage 2 LightGBM amendment")
    try:
        values = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("source_blocked: invalid locked LightGBM tree-candidate set") from exc
    if not values or len(set(values)) != len(values) or any(value <= 0 for value in values):
        raise RuntimeError("source_blocked: invalid locked LightGBM tree-candidate set")
    return values


def global_lightgbm_run(task: dict, outdir: Path):
    from models.baselines import GlobalLightGBM
    candidates = lightgbm_candidates(task)
    dataset = stage1.dataset_key(task["dataset"])
    selection, inner, refit, formal = load_formal_batches(dataset, task["forecast_origin"])
    _, inner_origin, _ = formal_split(dataset, task["forecast_origin"])
    if pd.Timestamp(inner_origin) + pd.Timedelta(days=27) >= pd.Timestamp(task["forecast_origin"]):
        raise RuntimeError("inner validation leakage detected")
    x_selection = np.concatenate([_reference_features(batch) for batch in selection])
    y_selection = np.concatenate([_reference_target(batch) for batch in selection])
    x_inner, y_inner = _reference_features(inner), _reference_target(inner)
    logs, best_model, best_trees, best_loss = [], None, None, float("inf")
    for trees in candidates:
        model = GlobalLightGBM(random_state=int(task["seed"]), n_estimators=trees).fit(x_selection, y_selection)
        loss = float(np.mean(np.abs(model.predict(x_inner) - y_inner)))
        logs.append({"n_estimators": trees, "inner_validation_loss": loss})
        if loss < best_loss:
            best_model, best_trees, best_loss = model, trees, loss
    stage1.atomic_torch(outdir / "checkpoint_best_inner_validation.pt", _reference_checkpoint(task, "global_lightgbm", best_model, best_trees))
    pd.DataFrame(logs).to_csv(outdir / "epoch_selection_log.csv", index=False, encoding="utf-8-sig")
    x_refit = np.concatenate([_reference_features(batch) for batch in refit])
    y_refit = np.concatenate([_reference_target(batch) for batch in refit])
    start = time.perf_counter()
    final_model = GlobalLightGBM(random_state=int(task["seed"]), n_estimators=best_trees).fit(x_refit, y_refit)
    raw = final_model.predict(_reference_features(formal))
    inference_seconds = time.perf_counter() - start
    log_prediction = torch.tensor(np.log1p(raw.T), dtype=torch.float32).unsqueeze(0)
    payload = _reference_checkpoint(task, "global_lightgbm", final_model, best_trees)
    stage1.atomic_torch(outdir / "checkpoint_last.pt", payload)
    stage1.atomic_torch(outdir / "checkpoint_final.pt", payload)
    reloaded = torch.load(outdir / "checkpoint_final.pt", map_location="cpu")
    reload_prediction = reloaded["state"].predict(_reference_features(formal))
    if not np.allclose(raw, reload_prediction, atol=1e-6, rtol=1e-5):
        raise RuntimeError("Global LightGBM checkpoint reload prediction mismatch")
    pd.DataFrame([{"n_estimators": best_trees, "refit_samples": len(x_refit)}]).to_csv(outdir / "final_refit_log.csv", index=False, encoding="utf-8-sig")
    metrics = prediction_products(task, outdir, log_prediction, formal, inference_seconds, None, {"used_fields": ["past_target"]})
    stage1.atomic_json(outdir / "checkpoint_reload_check.json", {"passed": True, "training_phase": "final_refit", "epoch": None, "config_sha256": task["config_sha256"], "selected_trees": best_trees})
    return {"best_epoch": None, "selected_trees": best_trees, "inner_origin": inner_origin, "leakage": True, "log_prediction": log_prediction,
            "resource": {"parameters": 0, "trainable_parameters": 0, "elapsed_seconds": None, "inference_seconds": inference_seconds, "peak_gpu_mib": 0.0, "within_5_2_gib": True, "batch_size": None}, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-root", help="isolated engineering-test output root only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-max-epochs", type=int)
    parser.add_argument("--test-min-epochs", type=int)
    parser.add_argument("--test-patience", type=int)
    args = parser.parse_args()
    task = read_task(args.task_id)
    validate_task(task)
    outdir = (Path(args.output_root) if args.output_root else RUN_ROOT) / task["full_run_task_id"]
    outdir.mkdir(parents=True, exist_ok=True)
    marker = outdir / "completion.marker"
    if marker.exists():
        print(json.dumps({"task_id": args.task_id, "status": "passed", "resumed": True}))
        return
    started = stage1.now()
    start_clock = time.perf_counter()
    stage1.atomic_json(outdir / "resolved_config.json", task)
    stage1.atomic_json(outdir / "run_status.json", {"task_id": task["full_run_task_id"], "status": "running", "start_time": started, "execution_stage": EXECUTION_STAGE})
    try:
        if task["model_id"] == "seasonal_naive":
            result = seasonal_naive_run(task, outdir)
        elif task["model_id"] == "global_lightgbm":
            result = global_lightgbm_run(task, outdir)
        else:
            limits = None
            if args.test_max_epochs is not None:
                limits = {"max_epochs": args.test_max_epochs, "min_epochs": args.test_min_epochs or 1, "patience": args.test_patience or 1}
            result = neural_run(task, outdir, limits, args.device)
        result["resource"]["elapsed_seconds"] = time.perf_counter() - start_clock
        stage1.atomic_json(outdir / "resource_profile.json", result["resource"])
        output = result.get("output")
        shape_ok = True if output is None else list(output.prediction.shape) == [1, 28, 90]
        finite_ok = bool(torch.isfinite(result.get("log_prediction", output.prediction if output is not None else torch.empty(0))).all())
        status = {
            "task_id": task["full_run_task_id"], "dataset": task["dataset"], "forecast_origin": task["forecast_origin"],
            "seed": int(task["seed"]), "model_id": task["model_id"], "execution_stage": EXECUTION_STAGE,
            "status": "passed", "start_time": started, "end_time": stage1.now(),
            "selected_epoch": result.get("best_epoch"), "selected_trees": result.get("selected_trees"),
            "inner_validation_origin": result.get("inner_origin"), "formal_origin": task["forecast_origin"],
            "leakage_check_passed": result["leakage"], "checkpoint_reload_passed": True,
            "prediction_shape_passed": shape_ok, "finite_output_passed": finite_ok,
            "graph_export_passed": (output is not None and output.learned_graph is not None) or task["graph_mode"] in {"none", "no_graph"},
            "peak_gpu_mib": result["resource"]["peak_gpu_mib"], **result["metrics"],
        }
        stage1.atomic_json(outdir / "run_status.json", status)
        stage1.atomic_json(marker, {"task_id": task["full_run_task_id"], "config_sha256": task["config_sha256"], "data_slice_sha256": task["data_slice_sha256"], "execution_stage": EXECUTION_STAGE, "completed": stage1.now()})
        print(json.dumps(status, ensure_ascii=False))
    except KeyboardInterrupt:
        stage1.atomic_json(outdir / "run_status.json", {"task_id": task["full_run_task_id"], "status": "interrupted", "updated": stage1.now(), "safe_checkpoint_exists": (outdir / "checkpoint_last.pt").exists()})
        raise
    except Exception as exc:
        (outdir / "exception.txt").write_text(repr(exc) + "\n" + traceback.format_exc(), encoding="utf-8")
        stage1.atomic_json(outdir / "run_status.json", {"task_id": task["full_run_task_id"], "status": "failed", "failure_stage": "formal_v1_stage2_task", "exception_type": type(exc).__name__, "exception_summary": str(exc), "updated": stage1.now(), "safe_checkpoint_exists": (outdir / "checkpoint_last.pt").exists()})
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


