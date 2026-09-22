from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts.run_compatibility_task import build_model as compatibility_build_model
from scripts.run_compatibility_task import config_for as compatibility_config_for
from src.adapters.legacy_adapter import ControlledTCNAdapter
from src.data.compatibility_data import concatenate_batches
from src.data.formal_data import formal_split, generic_groups, load_formal_batches, rmsse_scales
from src.graphs.diagnostics import graph_statistics


MANIFEST = ROOT / "artifacts/v1/manifests/full_unified_task_manifest_v1.csv"
RUN_ROOT = ROOT / "outputs/full_unified_run_v1/stage1"
MAX_EPOCHS = 50
MIN_EPOCHS = 10
PATIENCE = 7
MIN_RELATIVE_IMPROVEMENT = 0.001
BATCH_SIZE = 4
CLIP_NORM = 5.0


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state() -> dict:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(value: dict) -> None:
    random.setstate(value["python_random_state"])
    np.random.set_state(value["numpy_random_state"])
    torch.set_rng_state(value["torch_cpu_rng_state"])
    if torch.cuda.is_available() and value.get("torch_cuda_rng_state"):
        torch.cuda.set_rng_state_all(value["torch_cuda_rng_state"])


def read_task(task_id: str) -> dict:
    with MANIFEST.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["full_run_task_id"] == task_id)


def dataset_key(value: str) -> str:
    return value.lower().replace(" ", "_")


def build_model(task: dict):
    model_id = task["model_id"]
    dataset = dataset_key(task["dataset"])
    seed = int(task["seed"])
    config = compatibility_config_for(model_id, dataset, BATCH_SIZE)
    config.update({"seed": seed, "forecast_origin": task["forecast_origin"], "batch_size": BATCH_SIZE})
    if model_id.startswith("tcn_") and model_id != "tcn_target_only":
        return ControlledTCNAdapter.build(config)
    if model_id in {"tcn_target_only", "dlinear", "nlinear", "patchtst", "itransformer", "timemixerpp", "mixlinear"}:
        from src.adapters.legacy_adapter import ModernAdapter
        return ModernAdapter.build(config)
    from src.adapters.registry import build_adapter
    return build_adapter(model_id, config)


def optimizer_for(model, model_id: str):
    return torch.optim.Adam(model.parameters(), lr=0.01 if model_id == "mixlinear" else 0.001)


def batch_loss(model, batch):
    output = model(batch)
    loss = F.huber_loss(output.prediction, batch.target)
    if isinstance(model, ControlledTCNAdapter):
        loss = loss + model.regularization_loss()
    return loss, output


def checkpoint_payload(task, phase, epoch, model, optimizer, best_loss, best_epoch, counter, logs) -> dict:
    return {
        "experiment_version": "FULL_UNIFIED_RUN_V1",
        "full_run_task_id": task["full_run_task_id"],
        "config_sha256": task["config_sha256"],
        "data_slice_sha256": task["data_slice_sha256"],
        "node_ids_sha256": task["node_ids_sha256"],
        "source_lock": task["source_commit_or_lock"],
        "training_phase": phase,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "best_inner_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "early_stopping_counter": counter,
        "logs": logs,
        "rng": rng_state(),
        "timestamp": now(),
    }


def validate_checkpoint(task: dict, state: dict) -> None:
    for key in ("full_run_task_id", "config_sha256", "data_slice_sha256", "node_ids_sha256"):
        if str(state.get(key)) != str(task.get(key)):
            raise RuntimeError(f"unsafe_to_resume: {key} mismatch")


def save_log(path: Path, rows: list) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def train_epoch(model, optimizer, batches, seed: int, epoch: int, device) -> tuple[float, float]:
    model.train()
    order = np.random.default_rng(seed + epoch).permutation(len(batches))
    losses, gradients = [], []
    for start in range(0, len(order), BATCH_SIZE):
        batch = concatenate_batches([batches[index] for index in order[start:start + BATCH_SIZE]]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = batch_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)))
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)), float(np.mean(gradients))


@torch.no_grad()
def validation_loss(model, batch, device) -> float:
    model.eval()
    loss, _ = batch_loss(model, batch.to(device))
    return float(loss.detach())


def selection_phase(task, outdir, selection, inner, device, test_limits=None):
    seed = int(task["seed"])
    max_epochs = int((test_limits or {}).get("max_epochs", MAX_EPOCHS))
    min_epochs = int((test_limits or {}).get("min_epochs", MIN_EPOCHS))
    patience_limit = int((test_limits or {}).get("patience", PATIENCE))
    model = build_model(task).to(device)
    optimizer = optimizer_for(model, task["model_id"])
    last_path = outdir / "checkpoint_last.pt"
    best_path = outdir / "checkpoint_best_inner_validation.pt"
    start_epoch, best_loss, best_epoch, counter, logs = 1, float("inf"), 0, 0, []
    if last_path.exists():
        state = torch.load(last_path, map_location=device)
        validate_checkpoint(task, state)
        if state["training_phase"] == "epoch_selection":
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            restore_rng(state["rng"])
            start_epoch = int(state["epoch"]) + 1
            best_loss, best_epoch, counter, logs = float(state["best_inner_validation_loss"]), int(state["best_epoch"]), int(state["early_stopping_counter"]), list(state["logs"])
        elif state["training_phase"] in {"selection_complete", "final_refit"}:
            return int(state["best_epoch"]), logs
    for epoch in range(start_epoch, max_epochs + 1):
        train_loss, gradient_norm = train_epoch(model, optimizer, selection, seed, epoch, device)
        inner_loss = validation_loss(model, inner, device)
        improved = not np.isfinite(best_loss) or inner_loss < best_loss * (1.0 - MIN_RELATIVE_IMPROVEMENT)
        if improved:
            best_loss, best_epoch, counter = inner_loss, epoch, 0
            atomic_torch(best_path, {"model_state_dict": model.state_dict(), "best_epoch": best_epoch, "best_inner_validation_loss": best_loss, "config_sha256": task["config_sha256"]})
        else:
            counter += 1
        logs.append({"epoch": epoch, "train_loss": train_loss, "inner_validation_loss": inner_loss, "gradient_norm": gradient_norm, "improved": improved, "early_stopping_counter": counter})
        state = checkpoint_payload(task, "epoch_selection", epoch, model, optimizer, best_loss, best_epoch, counter, logs)
        atomic_torch(last_path, state)
        save_log(outdir / "epoch_selection_log.csv", logs)
        atomic_json(outdir / "run_status.json", {"task_id": task["full_run_task_id"], "status": "running_epoch_selection", "epoch": epoch, "best_epoch": best_epoch, "best_inner_validation_loss": best_loss, "updated": now()})
        if epoch >= min_epochs and counter >= patience_limit:
            break
    if best_epoch < 1:
        raise RuntimeError("early stopping did not select an epoch")
    state = checkpoint_payload(task, "selection_complete", len(logs), model, optimizer, best_loss, best_epoch, counter, logs)
    atomic_torch(last_path, state)
    return best_epoch, logs


def refit_phase(task, outdir, refit, best_epoch, device):
    seed = int(task["seed"])
    seed_all(seed)
    model = build_model(task).to(device)
    optimizer = optimizer_for(model, task["model_id"])
    last_path = outdir / "checkpoint_last.pt"
    start_epoch, logs = 1, []
    if last_path.exists():
        state = torch.load(last_path, map_location=device)
        validate_checkpoint(task, state)
        if state["training_phase"] == "final_refit":
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            restore_rng(state["rng"])
            start_epoch, logs = int(state["epoch"]) + 1, list(state["logs"])
    for epoch in range(start_epoch, best_epoch + 1):
        train_loss, gradient_norm = train_epoch(model, optimizer, refit, seed, epoch, device)
        logs.append({"epoch": epoch, "train_loss": train_loss, "gradient_norm": gradient_norm})
        state = checkpoint_payload(task, "final_refit", epoch, model, optimizer, None, best_epoch, 0, logs)
        atomic_torch(last_path, state)
        save_log(outdir / "final_refit_log.csv", logs)
        atomic_json(outdir / "run_status.json", {"task_id": task["full_run_task_id"], "status": "running_final_refit", "epoch": epoch, "selected_epoch": best_epoch, "updated": now()})
    atomic_torch(outdir / "checkpoint_final.pt", checkpoint_payload(task, "final_refit", best_epoch, model, optimizer, None, best_epoch, 0, logs))
    return model, logs


def prediction_products(task, outdir, model, formal, device):
    model.eval()
    started = time.perf_counter()
    output = model.predict(formal.to(device))
    inference_seconds = time.perf_counter() - started
    prediction = np.clip(np.expm1(output.prediction.detach().cpu().numpy()[0]), 0, None)
    actual = np.clip(np.expm1(formal.target.detach().cpu().numpy()[0]), 0, None)
    scales = rmsse_scales(dataset_key(task["dataset"]), task["forecast_origin"])
    dates = pd.date_range(task["forecast_origin"], periods=28, freq="D")
    groups = generic_groups(list(map(str, formal.node_ids))).set_index("node_id")
    records = []
    for horizon in range(28):
        for node, node_id in enumerate(map(str, formal.node_ids)):
            records.append({
                "experiment_version": "FULL_UNIFIED_RUN_V1", "execution_stage": "stage_1",
                "dataset": task["dataset"], "forecast_origin": task["forecast_origin"], "seed": int(task["seed"]),
                "model_id": task["model_id"], "node_id": node_id, "node_index": node,
                "item_block": groups.loc[node_id, "item_block"], "store_slot": groups.loc[node_id, "store_slot"],
                "target_date": str(dates[horizon].date()), "horizon_step": horizon + 1,
                "actual": float(actual[horizon, node]), "prediction": float(prediction[horizon, node]),
                "rmsse_scale_denominator": float(scales[node]),
            })
    frame = pd.DataFrame(records)
    frame.to_csv(outdir / "predictions.csv.gz", index=False, compression="gzip")
    error = frame.actual - frame.prediction
    frame["absolute_error"] = error.abs()
    frame["squared_error"] = error ** 2
    node = frame.groupby(["node_id", "node_index", "item_block", "store_slot"], as_index=False).agg(MAE=("absolute_error", "mean"), squared_error=("squared_error", "mean"), actual_sum=("actual", "sum"), absolute_error_sum=("absolute_error", "sum"), rmsse_scale_denominator=("rmsse_scale_denominator", "first"))
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
    atomic_json(outdir / "metric_results.json", metrics)
    graph = output.learned_graph
    atomic_json(outdir / "graph_diagnostics.json", graph_statistics(graph) if graph is not None else {"not_applicable": True})
    atomic_json(outdir / "input_usage.json", {"used_fields": output.diagnostics.get("used_fields", []), "target_exposed_to_forward": False})
    return output, metrics, inference_seconds


def reload_check(task, outdir, formal, prediction, device):
    state = torch.load(outdir / "checkpoint_final.pt", map_location=device)
    validate_checkpoint(task, state)
    model = build_model(task).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    reloaded = model.predict(formal.to(device)).prediction.detach().cpu()
    passed = bool(torch.allclose(prediction.prediction.detach().cpu(), reloaded, atol=1e-6, rtol=1e-5))
    atomic_json(outdir / "checkpoint_reload_check.json", {"passed": passed, "training_phase": state["training_phase"], "epoch": state["epoch"], "config_sha256": state["config_sha256"]})
    if not passed:
        raise RuntimeError("final checkpoint reload prediction mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-root", help="test-only isolated output root; never use for formal stage outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-max-epochs", type=int)
    parser.add_argument("--test-min-epochs", type=int)
    parser.add_argument("--test-patience", type=int)
    args = parser.parse_args()
    task = read_task(args.task_id)
    run_root = Path(args.output_root) if args.output_root else RUN_ROOT
    outdir = run_root / task["full_run_task_id"]
    outdir.mkdir(parents=True, exist_ok=True)
    marker = outdir / "completion.marker"
    if marker.exists():
        print(json.dumps({"task_id": args.task_id, "status": "passed", "resumed": True}))
        return
    if task["data_slice_sha256"] == "RESOLVE_DURING_PREFLIGHT" or task["node_ids_sha256"] == "RESOLVE_DURING_PREFLIGHT":
        raise RuntimeError("formal manifest data hashes are not resolved")
    started = now()
    start_clock = time.perf_counter()
    atomic_json(outdir / "resolved_config.json", task)
    atomic_json(outdir / "run_status.json", {"task_id": args.task_id, "status": "running_epoch_selection", "start_time": started})
    try:
        dataset = dataset_key(task["dataset"])
        selection, inner, refit, formal = load_formal_batches(dataset, task["forecast_origin"])
        selection_origins, inner_origin, refit_origins = formal_split(dataset, task["forecast_origin"])
        leakage = pd.Timestamp(inner_origin) + pd.Timedelta(days=27) < pd.Timestamp(task["forecast_origin"])
        if not leakage:
            raise RuntimeError("inner validation leakage detected")
        device = torch.device(args.device)
        seed_all(int(task["seed"]))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        limits = None
        if args.test_max_epochs is not None:
            limits = {"max_epochs": args.test_max_epochs, "min_epochs": args.test_min_epochs or 1, "patience": args.test_patience or 1}
        best_epoch, selection_logs = selection_phase(task, outdir, selection, inner, device, limits)
        model, refit_logs = refit_phase(task, outdir, refit, best_epoch, device)
        output, metrics, inference_seconds = prediction_products(task, outdir, model, formal, device)
        reload_check(task, outdir, formal, output, device)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
        resource = {"parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "elapsed_seconds": time.perf_counter() - start_clock, "inference_seconds": inference_seconds, "peak_gpu_mib": peak, "within_5_2_gib": peak < 5.2 * 1024, "batch_size": BATCH_SIZE}
        atomic_json(outdir / "resource_profile.json", resource)
        status = {"task_id": args.task_id, "dataset": task["dataset"], "forecast_origin": task["forecast_origin"], "seed": int(task["seed"]), "model_id": task["model_id"], "status": "passed", "start_time": started, "end_time": now(), "selected_epoch": best_epoch, "selection_epochs_completed": len(selection_logs), "refit_epochs_completed": len(refit_logs), "inner_validation_origin": inner_origin, "formal_origin": task["forecast_origin"], "leakage_check_passed": leakage, "checkpoint_reload_passed": True, "prediction_shape_passed": list(output.prediction.shape) == [1, 28, 90], "finite_output_passed": bool(torch.isfinite(output.prediction).all()), "graph_export_passed": output.learned_graph is not None or task["graph_mode"] in {"none", "no_graph"}, "peak_gpu_mib": peak, **metrics}
        atomic_json(outdir / "run_status.json", status)
        atomic_json(marker, {"task_id": args.task_id, "config_sha256": task["config_sha256"], "data_slice_sha256": task["data_slice_sha256"], "completed": now()})
        print(json.dumps(status, ensure_ascii=False))
    except KeyboardInterrupt:
        atomic_json(outdir / "run_status.json", {"task_id": args.task_id, "status": "interrupted", "updated": now(), "safe_checkpoint_exists": (outdir / "checkpoint_last.pt").exists()})
        raise
    except Exception as exc:
        (outdir / "exception.txt").write_text(repr(exc) + "\n" + traceback.format_exc(), encoding="utf-8")
        atomic_json(outdir / "run_status.json", {"task_id": args.task_id, "status": "failed", "failure_stage": "formal_v1_task", "exception_type": type(exc).__name__, "exception_summary": str(exc), "updated": now(), "safe_checkpoint_exists": (outdir / "checkpoint_last.pt").exists()})
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
