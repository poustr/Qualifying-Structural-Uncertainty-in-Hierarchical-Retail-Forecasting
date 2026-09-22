# v1 experiment protocol

`unified_rerun_v1` is the sole formal result set released here. The protocol locks M5, Corporacion Favorita, and Store Item; 90 bottom-level series; 84 historical input days; 28 forecast days; three origins per data set; neural seeds 13, 29, and 47; RMSSE, WAPE, and MAE; log1p Huber training; early stopping; refit at the selected epoch; and serial single-GPU execution.

The formal task manifest contains 639 tasks: 351 in stage 1 and 288 in stage 2. Store Item semantic tasks are not applicable. The MAGE placebo control is documented as approximate. See `configs/v1/FULL_UNIFIED_RUN_V1_LOCK.json` and the released task manifests for exact model rosters and settings.

This public release does not reclassify earlier validation, independent-test, post-hoc, simulation, screening, or mechanism-ablation materials as v1 evidence.
