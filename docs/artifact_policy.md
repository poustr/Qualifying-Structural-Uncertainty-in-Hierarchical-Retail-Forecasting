# Artifact policy

The Git repository contains only small derived v1 tables, manifests, configurations, documentation, code, and the one released identity graph matrix needed for fast graph-reliability recomputation. It contains no raw data, processed panels, transaction records, predictions, other graph arrays, checkpoints, caches, logs, stack traces, credentials, or third-party source trees.

The local formal archive retains checkpoints: 640 final checkpoints and 1,922 checkpoint files overall. They are not in Git. The 639 formal final checkpoints are planned as externally hosted replay assets and are catalogued by SHA-256 in the checkpoint manifest. Any future model or prediction asset must be versioned outside Git history and linked to its v1 task/configuration/data hashes.
