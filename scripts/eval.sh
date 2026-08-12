#!/usr/bin/env bash
# Evaluate a trained checkpoint on a split's src.test (in-dist) + tgt.test (OOD).
# geom_mode + conditioning are read from the checkpoint's meta block. Usage:
#   scripts/eval.sh <checkpoint.pt> <data_dir> <splits.json> [difficulty]
set -euo pipefail
CKPT=${1:?checkpoint.pt}; DATA=${2:?data dir}; SPLITS=${3:?splits.json}; DIFF=${4:-medium}

python -m kuber.train_simshift \
    --data "$DATA" --splits "$SPLITS" --difficulty "$DIFF" \
    --eval_only "$CKPT"

# Stability / no-explosion proof (CPU by default):
python -m kuber.edge_proof \
    --ckpt "$CKPT" --data "$DATA" --splits "$SPLITS" --difficulty "$DIFF" \
    --out "edge_proof_${DIFF}.json"
