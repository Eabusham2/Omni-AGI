#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: scripts/train-distributed.sh DATASET OUTPUT [torchrun/train options...]" >&2
  exit 2
fi

dataset_path=$1
output_path=$2
shift 2
python_bin=${OMNI_PYTHON:-python3}
processes=${OMNI_GPU_PROCESSES:-gpu}

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}engine"
exec "$python_bin" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "$processes" \
  engine/distributed_train.py train \
  --dataset "$dataset_path" \
  --output "$output_path" \
  "$@"
