#!/bin/bash
# PET sweep only - must be run from within the correct conda env
set -e
cd "$(dirname "$0")"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /mnt/nfs/venvs/metatomic-bench
export LD_LIBRARY_PATH=/mnt/nfs/venvs/metatomic-bench/lib:/mnt/nfs/venvs/metatomic-bench/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
export BENCH_MODEL=pet

echo ">>> PET sweep: $(date)"
echo "Python: $(which python)"
python sweep_all_real.py 2>&1
echo ">>> PET done: $(date)"
