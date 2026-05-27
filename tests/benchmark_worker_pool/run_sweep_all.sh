#!/bin/bash
# Run the full benchmark sweep for all 3 models sequentially.
# Usage: nohup bash run_sweep_all.sh > sweep_all.log 2>&1 &

set -e
cd "$(dirname "$0")"
SCRIPT="sweep_all_real.py"
LOG_DIR="$(pwd)"

echo "=========================================="
echo "Starting full benchmark sweep: $(date)"
echo "=========================================="

# --- MACE ---
echo ""
echo ">>> MACE: $(date)"
export PATH=/mnt/nfs/venvs/general-gpu/bin:$PATH
export LD_PRELOAD=/mnt/nfs/venvs/general-gpu/lib/libstdc++.so.6
export BENCH_MODEL=mace
python "$SCRIPT" 2>&1
echo ">>> MACE done: $(date)"

# --- PET ---
echo ""
echo ">>> PET: $(date)"
unset LD_PRELOAD
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /mnt/nfs/venvs/metatomic-bench
export LD_LIBRARY_PATH=/mnt/nfs/venvs/metatomic-bench/lib:/mnt/nfs/venvs/metatomic-bench/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
export BENCH_MODEL=pet
python "$SCRIPT" 2>&1
conda deactivate
echo ">>> PET done: $(date)"

# --- NequIP ---
echo ""
echo ">>> NequIP: $(date)"
export PATH=/mnt/nfs/venvs/allegro/bin:$PATH
export LD_PRELOAD=/mnt/nfs/venvs/allegro/lib/libstdc++.so.6
export BENCH_MODEL=nequip
python "$SCRIPT" 2>&1
echo ">>> NequIP done: $(date)"

echo ""
echo "=========================================="
echo "ALL DONE: $(date)"
echo "Results: ${LOG_DIR}/sweep_results.json"
echo "=========================================="
