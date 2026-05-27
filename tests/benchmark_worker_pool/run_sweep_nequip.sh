#!/bin/bash
# NequIP sweep only
set -e
cd "$(dirname "$0")"

export PATH=/mnt/nfs/venvs/allegro/bin:$PATH
export LD_PRELOAD=/mnt/nfs/venvs/allegro/lib/libstdc++.so.6
export BENCH_MODEL=nequip

echo ">>> NequIP sweep: $(date)"
echo "Python: $(which python)"
python sweep_all_real.py 2>&1
echo ">>> NequIP done: $(date)"
