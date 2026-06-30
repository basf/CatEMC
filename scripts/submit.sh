#!/bin/bash
# Example PBS submission script for accelerated CatEMC WorkerPool runs.
# Override variables with qsub -v, e.g.
# qsub -v RUN_SCRIPT=gcemc_run_fast.py,GPU_DEVICES=0,WORKERS_PER_GPU=36,TRAJ_WRITE_INTERVAL=50,SCRATCH_DIR=/mnt/scratch/$USER/rhmno_gc,VERBOSE=0 submit.sh

#PBS -l select=1:ncpus=40:ngpus=1:mem=320gb:mpiprocs=1:ompthreads=1
#PBS -l walltime=167:00:00
#PBS -N CatEMC_fast

set -u

: "${RUN_SCRIPT:=gcemc_run_fast.py}"          # or geemc_run_fast.py
: "${GPU_DEVICES:=0}"
: "${WORKERS_PER_GPU:=36}"
: "${TRAJ_WRITE_INTERVAL:=50}"                # write emc_*.traj every N generations
: "${SCRATCH_DIR:=/mnt/scratch/${USER}/catemc_${PBS_JOBID:-manual}}"
: "${SCRATCH_SYNC_INTERVAL:=0}"               # 0/None = sync DB only at clean exit; N = also copy every N generations
: "${VERBOSE:=0}"                             # 1 = old detailed output, 0 = concise summaries
: "${CATEMC_OUT:=emc.out}"
: "${CATEMC_ERR:=emc.err}"

mkdir -p "$SCRATCH_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Optional site-specific environment setup.  Examples:
# module purge
# module load gcc/12.4.0 openmpi/4.1.7 cuda/12.2
# source /path/to/miniforge3/etc/profile.d/conda.sh
# conda activate /path/to/env

# Keep your CatEMC checkout importable when submitting from scripts/.
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"

# CUDA MPS helps many worker processes share one GPU.  Ignore failures on systems
# where MPS is unavailable.
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    nvidia-cuda-mps-control -d || true
fi

python -u "$RUN_SCRIPT" "$GPU_DEVICES" "$WORKERS_PER_GPU" "$TRAJ_WRITE_INTERVAL" "$SCRATCH_DIR" "$SCRATCH_SYNC_INTERVAL" "$VERBOSE" \
    >>"$CATEMC_OUT" 2>"$CATEMC_ERR"
EXIT_CODE=$?

# Crash recovery: if Python failed before clean scratch sync, copy the scratch DB
# back to persistent storage under *_recovered.db.
if [[ $EXIT_CODE -ne 0 && -f .emc_scratch_info ]]; then
    # shellcheck disable=SC1091
    source .emc_scratch_info
    echo "[CRASH-RECOVERY] Python exited with code $EXIT_CODE" >>"$CATEMC_OUT"
    if [[ -f "$SCRATCH_DB" ]]; then
        cp "$SCRATCH_DB" "${PERSISTENT_DB%.db}_recovered.db"
        echo "[CRASH-RECOVERY] Copied $SCRATCH_DB to ${PERSISTENT_DB%.db}_recovered.db" >>"$CATEMC_OUT"
    fi
fi

exit $EXIT_CODE
