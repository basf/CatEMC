#!/bin/bash
#PBS -N gpu_determ_5x
#PBS -l select=1:ncpus=40:ngpus=1:mem=64gb
#PBS -l walltime=01:30:00
#PBS -q azure_gpuq
#PBS -o /fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC/tests/benchmark_io_reduction/gpu_determinism.out
#PBS -e /fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC/tests/benchmark_io_reduction/gpu_determinism.err
#PBS -j oe

source /usr/share/lmod/lmod/init/bash 2>/dev/null
module purge
module unuse /gpfs/software/milan9/v2.1/lmod/Core 2>/dev/null
module use /gpfs/software/azure9/v2.1/lmod/Core
module load gcc/12.4.0
export LD_PRELOAD=$(dirname $(which g++))/../lib64/libstdc++.so.6

eval "$(/fs/cephfs/data/qm_inorganics/azure-conda/miniforge3/bin/conda shell.bash hook)"
conda activate /fs/cephfs/data/qm_inorganics/venvs/nequip-h100

export TMPDIR=/fs/cephfs/data/qm_inorganics/venvs/.tmp_desa
nvidia-cuda-mps-control -d 2>/dev/null

cd /fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC/tests/benchmark_io_reduction
python validate_gpu_determinism.py
