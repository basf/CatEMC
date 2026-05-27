#!/bin/bash
#PBS -N sweep_workers
#PBS -l select=1:ncpus=40:ngpus=1:mem=64gb
#PBS -l walltime=06:00:00
#PBS -q azure_gpuq
#PBS -j oe
#PBS -o sweep_workers.out

cd $PBS_O_WORKDIR

source /usr/share/lmod/lmod/init/bash 2>/dev/null
module purge
module unuse /gpfs/software/milan9/v2.1/lmod/Core
module use /gpfs/software/azure9/v2.1/lmod/Core
module load gcc/12.4.0
export LD_PRELOAD=$(dirname $(which g++))/../lib64/libstdc++.so.6
eval "$(/fs/cephfs/data/qm_inorganics/azure-conda/miniforge3/bin/conda shell.bash hook)"
conda activate /fs/cephfs/data/qm_inorganics/venvs/nequip-h100
export TMPDIR=/fs/cephfs/data/qm_inorganics/venvs/.tmp_desa

nvidia-cuda-mps-control -d 2>/dev/null

python -u sweep_workers.py
