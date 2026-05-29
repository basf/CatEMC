#!/bin/bash
#PBS -N prod_test
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=01:00:00
#PBS -q azure_gpuq
#PBS -o prod_test.out
#PBS -e prod_test.err
#PBS -j oe

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

export PYTHONPATH="/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC:/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk:$PYTHONPATH"

python -u production_smoke_test.py
