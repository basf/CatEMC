#!/bin/bash
#PBS -N scratch_dir_test
#PBS -l select=1:ncpus=40:ngpus=1:mem=320gb:vm_size=Standard_NC40ads_H100_v5
#PBS -l walltime=00:30:00
#PBS -q azure_gpuq
#PBS -o /fs/cephfs/data/qm_inorganics/sandip/emc_scratch_dir_test/test.out
#PBS -e /fs/cephfs/data/qm_inorganics/sandip/emc_scratch_dir_test/test.err

module purge
module unuse /gpfs/software/milan9/v2.1/lmod/Core
module use /gpfs/software/azure9/v2.1/lmod/Core
module load gcc/12.4.0 openmpi/4.1.7

export TMPDIR=/fs/cephfs/data/qm_inorganics/venvs/.tmp
mkdir -p $TMPDIR

export CUDA_HOME=/fs/cephfs/software/azure9/v2.1/apps/linux-rhel9-zen3/gcc-12.4.0/cuda-12.2.2-4sdxvaer7ivh5k5gmlat6vfrfj6tanpg
export PATH=$CUDA_HOME/bin:$PATH
export LD_PRELOAD=$(dirname $(which g++))/../lib64/libstdc++.so.6

eval "$(/fs/cephfs/data/qm_inorganics/azure-conda/miniforge3/bin/conda shell.bash hook)"
conda activate /fs/cephfs/data/qm_inorganics/venvs/nequip-h100

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC:${PYTHONPATH}"

cd $PBS_O_WORKDIR
mkdir -p /fs/cephfs/data/qm_inorganics/sandip/emc_scratch_dir_test

nvidia-cuda-mps-control -d

python -u test_scratch_dir.py 2>&1 | tee /fs/cephfs/data/qm_inorganics/sandip/emc_scratch_dir_test/run.log

echo quit | nvidia-cuda-mps-control
