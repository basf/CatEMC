#!/bin/bash
#PBS -l select=1:ncpus=40:ngpus=1:mem=320gb:mpiprocs=1:ompthreads=1:vm_size=Standard_NC40ads_H100_v5
#PBS -l walltime=01:00:00
#PBS -q azure_gpuq
#PBS -N fs_bench_48w
#PBS -P OER

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC:${PYTHONPATH}"

cd $PBS_O_WORKDIR
nvidia-cuda-mps-control -d

python -u scratch_fs_test_48w.py >>scratch_fs_48w.out 2>scratch_fs_48w.err
