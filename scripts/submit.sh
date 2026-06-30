#!/bin/bash
#SBATCH --job-name=catemc
# --array=0-1                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=kernph-rtxpro6000,kernph-rtx6000
#SBATCH --qos=kernph
#SBATCH --nodelist=pgd01,pgc01
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err

source /scicore/home/goedecke/han0011/.bashrc
conda activate /scicore/home/goedecke/han0011/venvs/general-gpu
export OMP_NUM_THREADS=8
cd $SLURM_SUBMIT_DIR

python -u gcemc_run.py 0 16 >> emc.out 2> pbs.err
