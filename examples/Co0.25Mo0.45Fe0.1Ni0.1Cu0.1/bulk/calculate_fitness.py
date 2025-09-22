from catemc.calculation_templates import calculate_bulk_fitness
from catemc.repulsive import RepulsivePotential 
from mace.calculators.mace import MACECalculator
from ase import units
from ase.io import read, write
import numpy as np
import sys
import os


#======================Don't change this=====================
traj_path = sys.argv[1]
run_md = bool(int(sys.argv[2]))
gpu_device = sys.argv[3]
if sys.argv[4] == 'None':
    fix_indices = None
else:
    fix_indices = [int(i) for i in sys.argv[4].split(',')]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
dir_name = '/'.join(traj_path.split('/')[:-1])
atoms = read(traj_path, index=-1)
atoms.wrap()
#============================================================

calc = MACECalculator(model_paths="/mnt/nfs/emc_tutorial/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/models/CoMoFeNiCuNH_mace.model", 
                      dispersion=False, device='cuda', default_dtype='float64')
fmax = 0.05

calculate_bulk_fitness(
        atoms, traj_path=traj_path,
        calculator_list=[calc],
        fmax_list=[fmax],
        target_metal_species=['Co','Mo','Fe','Ni','Cu'],
        run_md=run_md,
        md_timestep=1*units.fs,
        md_taut=0.5*1000**units.fs,
        md_steps=400,
        md_write_interval=10,
        pressure=None,
        fix_indices=fix_indices,
        neighbor_cutoff=3.5,
        raw_score_cutoff=-10000.,
        temperature_key='temperature',
        replica_key='replica')
