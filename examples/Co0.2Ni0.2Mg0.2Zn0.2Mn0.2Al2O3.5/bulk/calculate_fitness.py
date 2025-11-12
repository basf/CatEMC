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

calc1 = RepulsivePotential()
fmax1 = 0.5
calc2 = MACECalculator(model_paths="/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/models/CoNiMgZnMnAlCOH_mace.model", 
                       dispersion=False, device='cuda', default_dtype='float64')
fmax2 = 0.05

calculate_bulk_fitness(
        atoms, traj_path=traj_path,
        calculator_list=[calc1, calc2],
        fmax_list=[fmax1, fmax2],
        target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
        run_md=run_md,
        md_timestep=1*units.fs,
        md_taut=0.5*1000**units.fs,
        md_steps=400,
        md_write_interval=10,
        pressure=0.,
        md_taup=1000*units.fs,
        fix_indices=fix_indices,
        neighbor_cutoff=3.5,
        raw_score_cutoff=-10000.,
        temperature_key='temperature',
        replica_key='replica')
