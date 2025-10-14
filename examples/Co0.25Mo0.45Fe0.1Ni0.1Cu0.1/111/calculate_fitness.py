from catemc.calculation_templates import calculate_surface_fitness
from catemc.repulsive import RepulsivePotential
from mace.calculators.mace import MACECalculator
from ase import units
from ase.io import read, write
from ase.neighborlist import NeighborList
from ase.constraints import FixAtoms
from ase.optimize import BFGS, LBFGS
from ase.io.trajectory import Trajectory
from ase.md import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from acat.adsorption_sites import SlabAdsorptionSites
from acat.adsorbate_coverage import SlabAdsorbateCoverage
from acat.settings import adsorbate_elements
from itertools import combinations
import numpy as np
import pickle
import math
import sys
import os

#=====================Don't change this======================
traj_path = sys.argv[1]
run_md = bool(int(sys.argv[2]))
gpu_device = sys.argv[3]
if sys.argv[4] == 'None':
    fix_indices = None
else:
    fix_indices = [int(i) for i in sys.argv[4].split(',')]
if sys.argv[5] == 'None':
    slab_indices = None
else:
    slab_indices = [int(i) for i in sys.argv[5].split(',')]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
dir_name = '/'.join(traj_path.split('/')[:-1])
atoms = read(traj_path, index=-1)
atoms.wrap()
#============================================================

calc = MACECalculator(model_paths='/mnt/nfs/emc_tutorial/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/models/CoMoFeNiCuNH_mace.model', 
                       dispersion=False, device='cuda', default_dtype='float64')
fmax = 0.05
E_N2 = -16.293 + 0.155 # Etot incl. ZPE RPBE 
E_H2 = -6.984 + 0.269 # Etot incl. ZPE RPBE
rep = atoms.info['key_value_pairs']['replica']
site_path = f'{dir_name}/sites_{rep}.pkl'
try:
    with open(site_path, 'rb') as f:
        sas = pickle.load(f)
except:
    with open(f'/mnt/nfs/emc_tutorial/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/111/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1_111_sites.pkl', 'rb') as f:
        sas = pickle.load(f)
with open(f'/mnt/nfs/emc_tutorial/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1_bulk_dmu_estimates.pkl', 'rb') as f:
    tdmu_map = pickle.load(f)
allowed_adsorbate_species = ['H','N','NH','NH2','N2','H2','NH3']

calculate_surface_fitness(
        atoms, traj_path=traj_path,
        calculator_list=[calc],
        fmax_list=[fmax],
        target_metal_species=['Co','Mo','Fe','Ni','Cu'],
        target_metal_composition={'Co':0.25,'Mo':0.45,'Fe':0.1,'Ni':0.1,'Cu':0.1},
        allowed_adsorbate_species=allowed_adsorbate_species,
        run_md=run_md,
        md_timestep=1*units.fs,
        md_taut=0.5*1000**units.fs,
        md_steps=50,
        md_write_interval=10,
        temperature_dmu_map=tdmu_map,
        element_column_map={'N': 0, 'H': 1},
        element_eref_map={'N': E_N2/2, 'H': E_H2/2},
        element_y2x_map={'N': 0., 'H': 0.},
        adsorption_sites=sas,
        slab_indices=slab_indices, 
        fix_indices=fix_indices,
        adsorbate_surface_dmax=2.5,
        neighbor_cutoff=2.5,
        raw_score_cutoff=-10000.,
        temperature_key='temperature',
        replica_key='replica')
