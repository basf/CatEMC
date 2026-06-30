#!/usr/bin/env python3

from catemc.gcemc import GCEMC
from calculate_fitness import get_fitness_kwargs
from ase import data
from ase.io import read, write
from ase.geometry import get_layers
from acat.adsorption_sites import SlabAdsorptionSites
from collections import Counter
import numpy as np
import pickle
import math
import sys
import os

gpu_devices = [int(di) for di in sys.argv[1].split(',')]
ppg = int(sys.argv[2])
traj_write_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 1
scratch_dir = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != 'None' else None
scratch_sync_interval = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != 'None' else None
verbose = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])
comp = 'Rh13_MnO'
surface = '100'
system_name = f'{comp}COH_{surface}'
dir_name = f'/scicore/home/goedecke/han0011/projects/RhMnO/{comp}/pet_{surface}'
input_pop_fname = f'{dir_name}/{comp}COH_{surface}_input_pop.traj'
input_pop = read(input_pop_fname, index=':')
template = read(f'{dir_name}/Rh13_MnO_100_sym_0.extxyz')
target_metal_species = ['Mn','Rh']
target_metal_composition = {'Mn': 0.9326424870466321, 'Rh': 0.06735751295336788}
target_nonmetal_species = ['O','X']
adsorbate_species = ['O','H','OH','C','CO','CO2','CH','CH2','CH3','CHO','COH','CH3O']
other_adsorbate_species = ['OH2','H2','O2','O2H','OOH','OC','OCH3']
slab_indices = [a.index for a in template]
z_fix = template.cell[2][2] * 0.5
fix_indices = [a.index for a in template if a.position[2] < z_fix]
mut_metal_indices = [a.index for a in template if (a.position[2] >= z_fix) and (a.symbol in target_metal_species)]
mut_nonmetal_indices = [a.index for a in template if (a.position[2] >= z_fix) and (a.symbol not in target_metal_species)]
volume = np.linalg.norm(np.cross(template.cell[0], template.cell[1])) * (11 + 3 + 2.5)
ct = Counter(template.symbols)
volume -= sum([n*4/3*math.pi*(data.covalent_radii[data.atomic_numbers[k]])**3
               for k, n in ct.items() if k in target_metal_species])

E_C = -9.944    # Etot per atom graphite PBE
E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE

dmu_C = np.arange(1, 3.001, 0.1)
dmu_O = np.arange(-4., -2.999, 0.1)
dmu_H = np.arange(-1.0, 0.001, 0.1)

temperatures = sorted({atoms.info['key_value_pairs']['temperature'] for atoms in input_pop})
reaction_conditions = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H, temperatures)).T.reshape(-1, 4)
element_column_map = {'C': 0, 'O': 1, 'H': 2}
element_eref_map = {'C': E_C, 'O': E_O2/2, 'H': E_H2/2}
element_y2x_map = {'C': 0., 'O': 1., 'H': 0.}

#with open(f'/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_bulk_wls_dmu_estimates.pkl', 'rb') as f:
#   tdmu_map = pickle.load(f)

tdmu_map = None

with open(f'{dir_name}/COH_internal_data.pkl', 'rb') as handle:
   tsi_map = pickle.load(handle)

sas = SlabAdsorptionSites(template, surface='autoadsorbate', 
                         composition_effect=False,
                         ignore_sites='bridge')

with open(f'{dir_name}/{comp}_{surface}_sites.pkl', 'wb') as handle:
   pickle.dump(sas, handle, protocol=pickle.HIGHEST_PROTOCOL)

emc = GCEMC(system_name,
           calculation_dir=dir_name,
           calculation_script='calculate_fitness.py',
           input_pop=input_pop,
           fix_indices=fix_indices,
           trim_generation=100,
           md_generation=10,
           mu_walk_generation=1,
           exchange_generation=1,
           bias_starting_step=300,
           mbar_starting_step=100,
           mbar_sample_stride=10,
           crossover_rate=0.1,
           interswap_rate=0.0,       # otherwise need wls
           slab_intraswap_rate=0.4,
           slab_mutation_rate=0.0,  # otherwise need wls
           adsorbate_insertion_rate=0.125,
           adsorbate_removal_rate=0.125,
           adsorbate_displacement_rate=0.125,
           adsorbate_mutation_rate=0.125,
           adsorbate_surface_dmax=2.5,
           adsorbate_adsorbate_dmin=2.,
           max_adsorbate_species=3,
           max_adsorbate_coverage=1.,
           metal_element_pool=target_metal_species, 
           nonmetal_element_pool=target_nonmetal_species,
           bulk_metal_composition=target_metal_composition,
           adsorbate_species=adsorbate_species,
           other_adsorbate_species=other_adsorbate_species,
           mutable_metal_indices=mut_metal_indices,
           mutable_nonmetal_indices=mut_nonmetal_indices,
           slab_indices=slab_indices,
           volume=volume,
           reaction_conditions=reaction_conditions,
           element_column_map=element_column_map,
           element_eref_map=element_eref_map,
           element_y2x_map=element_y2x_map,
           temperature_dmu_map=tdmu_map,
           temperature_species_internal_map=tsi_map,
           adsorption_sites=sas,
           ignore_sites='bridge',
           temperature_key='temperature',
           replica_key='replica',
           acceptance_key='accepted',
           raw_score_cutoff=-10000.,
           gpu_devices=gpu_devices,
           procs_per_device=ppg,
           use_worker_pool=True,
           fitness_kwargs=get_fitness_kwargs(),
           scratch_dir=scratch_dir,
           scratch_sync_interval=scratch_sync_interval,
           traj_write_interval=traj_write_interval,
           verbose=verbose)

emc.run(starting_step=1, total_steps=4000)
