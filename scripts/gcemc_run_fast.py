#!/usr/bin/env python3
"""Surface GC-EMC run using persistent WorkerPool + optional scratch DB.

Usage:
    python gcemc_run_fast.py <gpu_devices> <workers_per_gpu> [traj_write_interval] [scratch_dir] [scratch_sync_interval] [verbose]

Example:
    CATEMC_MODEL_PATH=/path/model.pt CATEMC_CALC_KIND=metatomic \
    python -u gcemc_run_fast.py 0 36 50 /mnt/scratch/$USER/rhmno_gcemc 0 0

verbose=1 keeps the original detailed per-move output. verbose=0 prints only
per-generation summaries, including acceptance rates.
"""
from catemc.gcemc import GCEMC
from catemc.worker_pool import WorkerPool
from ase import data, units
from ase.io import read
from acat.adsorption_sites import SlabAdsorptionSites
from collections import Counter
import numpy as np
import pickle
import math
import sys
import os

try:
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
except Exception:
    pass


def parse_optional_arg(i, default=None):
    if len(sys.argv) <= i:
        return default
    value = sys.argv[i]
    if value in {"None", "none", "", "0"} and default is None:
        return None
    return value


gpu_devices = [int(di) for di in sys.argv[1].split(',')]
workers_per_gpu = int(sys.argv[2])
traj_write_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 50
scratch_dir = parse_optional_arg(4, None)
scratch_sync_interval = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in {"None", "none", "", "0"} else None
verbose = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True

# Keep the parent process on the first visible GPU; workers set their own device.
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])

# -----------------------------------------------------------------------------
# User-editable system block.  This mirrors your current Rh13_MnO(100) GC-EMC
# example, but paths are local to the script directory by default.
# -----------------------------------------------------------------------------
comp = os.environ.get('CATEMC_COMP', 'Rh13_MnO')
surface = os.environ.get('CATEMC_SURFACE', '100')
system_name = os.environ.get('CATEMC_SYSTEM_NAME', f'{comp}COH_{surface}')
dir_name = os.environ.get('CATEMC_RUN_DIR', os.path.dirname(os.path.abspath(__file__)))
input_pop_fname = os.environ.get('CATEMC_INPUT_POP', f'{dir_name}/{comp}COH_{surface}_input_pop.traj')
template_fname = os.environ.get('CATEMC_TEMPLATE', f'{dir_name}/Rh13_MnO_100_sym_0.extxyz')

input_pop = read(input_pop_fname, index=':')
template = read(template_fname)

target_metal_species = ['Mn', 'Rh']
target_metal_composition = {'Mn': 0.9326424870466321, 'Rh': 0.06735751295336788}
target_nonmetal_species = ['O', 'X']
adsorbate_species = ['O', 'H', 'OH', 'C', 'CO', 'CO2', 'CH', 'CH2', 'CH3', 'CHO', 'COH', 'CH3O']
other_adsorbate_species = ['OH2', 'H2', 'O2', 'O2H', 'OOH', 'OC', 'OCH3']
allowed_adsorbate_species = sorted(set(adsorbate_species + other_adsorbate_species))

slab_indices = [a.index for a in template]
z_fix = template.cell[2][2] * 0.5
fix_indices = [a.index for a in template if a.position[2] < z_fix]
mutable_metal_indices = [a.index for a in template if (a.position[2] >= z_fix) and (a.symbol in target_metal_species)]
mutable_nonmetal_indices = [a.index for a in template if (a.position[2] >= z_fix) and (a.symbol not in target_metal_species)]

volume = np.linalg.norm(np.cross(template.cell[0], template.cell[1])) * (11 + 3 + 2.5)
ct = Counter(template.symbols)
volume -= sum(n * 4 / 3 * math.pi * (data.covalent_radii[data.atomic_numbers[k]]) ** 3
              for k, n in ct.items() if k in target_metal_species)

# Reference energies; edit for the functional/model you use.
E_C = -9.944
E_O2 = -8.768 + 0.101
E_H2 = -6.773 + 0.267

dmu_C = np.arange(1.0, 3.001, 0.1)
dmu_O = np.arange(-4.0, -2.999, 0.1)
dmu_H = np.arange(-1.0, 0.001, 0.1)
temperatures = sorted({atoms.info['key_value_pairs']['temperature'] for atoms in input_pop})
reaction_conditions = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H, temperatures)).T.reshape(-1, 4)
element_column_map = {'C': 0, 'O': 1, 'H': 2}
element_eref_map = {'C': E_C, 'O': E_O2 / 2, 'H': E_H2 / 2}
element_y2x_map = {'C': 0.0, 'O': 1.0, 'H': 0.0}

temperature_dmu_map = None
internal_data_path = os.environ.get('CATEMC_INTERNAL_DATA', f'{dir_name}/COH_internal_data.pkl')
with open(internal_data_path, 'rb') as handle:
    temperature_species_internal_map = pickle.load(handle)

sites_path = os.environ.get('CATEMC_SITES_PKL', f'{dir_name}/{comp}_{surface}_sites.pkl')
if not os.path.isfile(sites_path):
    sas = SlabAdsorptionSites(template, surface='autoadsorbate', composition_effect=False, ignore_sites='bridge')
    with open(sites_path, 'wb') as handle:
        pickle.dump(sas, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------------------------------------------------------
# Persistent workers: calculators are loaded once per worker, not once per move.
# fmax_list must match whatever create_calculators() returns.  The example
# factory prepends a RepulsivePotential by default, so [0.5, 0.05] is typical.
# -----------------------------------------------------------------------------
fitness_kwargs = {
    'fmax_list': [0.5, 0.05],
    'target_metal_species': target_metal_species,
    'target_metal_composition': target_metal_composition,
    'allowed_adsorbate_species': allowed_adsorbate_species,
    'element_column_map': element_column_map,
    'element_eref_map': element_eref_map,
    'element_y2x_map': element_y2x_map,
    'adsorption_sites': sites_path,
    'md_timestep': 1.0 * units.fs,
    'md_taut': 500.0 * units.fs,
    'md_steps': 50,
    'md_write_interval': 10,
    'temperature_dmu_map': temperature_dmu_map,
    'max_relax_steps': 300,
    'adsorbate_surface_dmax': 2.5,
    'neighbor_cutoff': 2.5,
    'update_relaxed_positions': False,
    'raw_score_cutoff': -10000.0,
    'temperature_key': 'temperature',
    'replica_key': 'replica',
    'ignore_sites': 'bridge',
}

sys.path.insert(0, dir_name)
pool = WorkerPool(
    calculator_factory_module=os.environ.get('CATEMC_CALC_FACTORY_MODULE', 'calc_factory_example'),
    calculator_factory_func=os.environ.get('CATEMC_CALC_FACTORY_FUNC', 'create_calculators'),
    calculator_factory_kwargs={},
    fitness_func_module='catemc.calculation_templates',
    fitness_func_name='calculate_surface_fitness_inprocess',
    fitness_kwargs=fitness_kwargs,
    num_workers=workers_per_gpu * len(gpu_devices),
    gpu_devices=gpu_devices,
    startup_timeout=600.0,
)

try:
    emc = GCEMC(
        system_name,
        calculation_dir=dir_name,
        input_pop=input_pop,
        fix_indices=fix_indices,
        trim_generation=int(os.environ.get('CATEMC_TRIM_GENERATION', 100)),
        md_generation=10,
        mu_walk_generation=1,
        exchange_generation=1,
        bias_starting_step=300,
        mbar_starting_step=100,
        mbar_sample_stride=10,
        crossover_rate=0.1,
        interswap_rate=0.0,
        slab_intraswap_rate=0.4,
        slab_mutation_rate=0.0,
        adsorbate_insertion_rate=0.125,
        adsorbate_removal_rate=0.125,
        adsorbate_displacement_rate=0.125,
        adsorbate_mutation_rate=0.125,
        adsorbate_surface_dmax=2.5,
        adsorbate_adsorbate_dmin=2.0,
        max_adsorbate_species=3,
        max_adsorbate_coverage=1.0,
        metal_element_pool=target_metal_species,
        nonmetal_element_pool=target_nonmetal_species,
        bulk_metal_composition=target_metal_composition,
        adsorbate_species=adsorbate_species,
        other_adsorbate_species=other_adsorbate_species,
        mutable_metal_indices=mutable_metal_indices,
        mutable_nonmetal_indices=mutable_nonmetal_indices,
        slab_indices=slab_indices,
        volume=volume,
        reaction_conditions=reaction_conditions,
        element_column_map=element_column_map,
        element_eref_map=element_eref_map,
        element_y2x_map=element_y2x_map,
        temperature_dmu_map=temperature_dmu_map,
        temperature_species_internal_map=temperature_species_internal_map,
        adsorption_sites=sites_path,
        ignore_sites='bridge',
        temperature_key='temperature',
        replica_key='replica',
        acceptance_key='accepted',
        raw_score_cutoff=-10000.0,
        gpu_devices=gpu_devices,
        procs_per_device=workers_per_gpu,
        worker_pool=pool,
        traj_write_interval=traj_write_interval,
        scratch_dir=scratch_dir,
        scratch_sync_interval=scratch_sync_interval,
        verbose=verbose,
    )
    emc.run(starting_step=int(os.environ.get('CATEMC_STARTING_STEP', 1)),
            total_steps=int(os.environ.get('CATEMC_TOTAL_STEPS', 4000)))
finally:
    pool.shutdown()
