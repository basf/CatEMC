#!/usr/bin/env python3
"""Bulk GE-EMC run using persistent WorkerPool + optional scratch DB.

Usage:
    python geemc_run_fast.py <gpu_devices> <workers_per_gpu> [traj_write_interval] [scratch_dir] [scratch_sync_interval] [verbose]

Example:
    CATEMC_MODEL_PATH=/path/model.nequip.pt2 CATEMC_CALC_KIND=nequip \
    python -u geemc_run_fast.py 0 36 50 /mnt/scratch/$USER/curuox_geemc 0 0
"""
from catemc.geemc import GEEMC
from catemc.worker_pool import WorkerPool
from ase import units
from ase.io import read
import os
import sys

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
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])

# -----------------------------------------------------------------------------
# User-editable bulk system block.
# -----------------------------------------------------------------------------
dir_name = os.environ.get('CATEMC_RUN_DIR', os.path.dirname(os.path.abspath(__file__)))
system_name = os.environ.get('CATEMC_SYSTEM_NAME', 'Cu0.1Ru0.9Ox_bulk')
input_pop_fname = os.environ.get('CATEMC_INPUT_POP', f'{dir_name}/{system_name}_input_pop.traj')
input_pop = read(input_pop_fname, index=':')

target_metal_species = os.environ.get('CATEMC_METALS', 'Cu,Ru').split(',')
target_nonmetal_species = os.environ.get('CATEMC_NONMETALS', 'O,X').split(',')

# If your population stores a key such as dopant_valence and swaps should only
# occur within that class, set CATEMC_INTERSWAP_WITHIN_KEY=dopant_valence.
interswap_within_key = os.environ.get('CATEMC_INTERSWAP_WITHIN_KEY') or None

# Persistent worker setup.  fmax_list must match calc_factory_example output.
fitness_kwargs = {
    'fmax_list': [0.5, 0.05] if os.environ.get('CATEMC_USE_REPULSIVE', '1') not in {'0', 'false', 'False'} else [0.05],
    'target_metal_species': target_metal_species,
    'md_timestep': 1.0 * units.fs,
    'md_taut': 500.0 * units.fs,
    'md_steps': int(os.environ.get('CATEMC_MD_STEPS', 200)),
    'md_write_interval': 10,
    'max_relax_steps': int(os.environ.get('CATEMC_MAX_RELAX_STEPS', 300)),
    'neighbor_cutoff': float(os.environ.get('CATEMC_NEIGHBOR_CUTOFF', 3.5)),
    'raw_score_cutoff': -10000.0,
    'temperature_key': 'temperature',
    'replica_key': 'replica',
    'pressure': None,
    'optimize_cell': False,
}

sys.path.insert(0, dir_name)
pool = WorkerPool(
    calculator_factory_module=os.environ.get('CATEMC_CALC_FACTORY_MODULE', 'calc_factory_example'),
    calculator_factory_func=os.environ.get('CATEMC_CALC_FACTORY_FUNC', 'create_calculators'),
    calculator_factory_kwargs={},
    fitness_func_module='catemc.calculation_templates',
    fitness_func_name='calculate_bulk_fitness_inprocess',
    fitness_kwargs=fitness_kwargs,
    num_workers=workers_per_gpu * len(gpu_devices),
    gpu_devices=gpu_devices,
    startup_timeout=600.0,
)

try:
    emc = GEEMC(
        system_name,
        calculation_dir=dir_name,
        input_pop=input_pop,
        fix_indices=None,
        trim_generation=int(os.environ.get('CATEMC_TRIM_GENERATION', 100)),
        pressure=None,  # fixed-cell GE-EMC; set and use optimize_cell in fitness_kwargs for NPT-like runs
        md_generation=10,
        exchange_generation=1,
        bias_starting_step=int(os.environ['CATEMC_BIAS_STARTING_STEP']) if os.environ.get('CATEMC_BIAS_STARTING_STEP') else None,
        crossover_rate=float(os.environ.get('CATEMC_CROSSOVER_RATE', 0.1)),
        metal_interswap_rate=float(os.environ.get('CATEMC_METAL_INTERSWAP_RATE', 0.8)),
        nonmetal_interswap_rate=float(os.environ.get('CATEMC_NONMETAL_INTERSWAP_RATE', 0.0)),
        intraswap_rate=float(os.environ.get('CATEMC_INTRASWAP_RATE', 0.1)),
        metal_element_pool=target_metal_species,
        nonmetal_element_pool=target_nonmetal_species,
        mutable_metal_indices=None,
        mutable_nonmetal_indices=None,
        raw_score_cutoff=-10000.0,
        temperature_key='temperature',
        replica_key='replica',
        acceptance_key='accepted',
        interswap_within_key=interswap_within_key,
        gpu_devices=gpu_devices,
        procs_per_device=workers_per_gpu,
        worker_pool=pool,
        traj_write_interval=traj_write_interval,
        scratch_dir=scratch_dir,
        scratch_sync_interval=scratch_sync_interval,
        verbose=verbose,
    )
    emc.run(starting_step=int(os.environ.get('CATEMC_STARTING_STEP', 1)),
            total_steps=int(os.environ.get('CATEMC_TOTAL_STEPS', 5000)))
finally:
    pool.shutdown()
