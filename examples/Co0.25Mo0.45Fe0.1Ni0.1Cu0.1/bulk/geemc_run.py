from catemc.geemc import GEEMC
from ase.io import read, write
import numpy as np
import sys
import os
import argparse


parser = argparse.ArgumentParser(description='Run GE-EMC bulk simulation')
parser.add_argument('gpu_devices', type=str, help='Comma-separated GPU device IDs')
parser.add_argument('ppg', type=int, help='Processes per GPU (subprocess mode) or workers (worker_pool mode)')
parser.add_argument('--mode', choices=['subprocess', 'worker_pool'], default='subprocess',
                    help='Execution mode: subprocess (default) or worker_pool (persistent workers, faster)')
parser.add_argument('--calc-factory', type=str, default=None,
                    help='Calculator factory module (required for worker_pool mode)')
args = parser.parse_args()

gpu_devices = [int(di) for di in args.gpu_devices.split(',')]
ppg = args.ppg
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])

comp = 'Co0.25Mo0.45Fe0.1Ni0.1Cu0.1'
structure = 'bulk'
system_name = f'{comp}_{structure}'
dir_name = f'/mnt/nfs/emc_tutorial/{comp}/{structure}'
input_pop_fname = f'{dir_name}/{system_name}_input_pop.traj'
target_metal_species = ['Co','Mo','Fe','Ni','Cu']
input_pop = read(input_pop_fname, index=':')

# --- Worker pool setup (optional, activated with --mode worker_pool) ---
worker_pool = None
if args.mode == 'worker_pool':
    from catemc.worker_pool import WorkerPool
    from ase import units

    calc_factory = args.calc_factory or 'calc_factory'  # default module name

    fitness_kwargs = {
        'fmax_list': [0.10],
        'target_metal_species': target_metal_species,
        'md_timestep': 1 * units.fs,
        'md_taut': 0.5 * 1000 * units.fs,
        'md_steps': 200,
        'md_write_interval': 10,
        'max_relax_steps': 100,
        'neighbor_cutoff': 3.5,
        'raw_score_cutoff': -10000.,
        'temperature_key': 'temperature',
        'replica_key': 'replica',
    }

    worker_pool = WorkerPool(
        calculator_factory_module=calc_factory,
        calculator_factory_func='create_calculators',
        fitness_func_module='catemc.calculation_templates',
        fitness_func_name='calculate_bulk_fitness_inprocess',
        fitness_kwargs=fitness_kwargs,
        num_workers=len(input_pop),
        gpu_devices=gpu_devices,
    )
    print(f"Worker pool mode: {len(input_pop)} persistent workers on GPUs {gpu_devices}")

# --- EMC setup ---
emc = GEEMC(system_name,
            calculation_dir=dir_name,
            calculation_script='calculate_fitness.py',
            input_pop=input_pop,
            fix_indices=None,
            trim_generation=100,
            pressure=None,
            md_generation=10,
            exchange_generation=1,
            bias_starting_step=300,
            crossover_rate=0.1,
            metal_interswap_rate=0.8,
            intraswap_rate=0.1,
            metal_element_pool=target_metal_species,
            raw_score_cutoff=-10000.,
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted',
            gpu_devices=gpu_devices,
            procs_per_device=ppg,
            worker_pool=worker_pool)
emc.run(starting_step=1, total_steps=5000)

if worker_pool is not None:
    worker_pool.shutdown()
