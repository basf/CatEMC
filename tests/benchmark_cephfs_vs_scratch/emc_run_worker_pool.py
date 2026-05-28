"""EMC run script with persistent worker pool mode.

Usage:
    # Worker pool mode (3-22× faster, model loaded once):
    python emc_run_worker_pool.py 0 20

    # Args: gpu_devices, procs_per_gpu, [traj_write_interval]
    # traj_write_interval: write emc_*.traj every N gens (default 10)
    # But uses persistent workers instead of subprocess spawning.
"""
import sys
import os

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

from catemc.geemc import EMC
from catemc.worker_pool import WorkerPool
from ase.io import read
from ase import units


def main():
    gpu_devices = [int(di) for di in sys.argv[1].split(",")]
    ppg = int(sys.argv[2])

    traj_write_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])

    dopant = 'Cu'
    comp = f'{dopant}0.1Ru0.9Ox_nequip_productionrun'
    structure = 'bulk'
    system_name = f'{comp}_{structure}'
    dir_name = f'/fs/cephfs/data/qm_inorganics/tiago/emc_oer/{comp}/{structure}'
    input_pop_fname = f'{dir_name}/{system_name}_input_pop.traj'
    target_metal_species = [dopant, 'Ru']
    target_nonmetal_species = ['O', 'X']
    input_pop = read(input_pop_fname, index=':')

    # --- Worker pool setup ---
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

    pool = WorkerPool(
        calculator_factory_module='calc_factory_nequip',
        calculator_factory_func='create_calculators',
        calculator_factory_kwargs={},
        fitness_func_module='catemc.calculation_templates',
        fitness_func_name='calculate_bulk_fitness_inprocess',
        fitness_kwargs=fitness_kwargs,
        num_workers=ppg,
        gpu_devices=gpu_devices,
        startup_timeout=600.0,
    )
    print(f"Worker pool ready: {len(input_pop)} persistent workers on GPUs {gpu_devices}")

    # --- EMC setup (same as emc_run.py but with worker_pool) ---
    emc = EMC(system_name,
              calculation_dir=dir_name,
              #calculation_script='calculate_fitness.py',
              input_pop=input_pop,
              fix_indices=None,
              trim_generation=100,
              pressure=0.,
              md_generation=10,
              exchange_generation=1,
              crossover_rate=0.,
              intraswap_rate=1.,
              metal_element_pool=target_metal_species,
              nonmetal_element_pool=target_nonmetal_species,
              raw_score_cutoff=-10000.,
              temperature_key='temperature',
              replica_key='replica',
              acceptance_key='accepted',
              category_key='dopant_valence',
              gpu_devices=gpu_devices,
              procs_per_device=ppg,
              worker_pool=pool,
              traj_write_interval=traj_write_interval)

    emc.run(total_steps=5000)
    pool.shutdown()


if __name__ == '__main__':
    main()
