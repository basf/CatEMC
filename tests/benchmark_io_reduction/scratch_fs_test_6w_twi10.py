"""Filesystem benchmark: CephFS with 6 workers, traj_write_interval=10.

Runs 20 EMC generations to compare against traj_write_interval=1 results.
"""
import sys
import os
import shutil
import time

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
sys.path.insert(0, '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC')
sys.path.insert(0, '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk')

from catemc.geemc import EMC
from catemc.worker_pool import WorkerPool
from ase.io import read
from ase import units


INPUT_POP = '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk/Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj'
NUM_WORKERS = 6
NUM_STEPS = 20
TRAJ_WRITE_INTERVAL = 10


def main():
    work_dir = '/fs/cephfs/data/qm_inorganics/sandip/emc_fs_bench_6w_twi10'
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    input_pop = read(INPUT_POP, index=':')
    dopant = 'Cu'
    target_metal_species = [dopant, 'Ru']
    target_nonmetal_species = ['O', 'X']

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
        num_workers=NUM_WORKERS,
        gpu_devices=[0],
        startup_timeout=600.0,
    )

    print("=" * 60)
    print("CephFS BENCHMARK — 6 workers, traj_write_interval=10")
    print(f"Workers: {NUM_WORKERS}, Steps: {NUM_STEPS}, traj_write_interval: {TRAJ_WRITE_INTERVAL}")
    print("=" * 60)

    system_name = 'fs_bench_6w_twi10'
    emc = EMC(system_name,
              calculation_dir=work_dir,
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
              gpu_devices=[0],
              procs_per_device=NUM_WORKERS,
              worker_pool=pool,
              traj_write_interval=TRAJ_WRITE_INTERVAL)

    t0 = time.time()
    emc.run(total_steps=NUM_STEPS)
    elapsed = time.time() - t0

    pool.shutdown()

    print(f"\n{'=' * 60}")
    print(f"RESULT: CephFS, 6 workers, traj_write_interval={TRAJ_WRITE_INTERVAL}")
    print(f"Total: {elapsed:.1f}s, Per gen: {elapsed/NUM_STEPS:.1f}s/gen")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
