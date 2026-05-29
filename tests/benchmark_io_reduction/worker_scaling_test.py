"""Worker scaling test: measure time per EMC generation for different worker counts.

Runs 5 EMC generations for each worker count (1, 2, 4, 6) on a single H100 GPU,
reports wall time per generation to find the optimal parallelism level.
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


WORK_BASE = '/fs/cephfs/data/qm_inorganics/sandip/emc_worker_scaling'
INPUT_POP = '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk/Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj'
WORKER_COUNTS = [1, 2, 4, 6]
NUM_STEPS = 5  # EMC generations per test


def run_test(num_workers):
    """Run EMC for NUM_STEPS generations with given worker count, return time per gen."""
    work_dir = os.path.join(WORK_BASE, f'workers_{num_workers}')
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
        num_workers=num_workers,
        gpu_devices=[0],
        startup_timeout=600.0,
    )

    system_name = f'worker_scaling_{num_workers}w'
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
              procs_per_device=num_workers,
              worker_pool=pool,
              traj_write_interval=100)  # disable traj writes during benchmark

    t0 = time.time()
    emc.run(total_steps=NUM_STEPS)
    elapsed = time.time() - t0

    pool.shutdown()
    time_per_gen = elapsed / NUM_STEPS
    return elapsed, time_per_gen


def main():
    print("=" * 60)
    print("WORKER SCALING TEST — H100 NequIP")
    print(f"Steps per test: {NUM_STEPS}, Population: 48 replicas")
    print("=" * 60)

    results = {}
    for nw in WORKER_COUNTS:
        print(f"\n{'─' * 60}")
        print(f"Testing {nw} worker(s)...")
        print(f"{'─' * 60}")
        elapsed, tpg = run_test(nw)
        results[nw] = {'elapsed': elapsed, 'time_per_gen': tpg}
        print(f"  → {nw} workers: {elapsed:.1f}s total, {tpg:.1f}s/gen")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Workers':<10} {'Total (s)':<12} {'Per gen (s)':<12} {'Speedup':<10}")
    print(f"{'─' * 44}")
    baseline = results[1]['time_per_gen']
    for nw in WORKER_COUNTS:
        r = results[nw]
        speedup = baseline / r['time_per_gen']
        print(f"{nw:<10} {r['elapsed']:<12.1f} {r['time_per_gen']:<12.1f} {speedup:<10.2f}x")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
