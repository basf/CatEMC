"""Benchmark: 24 workers, DB on /mnt/scratch vs CephFS.

Test 1: Both calculation_dir and db on CephFS (baseline)
Test 2: calculation_dir on CephFS, db_dir on /mnt/scratch (DB only on scratch)

This isolates the DB write overhead from traj write overhead.
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
CEPHFS_WORK = '/fs/cephfs/data/qm_inorganics/sandip/emc_db_bench_24w'
SCRATCH_DB = '/mnt/scratch/emc_db_24w'
NUM_STEPS = 20
NUM_WORKERS = 24
TWI = 1


def make_pool():
    fitness_kwargs = {
        'fmax_list': [0.10],
        'target_metal_species': ['Cu', 'Ru'],
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
    return WorkerPool(
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


def run_test(label, calc_dir, db_dir, pool):
    """Run EMC and return total elapsed time."""
    if os.path.exists(calc_dir):
        shutil.rmtree(calc_dir)
    os.makedirs(calc_dir, exist_ok=True)
    if db_dir and os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    input_pop = read(INPUT_POP, index=':')

    emc = EMC(f'bench_db_{label}',
              calculation_dir=calc_dir,
              input_pop=input_pop,
              fix_indices=None,
              trim_generation=100,
              pressure=0.,
              md_generation=10,
              exchange_generation=1,
              crossover_rate=0.,
              intraswap_rate=1.,
              metal_element_pool=['Cu', 'Ru'],
              nonmetal_element_pool=['O', 'X'],
              raw_score_cutoff=-10000.,
              temperature_key='temperature',
              replica_key='replica',
              acceptance_key='accepted',
              category_key='dopant_valence',
              gpu_devices=[0],
              procs_per_device=NUM_WORKERS,
              worker_pool=pool,
              traj_write_interval=TWI,
              db_dir=db_dir)

    t0 = time.time()
    emc.run(total_steps=NUM_STEPS)
    elapsed = time.time() - t0
    return elapsed


def main():
    print("=" * 60)
    print(f"DB-on-SCRATCH BENCHMARK — {NUM_WORKERS} workers, twi={TWI}")
    print(f"Steps: {NUM_STEPS}, 48 replicas, NequIP on H100")
    print("=" * 60)

    pool = make_pool()

    # Test 1: DB on CephFS (baseline)
    print(f"\n{'─' * 60}")
    print("Test 1: DB on CephFS (baseline)")
    print(f"{'─' * 60}")
    t_cephfs = run_test('cephfs', f'{CEPHFS_WORK}_cephfs', None, pool)
    print(f"  → DB on CephFS: {t_cephfs:.1f}s total, {t_cephfs/NUM_STEPS:.1f}s/gen")

    # Test 2: DB on /mnt/scratch
    print(f"\n{'─' * 60}")
    print("Test 2: DB on /mnt/scratch (local NVMe)")
    print(f"{'─' * 60}")
    t_scratch = run_test('scratch', f'{CEPHFS_WORK}_scratch', SCRATCH_DB, pool)
    print(f"  → DB on scratch: {t_scratch:.1f}s total, {t_scratch/NUM_STEPS:.1f}s/gen")

    # Copy DB back
    t_copy_start = time.time()
    dest_db = f'{CEPHFS_WORK}_scratch_db_copy'
    if os.path.exists(dest_db):
        shutil.rmtree(dest_db)
    shutil.copytree(SCRATCH_DB, dest_db)
    t_copy = time.time() - t_copy_start
    shutil.rmtree(SCRATCH_DB, ignore_errors=True)

    pool.shutdown()

    # Summary
    speedup = t_cephfs / t_scratch if t_scratch > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({NUM_WORKERS} workers, traj_write_interval={TWI})")
    print(f"{'=' * 60}")
    print(f"{'Config':<25} {'Total (s)':<12} {'Per gen (s)':<12} {'Speedup':<10}")
    print(f"{'─' * 59}")
    print(f"{'DB on CephFS':<25} {t_cephfs:<12.1f} {t_cephfs/NUM_STEPS:<12.1f} {'1.00x':<10}")
    print(f"{'DB on scratch':<25} {t_scratch:<12.1f} {t_scratch/NUM_STEPS:<12.1f} {f'{speedup:.2f}x':<10}")
    print(f"DB copy-back time: {t_copy:.1f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
