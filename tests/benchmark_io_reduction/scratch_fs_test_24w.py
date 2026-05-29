"""Filesystem benchmark: CephFS vs /mnt/scratch, 24 workers, traj_write_interval=1."""
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
NUM_WORKERS = 24
NUM_STEPS = 20
TRAJ_WRITE_INTERVAL = 1


def run_emc_on_fs(work_dir, label):
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

    system_name = f'fs_bench_24w_{label}'
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
    return elapsed


def main():
    print("=" * 60)
    print("FILESYSTEM BENCHMARK — CephFS vs /mnt/scratch (24 workers)")
    print(f"Workers: {NUM_WORKERS}, Steps: {NUM_STEPS}, traj_write_interval: {TRAJ_WRITE_INTERVAL}")
    print("=" * 60)

    print(f"\n{'─' * 60}")
    print("Test 1: CephFS")
    print(f"{'─' * 60}")
    cephfs_dir = '/fs/cephfs/data/qm_inorganics/sandip/emc_fs_bench_24w_cephfs'
    t_cephfs = run_emc_on_fs(cephfs_dir, 'cephfs')
    print(f"  → CephFS: {t_cephfs:.1f}s total, {t_cephfs/NUM_STEPS:.1f}s/gen")

    print(f"\n{'─' * 60}")
    print("Test 2: /mnt/scratch (local NVMe)")
    print(f"{'─' * 60}")
    scratch_dir = '/mnt/scratch/emc_fs_bench_24w_scratch'
    t_scratch = run_emc_on_fs(scratch_dir, 'scratch')
    print(f"  → scratch: {t_scratch:.1f}s total, {t_scratch/NUM_STEPS:.1f}s/gen")

    dest = '/fs/cephfs/data/qm_inorganics/sandip/emc_fs_bench_24w_scratch_results'
    if os.path.exists(dest):
        shutil.rmtree(dest)
    print(f"\nCopying scratch results back to CephFS...")
    t_copy_start = time.time()
    shutil.copytree(scratch_dir, dest)
    t_copy = time.time() - t_copy_start
    print(f"  → Copy back took {t_copy:.1f}s")
    shutil.rmtree(scratch_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print("SUMMARY (24 workers)")
    print(f"{'=' * 60}")
    print(f"{'Filesystem':<15} {'Total (s)':<12} {'Per gen (s)':<12} {'Speedup':<10}")
    print(f"{'─' * 49}")
    print(f"{'CephFS':<15} {t_cephfs:<12.1f} {t_cephfs/NUM_STEPS:<12.1f} {'1.00x':<10}")
    speedup = t_cephfs / t_scratch if t_scratch > 0 else 0
    print(f"{'scratch':<15} {t_scratch:<12.1f} {t_scratch/NUM_STEPS:<12.1f} {f'{speedup:.2f}x':<10}")
    print(f"\nCopy-back time: {t_copy:.1f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
