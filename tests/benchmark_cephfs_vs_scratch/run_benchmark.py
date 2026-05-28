"""CephFS vs Local Scratch I/O Benchmark for EMC Worker Scaling.

This benchmark measures per-generation wall time for the CatEMC evolutionary
Monte Carlo (EMC) simulation on CephFS vs local NVMe (/mnt/scratch).

It demonstrates that CephFS I/O (SQLite DB writes + traj appends) becomes the
dominant bottleneck as worker parallelism increases, since all workers complete
simultaneously and the main process must write 48 DB rows + 48 traj appends
in a burst.

Setup:
  - H100 GPU (Standard_NC40ads_H100_v5)
  - NequIP ML potential
  - 48 replicas (Cu0.1Ru0.9Ox bulk)
  - traj_write_interval=1 (write every generation to stress I/O)
  - md_generation=10 (MD re-relaxation every 10th gen)
  - 20 EMC generations per test

Usage:
  python run_benchmark.py <num_workers>

  Runs CephFS test then scratch test for the given worker count.
  Results are printed and appended to results.json.
"""
import sys
import os
import shutil
import time
import json

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
sys.path.insert(0, '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catemc.geemc import EMC
from catemc.worker_pool import WorkerPool
from ase.io import read
from ase import units


# ============ CONFIGURATION ============
INPUT_POP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj')
CEPHFS_BASE = '/fs/cephfs/data/qm_inorganics/sandip/emc_cephfs_bench'
SCRATCH_BASE = '/mnt/scratch/emc_scratch_bench'
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
NUM_STEPS = 20
TRAJ_WRITE_INTERVAL = 1
# =======================================


def run_emc_on_fs(work_dir, label, num_workers):
    """Run EMC in given directory, return (elapsed, per_gen_times)."""
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

    system_name = f'bench_{num_workers}w_{label}'
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
              traj_write_interval=TRAJ_WRITE_INTERVAL)

    t0 = time.time()
    emc.run(total_steps=NUM_STEPS)
    elapsed = time.time() - t0

    pool.shutdown()
    return elapsed


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_benchmark.py <num_workers>")
        print("  e.g.: python run_benchmark.py 6")
        sys.exit(1)

    num_workers = int(sys.argv[1])

    print("=" * 60)
    print(f"CephFS vs SCRATCH BENCHMARK — {num_workers} workers")
    print(f"Steps: {NUM_STEPS}, traj_write_interval: {TRAJ_WRITE_INTERVAL}")
    print(f"48 replicas, NequIP on H100")
    print("=" * 60)

    # --- CephFS ---
    print(f"\n{'─' * 60}")
    print("Test 1: CephFS")
    print(f"{'─' * 60}")
    cephfs_dir = f'{CEPHFS_BASE}_{num_workers}w'
    t_cephfs = run_emc_on_fs(cephfs_dir, 'cephfs', num_workers)
    print(f"  → CephFS: {t_cephfs:.1f}s total, {t_cephfs/NUM_STEPS:.1f}s/gen")

    # --- Scratch ---
    print(f"\n{'─' * 60}")
    print("Test 2: /mnt/scratch (local NVMe)")
    print(f"{'─' * 60}")
    scratch_dir = f'{SCRATCH_BASE}_{num_workers}w'
    t_scratch = run_emc_on_fs(scratch_dir, 'scratch', num_workers)
    print(f"  → scratch: {t_scratch:.1f}s total, {t_scratch/NUM_STEPS:.1f}s/gen")

    # Copy back
    dest = f'{CEPHFS_BASE}_{num_workers}w_scratch_results'
    if os.path.exists(dest):
        shutil.rmtree(dest)
    t_copy_start = time.time()
    shutil.copytree(scratch_dir, dest)
    t_copy = time.time() - t_copy_start
    shutil.rmtree(scratch_dir, ignore_errors=True)

    # --- Summary ---
    speedup = t_cephfs / t_scratch if t_scratch > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({num_workers} workers)")
    print(f"{'=' * 60}")
    print(f"{'Filesystem':<15} {'Total (s)':<12} {'Per gen (s)':<12} {'Speedup':<10}")
    print(f"{'─' * 49}")
    print(f"{'CephFS':<15} {t_cephfs:<12.1f} {t_cephfs/NUM_STEPS:<12.1f} {'1.00x':<10}")
    print(f"{'scratch':<15} {t_scratch:<12.1f} {t_scratch/NUM_STEPS:<12.1f} {f'{speedup:.2f}x':<10}")
    print(f"Copy-back time: {t_copy:.1f}s")
    print(f"{'=' * 60}")

    # Save to JSON
    result = {
        'num_workers': num_workers,
        'num_steps': NUM_STEPS,
        'traj_write_interval': TRAJ_WRITE_INTERVAL,
        'cephfs_total_s': round(t_cephfs, 1),
        'cephfs_per_gen_s': round(t_cephfs / NUM_STEPS, 1),
        'scratch_total_s': round(t_scratch, 1),
        'scratch_per_gen_s': round(t_scratch / NUM_STEPS, 1),
        'speedup': round(speedup, 2),
        'copy_back_s': round(t_copy, 1),
    }
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    # Update or append
    results = [r for r in results if r['num_workers'] != num_workers]
    results.append(result)
    results.sort(key=lambda r: r['num_workers'])
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()
