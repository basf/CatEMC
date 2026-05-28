"""Worker scaling sweep: measure time/gen for varying worker counts.

Runs 30 generations with 48 replicas, traj_write_interval=30 (write once at end).
Tests worker counts: 1, 6, 12, 18, 24, 30, 36, 42, 48

Output: JSON file with timing results for each worker count.
"""
import sys, os, shutil, time, json
import numpy as np
import random as pyrandom

CATEMC_DIR = '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC'
sys.path.insert(0, CATEMC_DIR)

BULK_DIR = '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk'
POP_FILE = os.path.join(BULK_DIR, 'Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj')
sys.path.insert(0, BULK_DIR)

from ase.io import read
from ase import units

N_GENS = 30
SEED = 42
TRAJ_WRITE_INTERVAL = 30  # write once at the end
WORKER_COUNTS = [1, 6, 12, 18, 24, 30, 36, 42, 48]

input_pop = read(POP_FILE, index=':')
print(f"Loaded {len(input_pop)} replicas, running {N_GENS} gens each")

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
}

target_metal_species = ['Cu', 'Ru', 'X']
target_nonmetal_species = ['O', 'X']


def run_sweep_point(num_workers, calc_dir):
    """Run EMC with given worker count and return timing info."""
    import re

    # Fresh seeds
    pyrandom.seed(SEED)
    np.random.seed(SEED)

    # Clean calc dir
    if os.path.exists(calc_dir):
        shutil.rmtree(calc_dir)
    os.makedirs(calc_dir)

    # Force fresh imports
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('catemc'):
            del sys.modules[mod_name]

    from catemc.geemc import EMC
    from catemc.worker_pool import WorkerPool

    gpu_devices = [0]

    t_pool_start = time.perf_counter()
    pool = WorkerPool(
        calculator_factory_module='calc_factory_nequip',
        calculator_factory_func='create_calculators',
        calculator_factory_kwargs={},
        fitness_func_module='catemc.calculation_templates',
        fitness_func_name='calculate_bulk_fitness_inprocess',
        fitness_kwargs=fitness_kwargs,
        num_workers=num_workers,
        gpu_devices=gpu_devices,
        startup_timeout=600.0,
    )
    pool_startup_time = time.perf_counter() - t_pool_start

    emc = EMC('sweep_test',
              calculation_dir=calc_dir,
              calculation_script=None,
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
              procs_per_device=num_workers,
              worker_pool=pool,
              traj_write_interval=TRAJ_WRITE_INTERVAL)

    # Capture stdout for timing extraction
    import io
    from contextlib import redirect_stdout
    captured = io.StringIO()

    t_run_start = time.perf_counter()
    with redirect_stdout(captured):
        emc.run(total_steps=N_GENS)
    total_run_time = time.perf_counter() - t_run_start

    pool.shutdown()
    output = captured.getvalue()

    # Parse per-generation times
    gen_times = []
    for line in output.split('\n'):
        if 'elapsed:' in line:
            m = re.search(r'elapsed: ([\d.]+)s', line)
            if m:
                gen_times.append(float(m.group(1)))

    # Skip gen 1 (elapsed=0) for avg
    actual_gen_times = [t for t in gen_times if t > 0.1]
    avg_time = sum(actual_gen_times) / len(actual_gen_times) if actual_gen_times else 0.0

    result = {
        'num_workers': num_workers,
        'pool_startup_time': round(pool_startup_time, 1),
        'total_run_time': round(total_run_time, 1),
        'avg_time_per_gen': round(avg_time, 2),
        'gen_times': [round(t, 2) for t in gen_times],
        'n_gens': N_GENS,
        'n_replicas': len(input_pop),
        'traj_write_interval': TRAJ_WRITE_INTERVAL,
    }
    return result


def main():
    print("=" * 60)
    print("WORKER SCALING SWEEP")
    print(f"Replicas: {len(input_pop)}, Gens: {N_GENS}, traj_interval: {TRAJ_WRITE_INTERVAL}")
    print(f"Worker counts: {WORKER_COUNTS}")
    print("=" * 60)
    sys.stdout.flush()

    base_dir = '/fs/cephfs/data/qm_inorganics/sandip/emc_sweep_workers'
    results = []

    for nw in WORKER_COUNTS:
        print(f"\n--- Running with {nw} workers ---")
        sys.stdout.flush()

        calc_dir = os.path.join(base_dir, f'workers_{nw}')
        result = run_sweep_point(nw, calc_dir)
        results.append(result)

        print(f"  Pool startup: {result['pool_startup_time']}s")
        print(f"  Total run:    {result['total_run_time']}s")
        print(f"  Avg/gen:      {result['avg_time_per_gen']}s")
        sys.stdout.flush()

    # Save results
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Workers':>8} {'Startup':>8} {'Run(s)':>8} {'Avg/gen':>8} {'Speedup':>8}")
    print("-" * 60)
    baseline = results[0]['avg_time_per_gen']
    for r in results:
        speedup = baseline / r['avg_time_per_gen'] if r['avg_time_per_gen'] > 0 else 0
        print(f"{r['num_workers']:>8} {r['pool_startup_time']:>7.1f}s {r['total_run_time']:>7.1f}s {r['avg_time_per_gen']:>7.2f}s {speedup:>7.1f}x")
    print("=" * 60)


if __name__ == '__main__':
    main()
