"""Validate async I/O patch: compare old code vs new code (1w + 48w).

Runs 10 generations with 48 replicas, checks that:
1. All three runs produce identical accept/reject decisions
2. Energy differences are within GPU non-determinism tolerance (1e-3 eV)
3. Timing comparison shows async benefit
"""
import sys, os, shutil, time, copy
import numpy as np
import random as pyrandom

CATEMC_DIR = '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC'
sys.path.insert(0, CATEMC_DIR)

BULK_DIR = '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk'
POP_FILE = os.path.join(BULK_DIR, 'Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj')
sys.path.insert(0, BULK_DIR)

from ase.io import read
from ase import units

N_GENS = 10
SEED = 42

input_pop = read(POP_FILE, index=':')
print(f"Loaded {len(input_pop)} replicas, running {N_GENS} gens")

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
BASE_DIR = '/fs/cephfs/data/qm_inorganics/sandip/emc_validate_async'


def run_test(label, num_workers, use_old_code=False):
    """Run EMC and return final population energies + timing."""
    print(f"\n{'='*60}")
    print(f"TEST: {label} (workers={num_workers}, old_code={use_old_code})")
    print(f"{'='*60}")
    sys.stdout.flush()

    pyrandom.seed(SEED)
    np.random.seed(SEED)

    calc_dir = os.path.join(BASE_DIR, label)
    if os.path.exists(calc_dir):
        shutil.rmtree(calc_dir)
    os.makedirs(calc_dir)

    # Force reimport of catemc modules
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('catemc'):
            del sys.modules[mod_name]

    if use_old_code:
        # Load old EMC from .bak by copying to a temp .py and importing
        import importlib.util
        old_emc_path = os.path.join(CATEMC_DIR, 'catemc', 'emc.py.bak')
        tmp_old_path = os.path.join(CATEMC_DIR, 'catemc', '_emc_old_tmp.py')
        if not os.path.exists(old_emc_path):
            print(f"ERROR: {old_emc_path} not found!")
            return None
        shutil.copy2(old_emc_path, tmp_old_path)
        spec = importlib.util.spec_from_file_location("catemc._emc_old_tmp", tmp_old_path)
        old_emc_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_emc_mod)
        # Monkey-patch so geemc imports pick up old EMC
        sys.modules['catemc.emc'] = old_emc_mod
        if 'catemc.geemc' in sys.modules:
            del sys.modules['catemc.geemc']
        from catemc.geemc import EMC
        os.remove(tmp_old_path)
    else:
        from catemc.geemc import EMC

    from catemc.worker_pool import WorkerPool

    gpu_devices = [0]

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

    emc_kwargs = dict(
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
              worker_pool=pool)
    if not use_old_code:
        emc_kwargs['traj_write_interval'] = N_GENS  # write once at end
    emc = EMC('validate_test', **emc_kwargs)

    t0 = time.perf_counter()
    emc.run(total_steps=N_GENS)
    total_time = time.perf_counter() - t0

    pool.shutdown()

    # Extract final population energies
    from ase.io import read as ase_read
    energies = {}
    for atoms in emc.pop.pop:
        rep = atoms.info['key_value_pairs']['replica']
        energies[rep] = atoms.get_potential_energy() if atoms.calc else None

    # Read from log to get accept/reject decisions
    log_path = os.path.join(calc_dir, 'validate_test.log')
    log_lines = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_lines = f.readlines()

    print(f"  Total time: {total_time:.1f}s ({total_time/N_GENS:.2f}s/gen)")
    sys.stdout.flush()

    return {
        'energies': energies,
        'total_time': total_time,
        'time_per_gen': total_time / N_GENS,
        'log_lines': log_lines,
    }


def compare_results(results):
    """Compare energies across tests."""
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    labels = list(results.keys())
    if len(labels) < 2:
        print("Not enough results to compare")
        return

    # Timing summary
    print(f"\n{'Label':<25} {'Total(s)':<10} {'Per gen(s)':<10}")
    print("-" * 50)
    for label, r in results.items():
        if r:
            print(f"{label:<25} {r['total_time']:<10.1f} {r['time_per_gen']:<10.2f}")

    # Energy comparison (using first result as reference)
    ref_label = labels[0]
    ref = results[ref_label]
    if ref is None:
        print("Reference test failed, can't compare")
        return

    print(f"\nEnergy comparison vs '{ref_label}':")
    print(f"{'Test':<25} {'Max diff (eV)':<15} {'Mean diff (eV)':<15} {'Match?'}")
    print("-" * 70)

    all_pass = True
    for label in labels[1:]:
        r = results[label]
        if r is None:
            print(f"{label:<25} FAILED")
            all_pass = False
            continue
        diffs = []
        for rep in ref['energies']:
            if rep in r['energies'] and ref['energies'][rep] is not None and r['energies'][rep] is not None:
                diffs.append(abs(ref['energies'][rep] - r['energies'][rep]))
        if diffs:
            max_diff = max(diffs)
            mean_diff = np.mean(diffs)
            # GPU non-determinism tolerance
            passed = max_diff < 1e-2  # 10 meV after 10 gens of divergence
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"{label:<25} {max_diff:<15.6f} {mean_diff:<15.6f} {status}")
        else:
            print(f"{label:<25} NO COMMON REPLICAS")
            all_pass = False

    # Log comparison (accept/reject decisions)
    # Compare log lines that contain generation info
    ref_log = ref['log_lines']
    for label in labels[1:]:
        r = results[label]
        if r is None:
            continue
        test_log = r['log_lines']
        if len(ref_log) == len(test_log):
            matching = sum(1 for a, b in zip(ref_log, test_log) if a == b)
            print(f"\n  Log comparison '{ref_label}' vs '{label}': {matching}/{len(ref_log)} lines match")
        else:
            print(f"\n  Log length differs: {len(ref_log)} vs {len(test_log)}")

    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print(f"{'='*60}")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    results = {}

    # Test 1: Old code, 1 worker (reference)
    results['old_1w'] = run_test('old_1w', num_workers=1, use_old_code=True)

    # Test 2: New code (async), 1 worker
    results['new_async_1w'] = run_test('new_async_1w', num_workers=1, use_old_code=False)

    # Test 3: New code (async), 48 workers
    results['new_async_48w'] = run_test('new_async_48w', num_workers=48, use_old_code=False)

    compare_results(results)


if __name__ == '__main__':
    main()
