"""3-way validation + timing test with 48 replicas.

Test 1: OLD code (file-based offspring, traj every gen), 1 worker
Test 2: NEW code (in-memory offspring, traj every 5 gens), 1 worker
Test 3: NEW code (in-memory offspring, traj every 5 gens), 48 workers

All use same seed=42, same 48 replicas, 5 generations.
Compare acceptance decisions + energies (within 1e-3 eV tolerance).
Report timing per generation (excluding calc startup).
"""
import sys, os, shutil, importlib, importlib.util, types, time
import numpy as np
import random as pyrandom

CATEMC_DIR = '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC'
sys.path.insert(0, CATEMC_DIR)

BULK_DIR = '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk'
POP_FILE = os.path.join(BULK_DIR, 'Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj')
sys.path.insert(0, BULK_DIR)

from ase.io import read, write
from ase import units

N_GENS = 5
SEED = 42

input_pop = read(POP_FILE, index=':')
print(f"Loaded {len(input_pop)} replicas")

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


def run_emc(calc_dir, num_workers, use_patched=True, traj_write_interval=1, label=""):
    """Run EMC and return (decisions, energies, gen_times)."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  workers={num_workers}, traj_write_interval={traj_write_interval}, patched={use_patched}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Fresh seeds
    pyrandom.seed(SEED)
    np.random.seed(SEED)

    # Clean calc dir
    if os.path.exists(calc_dir):
        shutil.rmtree(calc_dir)
    os.makedirs(calc_dir)

    # Force reload of the correct emc module
    if not use_patched:
        bak_path = os.path.join(CATEMC_DIR, 'catemc', 'emc.py.bak')
        mod = types.ModuleType('catemc.emc')
        mod.__file__ = bak_path
        mod.__package__ = 'catemc'
        # Need to set up the module's globals properly
        import catemc
        mod.__path__ = []
        with open(bak_path) as _f:
            exec(compile(_f.read(), bak_path, 'exec'), mod.__dict__)
        sys.modules['catemc.emc'] = mod
    else:
        if 'catemc.emc' in sys.modules:
            del sys.modules['catemc.emc']

    if 'catemc.geemc' in sys.modules:
        del sys.modules['catemc.geemc']

    from catemc.geemc import EMC
    from catemc.worker_pool import WorkerPool

    gpu_devices = [0]
    
    t_start_pool = time.perf_counter()
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
    pool_time = time.perf_counter() - t_start_pool
    print(f"  Pool startup: {pool_time:.1f}s")
    sys.stdout.flush()

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
        worker_pool=pool,
    )

    if use_patched:
        emc_kwargs['traj_write_interval'] = traj_write_interval

    system_name = 'validate_test'
    emc = EMC(system_name, **emc_kwargs)

    # Capture stdout to parse decisions
    import io
    from contextlib import redirect_stdout
    captured = io.StringIO()
    
    t_run_start = time.perf_counter()
    with redirect_stdout(captured):
        emc.run(total_steps=N_GENS)
    total_run_time = time.perf_counter() - t_run_start

    pool.shutdown()
    output = captured.getvalue()

    # Parse generation times from output
    gen_times = []
    for line in output.split('\n'):
        if 'elapsed:' in line:
            import re
            m = re.search(r'elapsed: ([\d.]+)s', line)
            if m:
                gen_times.append(float(m.group(1)))

    # Parse decisions
    decisions = []
    for line in output.split('\n'):
        if 'move accepted' in line or 'move rejected' in line:
            decisions.append(line.strip())

    # Read energies from DB
    from ase.db import connect
    db = connect(os.path.join(calc_dir, f'{system_name}.db'))
    final_energies = []
    for row in db.select('relaxed=1'):
        final_energies.append((
            row.key_value_pairs['replica'],
            round(row.key_value_pairs.get('raw_score', 0.0), 10),
            row.key_value_pairs.get('emc_step', None)
        ))
    final_energies.sort()

    # Print timing summary
    print(f"  Total run time (excl pool startup): {total_run_time:.1f}s")
    if gen_times:
        # Skip gen 1 (elapsed=0), report gen 2+
        actual_gen_times = [t for t in gen_times if t > 0.1]
        if actual_gen_times:
            avg = sum(actual_gen_times) / len(actual_gen_times)
            print(f"  Avg time/gen: {avg:.1f}s  (gens 2-{N_GENS})")
            print(f"  Per-gen times: {[f'{t:.1f}s' for t in gen_times]}")
    print(f"  Decisions: {len(decisions)}, DB entries: {len(final_energies)}")
    sys.stdout.flush()

    return decisions, final_energies, gen_times, total_run_time


def compare(label_a, res_a, label_b, res_b):
    """Compare two runs."""
    dec_a, eng_a, _, _ = res_a
    dec_b, eng_b, _, _ = res_b
    
    print(f"\n  Comparing {label_a} vs {label_b}:")
    
    # Decisions
    if dec_a == dec_b:
        print(f"    [PASS] All {len(dec_a)} decisions IDENTICAL")
    else:
        n_diff = sum(1 for a, b in zip(dec_a, dec_b) if a != b)
        print(f"    [FAIL] {n_diff}/{len(dec_a)} decisions differ")
        return False

    # Energies
    max_diff = 0.0
    for ea, eb in zip(eng_a, eng_b):
        diff = abs(ea[1] - eb[1])
        max_diff = max(max_diff, diff)
    tol = 1e-3
    if max_diff < tol:
        print(f"    [PASS] Energies match within {tol} eV (max diff: {max_diff:.6e} eV)")
    else:
        print(f"    [FAIL] Max energy diff: {max_diff:.6e} eV > {tol} eV")
        return False
    
    return True


def main():
    print("=" * 60)
    print("3-WAY VALIDATION + TIMING: 48 replicas, 5 generations")
    print("=" * 60)

    base = '/fs/cephfs/data/qm_inorganics/sandip/emc_validate_48'

    # Test 1: OLD code, 1 worker
    res1 = run_emc(
        os.path.join(base, 'test1_old_1w'),
        num_workers=1, use_patched=False, traj_write_interval=1,
        label="TEST 1: OLD code, 1 worker, traj every gen"
    )

    # Test 2: NEW code, 1 worker, traj every 5 gens
    res2 = run_emc(
        os.path.join(base, 'test2_new_1w'),
        num_workers=1, use_patched=True, traj_write_interval=5,
        label="TEST 2: NEW code, 1 worker, traj every 5 gens"
    )

    # Test 3: NEW code, 48 workers, traj every 5 gens
    res3 = run_emc(
        os.path.join(base, 'test3_new_48w'),
        num_workers=48, use_patched=True, traj_write_interval=5,
        label="TEST 3: NEW code, 48 workers, traj every 5 gens"
    )

    # Comparisons
    print("\n" + "=" * 60)
    print("COMPARISONS:")
    print("=" * 60)
    
    ok1 = compare("Test1 (old)", res1, "Test2 (new 1w)", res2)
    ok2 = compare("Test1 (old)", res1, "Test3 (new 48w)", res3)

    # Timing summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY (excluding pool startup):")
    print("=" * 60)
    print(f"  Test 1 (OLD, 1 worker):       {res1[3]:.1f}s total  ({res1[3]/N_GENS:.1f}s/gen)")
    print(f"  Test 2 (NEW, 1 worker):       {res2[3]:.1f}s total  ({res2[3]/N_GENS:.1f}s/gen)")
    print(f"  Test 3 (NEW, 48 workers):     {res3[3]:.1f}s total  ({res3[3]/N_GENS:.1f}s/gen)")
    print(f"\n  Speedup (old vs new-1w):  {res1[3]/res2[3]:.1f}x")
    print(f"  Speedup (old vs new-48w): {res1[3]/res3[3]:.1f}x")

    print("\n" + "=" * 60)
    if ok1 and ok2:
        print("RESULT: ALL PASS")
    else:
        print("RESULT: SOME TESTS FAILED")
    print("=" * 60)
    
    if not (ok1 and ok2):
        sys.exit(1)


if __name__ == '__main__':
    main()
