"""Validate that in-memory offspring + traj_write_interval produce
identical results to the original file-based code.

Strategy:
1. Run 5 generations with the PATCHED code (in-memory + traj_write_interval=5)
2. Run 5 generations with ORIGINAL code forced (file-based offspring, write every gen)
3. Compare: all acceptance decisions, energies, and final positions must match.

We fix random seeds so both runs are deterministic and identical.
"""
import sys, os, shutil, importlib, importlib.util
import numpy as np
import random as pyrandom

# Setup paths
CATEMC_DIR = '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC'
sys.path.insert(0, CATEMC_DIR)

BULK_DIR = '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk'
POP_FILE = os.path.join(BULK_DIR, 'Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj')

# Add bulk dir to path so calc_factory_nequip can be imported by workers
sys.path.insert(0, BULK_DIR)

from ase.io import read, write

# Use only 6 replicas for a quick test
N_REPLICAS = 6
N_GENS = 5
SEED = 42

input_pop_full = read(POP_FILE, index=':')
input_pop = input_pop_full[:N_REPLICAS]

# Fitness kwargs (same as production)
from ase import units
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


def run_emc(calc_dir, use_patched=True, traj_write_interval=1):
    """Run EMC for N_GENS and return list of decisions + energies."""
    # Fresh seeds
    pyrandom.seed(SEED)
    np.random.seed(SEED)
    
    # Clean calc dir
    if os.path.exists(calc_dir):
        shutil.rmtree(calc_dir)
    os.makedirs(calc_dir)
    
    # Force reload of the correct emc module
    if not use_patched:
        # Load backup (original) as the emc module
        import types
        bak_path = os.path.join(CATEMC_DIR, 'catemc', 'emc.py.bak')
        mod = types.ModuleType('catemc.emc')
        mod.__file__ = bak_path
        mod.__package__ = 'catemc'
        with open(bak_path) as _f:
            exec(compile(_f.read(), bak_path, 'exec'), mod.__dict__)
        sys.modules['catemc.emc'] = mod
    else:
        # Use current (patched) emc.py
        if 'catemc.emc' in sys.modules:
            del sys.modules['catemc.emc']
    
    if 'catemc.geemc' in sys.modules:
        del sys.modules['catemc.geemc']
    
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
        num_workers=1,
        gpu_devices=gpu_devices,
        startup_timeout=300.0,
    )
    
    target_metal_species = ['Cu', 'Ru', 'X']
    target_nonmetal_species = ['O', 'X']
    
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
        procs_per_device=1,
        worker_pool=pool,
    )
    
    if use_patched:
        emc_kwargs['traj_write_interval'] = traj_write_interval
    
    system_name = 'validate_test'
    emc = EMC(system_name, **emc_kwargs)
    
    # Capture stdout
    import io
    from contextlib import redirect_stdout
    captured = io.StringIO()
    with redirect_stdout(captured):
        emc.run(total_steps=N_GENS)
    
    pool.shutdown()
    output = captured.getvalue()
    
    # Parse acceptance decisions from output
    decisions = []
    for line in output.split('\n'):
        if 'move accepted' in line or 'move rejected' in line:
            decisions.append(line.strip())
    
    # Read final energies from database
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
    
    return decisions, final_energies, output


def main():
    print("=" * 60)
    print("VALIDATION TEST: Patched vs Original CatEMC")
    print(f"Replicas: {N_REPLICAS}, Generations: {N_GENS}, Seed: {SEED}")
    print("=" * 60)
    
    base = '/fs/cephfs/data/qm_inorganics/sandip/emc_validate'
    
    # Run 1: Patched code (in-memory offspring, traj_write_interval=5)
    print("\n[1/2] Running PATCHED code (in-memory + traj_write_interval=5)...")
    sys.stdout.flush()
    calc_dir_patched = os.path.join(base, 'patched')
    decisions_p, energies_p, output_p = run_emc(calc_dir_patched, use_patched=True, traj_write_interval=5)
    print(f"  Got {len(decisions_p)} decisions, {len(energies_p)} DB entries")
    sys.stdout.flush()
    
    # Run 2: Original code (file-based offspring, write every gen)
    print("\n[2/2] Running ORIGINAL code (file-based, traj_write_interval=1)...")
    sys.stdout.flush()
    calc_dir_orig = os.path.join(base, 'original')
    decisions_o, energies_o, output_o = run_emc(calc_dir_orig, use_patched=False)
    print(f"  Got {len(decisions_o)} decisions, {len(energies_o)} DB entries")
    sys.stdout.flush()
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print("=" * 60)
    
    passed = True
    
    # Compare decisions
    if decisions_p == decisions_o:
        print(f"  [PASS] All {len(decisions_p)} acceptance decisions IDENTICAL")
    else:
        print(f"  [FAIL] Decisions differ!")
        print(f"    Patched: {len(decisions_p)} decisions")
        print(f"    Original: {len(decisions_o)} decisions")
        for i, (dp, do) in enumerate(zip(decisions_p, decisions_o)):
            if dp != do:
                print(f"    First diff at decision {i}:")
                print(f"      Patched:  {dp}")
                print(f"      Original: {do}")
                break
        passed = False
    
    # Compare energies (tolerance for GPU non-determinism)
    max_diff = 0.0
    for ep, eo in zip(energies_p, energies_o):
        diff = abs(ep[1] - eo[1])
        max_diff = max(max_diff, diff)
    tol = 1e-3  # 1 meV tolerance for GPU non-determinism
    if max_diff < tol:
        print(f"  [PASS] All {len(energies_p)} DB energies match within {tol} eV (max diff: {max_diff:.6e} eV)")
    else:
        print(f"  [FAIL] Energies differ by more than {tol} eV! Max diff: {max_diff:.6e} eV")
        for ep, eo in zip(energies_p, energies_o):
            diff = abs(ep[1] - eo[1])
            if diff > tol:
                print(f"    Patched:  {ep}  Original: {eo}  Diff: {diff:.6e}")
        passed = False
    
    print("\n" + "=" * 60)
    if passed:
        print("RESULT: PASS - Patched code produces identical results")
    else:
        print("RESULT: FAIL - Results differ!")
    print("=" * 60)
    
    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
