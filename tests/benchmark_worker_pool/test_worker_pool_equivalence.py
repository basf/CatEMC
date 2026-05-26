#!/usr/bin/env python
"""Test: verify that the persistent worker pool produces the same results as subprocess mode.

This test relaxes a set of structures using both the worker pool and subprocess
modes, then compares the per-replica energies. The expected difference is < 0.02 eV,
which arises from non-deterministic CUDA execution (not a code bug).

Requirements:
    - GPU with CUDA support
    - NequIP compiled model (set COMPILED_MODEL below or via env var)
    - catemc installed or on PYTHONPATH
    - Initial population traj file (set POPULATION_FILE below or via env var)

Usage:
    export PYTHONPATH=/path/to/CatEMC:$PYTHONPATH
    export COMPILED_MODEL=/path/to/model.nequip.pt2
    export POPULATION_FILE=/path/to/input_pop.traj
    python test_worker_pool_equivalence.py
"""

import sys
import os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

import numpy as np
import time
import tempfile
from ase import units
from ase.io import read, write

# ============ Configuration (override via environment variables) ============
COMPILED_MODEL = os.environ.get(
    "COMPILED_MODEL",
    "/mnt/nfs/RuMOx_allegro/NequIP-OAM-S-stage6-oracle.nequip.pt2"
)
POPULATION_FILE = os.environ.get(
    "POPULATION_FILE",
    "/mnt/nfs/RuMOx_catemc/Cu0.1Ru0.9Ox_pet_productionrun/bulk/Cu0.1Ru0.9Ox_pet_productionrun_bulk_input_pop.traj"
)
CALC_SCRIPT_DIR = os.environ.get(
    "CALC_SCRIPT_DIR",
    "/mnt/nfs/RuMOx_catemc/Cu0.1Ru0.9Ox_pet_productionrun/bulk"
)
TARGET_METAL_SPECIES = ['Cu', 'Ru']
FMAX = 0.10
MAX_RELAX_STEPS = 100
NUM_REPLICAS = 5  # Keep small for fast testing
GPU_DEVICES = [0]
# ============================================================================


def load_population(n_replicas=NUM_REPLICAS):
    """Load test population from traj file."""
    pop = read(POPULATION_FILE, index=f':{n_replicas}')
    if not isinstance(pop, list):
        pop = [pop]
    return pop


def run_worker_pool(population):
    """Evaluate population using the persistent worker pool."""
    from catemc.worker_pool import WorkerPool

    # Write calculator factory to a temp file
    factory_path = os.path.join(tempfile.gettempdir(), '_test_calc_factory.py')
    with open(factory_path, 'w') as f:
        f.write(f"""
import sys, os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
def create_calculators():
    from nequip.integrations.ase import NequIPCalculator
    calc = NequIPCalculator.from_compiled_model("{COMPILED_MODEL}", device="cuda")
    return [calc]
""")

    if tempfile.gettempdir() not in sys.path:
        sys.path.insert(0, tempfile.gettempdir())

    fitness_kwargs = {
        'fmax_list': [FMAX],
        'target_metal_species': TARGET_METAL_SPECIES,
        'md_timestep': 1 * units.fs,
        'md_taut': 0.5 * 1000 * units.fs,
        'md_steps': 200,
        'md_write_interval': 10,
        'max_relax_steps': MAX_RELAX_STEPS,
        'neighbor_cutoff': 3.5,
        'raw_score_cutoff': -10000.,
        'temperature_key': 'temperature',
        'replica_key': 'replica',
    }

    pool = WorkerPool(
        calculator_factory_module='_test_calc_factory',
        calculator_factory_func='create_calculators',
        calculator_factory_kwargs={},
        fitness_func_module='catemc.calculation_templates',
        fitness_func_name='calculate_bulk_fitness_inprocess',
        fitness_kwargs=fitness_kwargs,
        num_workers=len(population),
        gpu_devices=GPU_DEVICES,
        startup_timeout=120.0,
    )

    t0 = time.time()
    results = pool.evaluate(population, run_md=False, fix_indices=None)
    elapsed = time.time() - t0
    pool.shutdown()

    return results, elapsed


def run_subprocess(population):
    """Evaluate population using the original subprocess mode."""
    from catemc.emc import EMC

    # Create tmp directory
    tmp_name = 'test_equivalence_subprocess'
    os.makedirs(f'{CALC_SCRIPT_DIR}/tmp_{tmp_name}', exist_ok=True)

    emc = EMC(
        name=tmp_name,
        calculation_dir=CALC_SCRIPT_DIR,
        calculation_script='calculate_fitness.py',
        input_pop=population,
        fix_indices=None,
        gpu_devices=GPU_DEVICES,
        procs_per_device=len(population),
    )

    t0 = time.time()
    results = emc.get_relaxed_candidates(population, run_md=False)
    elapsed = time.time() - t0

    return results, elapsed


def main():
    print("=" * 70)
    print("TEST: Worker Pool vs Subprocess Equivalence")
    print("=" * 70)
    print(f"Model: {COMPILED_MODEL}")
    print(f"Population: {POPULATION_FILE}")
    print(f"Replicas: {NUM_REPLICAS}")
    print()

    # Load population
    population = load_population()
    print(f"Loaded {len(population)} structures ({len(population[0])} atoms each)")

    # Run worker pool
    print("\n--- Worker Pool Mode ---")
    results_wp, time_wp = run_worker_pool(population)
    wp_energies = [r.info['key_value_pairs']['potential_energy'] for r in results_wp]
    wp_scores = [r.info['key_value_pairs']['raw_score'] for r in results_wp]
    print(f"  Time: {time_wp:.2f}s")
    print(f"  Energies: {[f'{e:.4f}' for e in wp_energies]}")

    # Run subprocess
    print("\n--- Subprocess Mode ---")
    results_sub, time_sub = run_subprocess(population)
    sub_energies = [r.info['key_value_pairs']['potential_energy'] for r in results_sub]
    sub_scores = [r.info['key_value_pairs']['raw_score'] for r in results_sub]
    print(f"  Time: {time_sub:.2f}s")
    print(f"  Energies: {[f'{e:.4f}' for e in sub_energies]}")

    # Compare
    print("\n--- Comparison ---")
    e_diff = np.array(wp_energies) - np.array(sub_energies)
    s_diff = np.array(wp_scores) - np.array(sub_scores)
    max_e_diff = np.max(np.abs(e_diff))
    mean_e_diff = np.mean(np.abs(e_diff))
    max_s_diff = np.max(np.abs(s_diff))

    print(f"  Energy diffs (eV): max={max_e_diff:.6f}, mean={mean_e_diff:.6f}")
    print(f"  Score diffs:       max={max_s_diff:.6f}")
    print(f"  Per-replica:       {[f'{d:.6f}' for d in e_diff]}")
    print(f"  Speedup:           {time_sub/time_wp:.1f}x")

    # Verdict
    print("\n--- Verdict ---")
    TOLERANCE = 0.05  # eV - generous to account for GPU non-determinism
    if max_e_diff < TOLERANCE:
        print(f"  PASS: max energy difference ({max_e_diff:.4f} eV) < tolerance ({TOLERANCE} eV)")
        print(f"  Note: ~0.01 eV differences are expected from non-deterministic CUDA atomics")
        return 0
    else:
        print(f"  FAIL: max energy difference ({max_e_diff:.4f} eV) >= tolerance ({TOLERANCE} eV)")
        print(f"  This suggests a bug in the worker pool implementation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
