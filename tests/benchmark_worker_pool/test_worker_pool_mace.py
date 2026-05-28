#!/usr/bin/env python
"""Test: worker pool equivalence & speedup with MACE model.

Shows the benefit of persistent workers by running multiple evaluation rounds.
Worker pool loads the model ONCE; old approach spawns a fresh subprocess each round
(full Python startup + CUDA init + model load every time).

Requirements:
    - GPU with CUDA support
    - general-gpu venv: /mnt/nfs/venvs/general-gpu
    - MACE model: /mnt/nfs/source/CatEMC_sandip/RuMOx_r2scan_ft_v4c_stagetwo.model
    - catemc on PYTHONPATH

Usage:
    export PATH=/mnt/nfs/venvs/general-gpu/bin:$PATH
    export PYTHONPATH=/path/to/CatEMC:$PYTHONPATH
    python tests/benchmark_worker_pool/test_worker_pool_mace.py
"""

import sys
import os
import numpy as np
import time
import tempfile
import subprocess
import json
from ase import units
from ase.io import read, write

# ============ Configuration ============
MACE_MODEL = os.environ.get(
    "MACE_MODEL",
    "/mnt/nfs/source/CatEMC_sandip/RuMOx_r2scan_ft_v4c_stagetwo.model"
)
POPULATION_FILE = os.environ.get(
    "POPULATION_FILE",
    "/mnt/nfs/RuMOx_catemc/Cu0.1Ru0.9Ox_pet_productionrun/bulk/Cu0.1Ru0.9Ox_pet_productionrun_bulk_input_pop.traj"
)
TARGET_METAL_SPECIES = ['Cu', 'Ru']
FMAX = 0.10
MAX_RELAX_STEPS = 50
NUM_REPLICAS = 3
NUM_ROUNDS = 3  # Simulate multiple EMC generations
GPU_DEVICES = [0]
# =======================================


def load_population():
    pop = read(POPULATION_FILE, index=f':{NUM_REPLICAS}')
    if not isinstance(pop, list):
        pop = [pop]
    return pop


def get_fitness_kwargs():
    return {
        'fmax_list': [FMAX],
        'target_metal_species': TARGET_METAL_SPECIES,
        'md_timestep': 1 * units.fs,
        'md_taut': 0.5 * 1000 * units.fs,
        'md_steps': 50,
        'md_write_interval': 10,
        'max_relax_steps': MAX_RELAX_STEPS,
        'neighbor_cutoff': 3.5,
        'raw_score_cutoff': -10000.,
        'temperature_key': 'temperature',
        'replica_key': 'replica',
    }


def run_worker_pool(population, num_rounds):
    """Worker pool: model loaded once at startup, reused across rounds."""
    from catemc.worker_pool import WorkerPool

    factory_path = os.path.join(tempfile.gettempdir(), '_test_mace_factory.py')
    with open(factory_path, 'w') as f:
        f.write(f'''
def create_calculators():
    from mace.calculators import MACECalculator
    calc = MACECalculator(
        model_paths="{MACE_MODEL}",
        device="cuda",
        default_dtype="float64",
    )
    return [calc]
''')

    if tempfile.gettempdir() not in sys.path:
        sys.path.insert(0, tempfile.gettempdir())

    # Time includes startup (model loading)
    t0 = time.time()
    pool = WorkerPool(
        calculator_factory_module='_test_mace_factory',
        calculator_factory_func='create_calculators',
        calculator_factory_kwargs={},
        fitness_func_module='catemc.calculation_templates',
        fitness_func_name='calculate_bulk_fitness_inprocess',
        fitness_kwargs=get_fitness_kwargs(),
        num_workers=NUM_REPLICAS,
        gpu_devices=GPU_DEVICES,
        startup_timeout=180.0,
    )
    startup_time = time.time() - t0

    # Run multiple rounds (simulating EMC generations)
    all_results = []
    round_times = []
    for i in range(num_rounds):
        tr = time.time()
        results = pool.evaluate(population, run_md=False, fix_indices=None)
        round_times.append(time.time() - tr)
        all_results.append(results)

    total_time = time.time() - t0
    pool.shutdown()
    return all_results, startup_time, round_times, total_time


def run_subprocess_each_round(population, num_rounds):
    """Old approach: spawn a fresh subprocess per round (full reload each time).
    
    This is what CatEMC's calculate_bulk_fitness() does — subprocess.Popen for
    each generation, with full Python startup + CUDA init + model load overhead.
    """
    tmpdir = tempfile.mkdtemp(prefix='mace_bench_')
    
    # Write the subprocess worker script
    worker_script = os.path.join(tmpdir, 'worker.py')
    with open(worker_script, 'w') as f:
        f.write(f'''
import sys, os, time
sys.path.insert(0, os.environ.get("CATEMC_PATH", ""))
from ase.io import read, write
from ase import units
from mace.calculators import MACECalculator
from catemc.calculation_templates import calculate_bulk_fitness_inprocess

t0 = time.time()
calc = MACECalculator(
    model_paths="{MACE_MODEL}",
    device="cuda",
    default_dtype="float64",
)
load_time = time.time() - t0

input_file = sys.argv[1]
output_file = sys.argv[2]

population = read(input_file, index=":")
if not isinstance(population, list):
    population = [population]

fitness_kwargs = {{
    'fmax_list': [{FMAX}],
    'target_metal_species': {TARGET_METAL_SPECIES},
    'md_timestep': 1 * units.fs,
    'md_taut': 0.5 * 1000 * units.fs,
    'md_steps': 50,
    'md_write_interval': 10,
    'max_relax_steps': {MAX_RELAX_STEPS},
    'neighbor_cutoff': 3.5,
    'raw_score_cutoff': -10000.,
    'temperature_key': 'temperature',
    'replica_key': 'replica',
}}

results = []
for atoms in population:
    r = calculate_bulk_fitness_inprocess(
        atoms, calculators=[calc], run_md=False, fix_indices=None,
        **fitness_kwargs
    )
    results.append(r)

write(output_file, results)
# Write timing info
with open(output_file + ".time", "w") as tf:
    tf.write(f"{{load_time:.4f}}")
''')

    # Write input population
    input_file = os.path.join(tmpdir, 'input_pop.traj')
    write(input_file, population)

    all_results = []
    round_times = []
    load_times = []
    total_t0 = time.time()

    catemc_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = os.environ.copy()
    env['CATEMC_PATH'] = catemc_path

    for i in range(num_rounds):
        output_file = os.path.join(tmpdir, f'output_{i}.traj')
        tr = time.time()
        proc = subprocess.run(
            [sys.executable, worker_script, input_file, output_file],
            env=env, capture_output=True, text=True
        )
        round_times.append(time.time() - tr)
        
        if proc.returncode != 0:
            print(f"  ERROR in round {i}: {proc.stderr[-500:]}")
            return None, None, None
        
        results = read(output_file, index=':')
        if not isinstance(results, list):
            results = [results]
        all_results.append(results)
        
        # Read load time
        with open(output_file + ".time") as tf:
            load_times.append(float(tf.read()))

    total_time = time.time() - total_t0
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    return all_results, round_times, total_time, load_times


def main():
    print("=" * 70)
    print("TEST: Worker Pool Equivalence & Speedup — MACE")
    print("=" * 70)
    print(f"Model: {MACE_MODEL}")
    print(f"Replicas: {NUM_REPLICAS}, max_relax_steps: {MAX_RELAX_STEPS}")
    print(f"Rounds: {NUM_ROUNDS} (simulating EMC generations)")
    print()

    population = load_population()
    print(f"Loaded {len(population)} structures ({len(population[0])} atoms each)")

    # Worker pool (persistent model)
    print("\n--- Worker Pool Mode (model loaded once) ---")
    wp_results, wp_startup, wp_rounds, wp_total = run_worker_pool(population, NUM_ROUNDS)
    wp_energies = [[r.info['key_value_pairs']['potential_energy'] for r in rnd] for rnd in wp_results]
    print(f"  Startup (model load): {wp_startup:.2f}s")
    print(f"  Round times: {[f'{t:.2f}s' for t in wp_rounds]}")
    print(f"  Total: {wp_total:.2f}s")
    print(f"  Last-round energies: {[f'{e:.4f}' for e in wp_energies[-1]]}")

    # Subprocess approach (model reloaded each round in fresh process)
    print("\n--- Subprocess Each Round (old approach — fresh process) ---")
    sp_results, sp_rounds, sp_total, sp_loads = run_subprocess_each_round(population, NUM_ROUNDS)
    if sp_results is None:
        print("  FAILED — see error above")
        return 1
    sp_energies = [[r.info['key_value_pairs']['potential_energy'] for r in rnd] for rnd in sp_results]
    print(f"  Round times: {[f'{t:.2f}s' for t in sp_rounds]}")
    print(f"  Model load times: {[f'{t:.2f}s' for t in sp_loads]}")
    print(f"  Total: {sp_total:.2f}s")
    print(f"  Last-round energies: {[f'{e:.4f}' for e in sp_energies[-1]]}")

    # Compare energies (last round — same input so should match)
    print("\n--- Equivalence Check ---")
    e_diff = np.array(wp_energies[-1]) - np.array(sp_energies[-1])
    max_diff = np.max(np.abs(e_diff))
    print(f"  Energy diffs (eV): max={max_diff:.6f}")
    print(f"  Per-replica: {[f'{d:.6f}' for d in e_diff]}")

    # Speedup
    print("\n--- Performance ---")
    wp_eval_only = sum(wp_rounds)
    sp_eval_only = sum(sp_rounds)
    print(f"  Worker pool total ({NUM_ROUNDS} rounds incl startup): {wp_total:.2f}s")
    print(f"  Subprocess total ({NUM_ROUNDS} rounds): {sp_total:.2f}s")
    print(f"  Speedup: {sp_total/wp_total:.1f}x")
    avg_load = np.mean(sp_loads)
    avg_round_wp = np.mean(wp_rounds)
    print(f"  Avg model load per subprocess: {avg_load:.2f}s")
    print(f"  Avg worker pool round (no reload): {avg_round_wp:.2f}s")

    TOLERANCE = 0.05
    print("\n--- Verdict ---")
    if max_diff < TOLERANCE:
        print(f"  PASS: max diff ({max_diff:.4f} eV) < tolerance ({TOLERANCE} eV)")
        return 0
    else:
        print(f"  FAIL: max diff ({max_diff:.4f} eV) >= tolerance ({TOLERANCE} eV)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
