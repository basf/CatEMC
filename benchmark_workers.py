#!/usr/bin/env python
"""Benchmark: persistent worker pool vs subprocess spawning.

Run from /mnt/nfs/source/CatEMC_sandip/CatEMC with:
    python benchmark_workers.py --mode workers --generations 10
    python benchmark_workers.py --mode subprocess --generations 10
"""

import argparse
import sys
import os
import time
import numpy as np

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

# Must set environment before imports
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from ase.io import read, write
from ase import units


# ============ Configuration ============
COMPILED_MODEL = "/mnt/nfs/RuMOx_allegro/NequIP-OAM-S-stage6-oracle.nequip.pt2"
POPULATION_DIR = "/mnt/nfs/RuMOx_catemc/Cu0.1Ru0.9Ox_pet_productionrun/bulk"
TARGET_METAL_SPECIES = ['Cu', 'Ru']
FMAX = 0.10
MAX_RELAX_STEPS = 100
MD_STEPS = 200
NUM_WORKERS = 20
GPU_DEVICES = [0]
# =======================================


def get_calculator_factory():
    """Factory function that workers call to load the model."""
    def factory():
        from nequip.integrations.ase import NequIPCalculator
        calc = NequIPCalculator.from_compiled_model(COMPILED_MODEL, device="cuda")
        return [calc]
    return factory


def run_worker_pool_benchmark(n_generations, population):
    """Run using persistent worker pool."""
    from catemc.worker_pool import WorkerPool
    from catemc.calculation_templates import calculate_bulk_fitness_inprocess
    
    # For workers, we need a factory module. Let's write a small one.
    factory_path = "/tmp/_benchmark_calc_factory.py"
    with open(factory_path, 'w') as f:
        f.write(f"""
import sys, os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

def create_calculators():
    from nequip.integrations.ase import NequIPCalculator
    calc = NequIPCalculator.from_compiled_model(
        "{COMPILED_MODEL}", device="cuda")
    return [calc]
""")
    
    # Ensure /tmp is in path
    if '/tmp' not in sys.path:
        sys.path.insert(0, '/tmp')
    
    fitness_kwargs = {
        'fmax_list': [FMAX],
        'target_metal_species': TARGET_METAL_SPECIES,
        'md_timestep': 1*units.fs,
        'md_taut': 0.5*1000*units.fs,
        'md_steps': MD_STEPS,
        'md_write_interval': 10,
        'max_relax_steps': MAX_RELAX_STEPS,
        'neighbor_cutoff': 3.5,
        'raw_score_cutoff': -10000.,
        'temperature_key': 'temperature',
        'replica_key': 'replica',
    }
    
    print(f"Starting worker pool with {NUM_WORKERS} workers...")
    t0 = time.time()
    pool = WorkerPool(
        calculator_factory_module='_benchmark_calc_factory',
        calculator_factory_func='create_calculators',
        calculator_factory_kwargs={},
        fitness_func_module='catemc.calculation_templates',
        fitness_func_name='calculate_bulk_fitness_inprocess',
        fitness_kwargs=fitness_kwargs,
        num_workers=NUM_WORKERS,
        gpu_devices=GPU_DEVICES,
        startup_timeout=120.0,
    )
    startup_time = time.time() - t0
    print(f"Workers ready in {startup_time:.1f}s")
    
    gen_times = []
    for gen in range(n_generations):
        t_gen = time.time()
        # Simulate: evaluate all structures (relax only, no MD for MC steps)
        results = pool.evaluate(
            population,
            run_md=(gen % 10 == 0),  # MD every 10 generations
            fix_indices=None,
        )
        elapsed = time.time() - t_gen
        gen_times.append(elapsed)
        
        energies = [r.info['key_value_pairs'].get('potential_energy', np.nan) for r in results]
        print(f"  Gen {gen+1}/{n_generations}: {elapsed:.2f}s, "
              f"E_mean={np.mean(energies):.2f} eV")
    
    pool.shutdown()
    return gen_times, startup_time


def run_subprocess_benchmark(n_generations, population):
    """Run using the original subprocess method (baseline)."""
    from catemc.geemc import GEEMC
    
    # We just time get_relaxed_candidates directly
    # Create a minimal GEEMC-like setup
    calc_dir = POPULATION_DIR
    calc_script = "calculate_fitness.py"
    
    from catemc.emc import EMC
    
    # Create a minimal EMC instance for benchmarking
    emc = EMC(
        name="benchmark_subprocess",
        calculation_dir=calc_dir,
        calculation_script=calc_script,
        input_pop=population,
        fix_indices=None,
        gpu_devices=GPU_DEVICES,
        procs_per_device=NUM_WORKERS,
    )
    
    tmp_dir = f"{calc_dir}/tmp_benchmark_subprocess"
    os.makedirs(tmp_dir, exist_ok=True)
    emc.name = "benchmark_subprocess"
    
    gen_times = []
    for gen in range(n_generations):
        t_gen = time.time()
        run_md = (gen % 10 == 0)
        
        # Use the subprocess method
        results = emc.get_relaxed_candidates(population, run_md=run_md)
        elapsed = time.time() - t_gen
        gen_times.append(elapsed)
        
        energies = [r.info['key_value_pairs'].get('potential_energy', np.nan) 
                    for r in results if 'key_value_pairs' in r.info]
        print(f"  Gen {gen+1}/{n_generations}: {elapsed:.2f}s, "
              f"E_mean={np.nanmean(energies):.2f} eV")
    
    return gen_times, 0.0


def load_population():
    """Load the initial population from the production run."""
    pop_file = f"{POPULATION_DIR}/tmp_Cu0.1Ru0.9Ox_pet/emc_0.traj"
    if os.path.exists(pop_file):
        # Read last frame from each replica's trajectory
        pop = []
        for i in range(48):
            traj_file = f"{POPULATION_DIR}/tmp_Cu0.1Ru0.9Ox_pet/emc_{i}.traj"
            if os.path.exists(traj_file):
                atoms = read(traj_file, index=-1)
                pop.append(atoms)
        if pop:
            return pop
    
    # Fallback: read from initial population
    init_file = f"{POPULATION_DIR}/initial_population.traj"
    if os.path.exists(init_file):
        return read(init_file, index=':')
    
    raise FileNotFoundError(
        f"No population found. Check {POPULATION_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CatEMC worker pool")
    parser.add_argument('--mode', choices=['workers', 'subprocess', 'both'], 
                        default='workers', help="Which mode to benchmark")
    parser.add_argument('--generations', type=int, default=10,
                        help="Number of generations to run")
    parser.add_argument('--replicas', type=int, default=None,
                        help="Limit number of replicas (default: all)")
    args = parser.parse_args()
    
    print("Loading population...")
    population = load_population()
    if args.replicas:
        population = population[:args.replicas]
    print(f"Loaded {len(population)} structures "
          f"({len(population[0])} atoms each)")
    
    if args.mode in ('workers', 'both'):
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Persistent Worker Pool ({args.generations} gen)")
        print(f"{'='*60}")
        worker_times, startup = run_worker_pool_benchmark(args.generations, population)
        print(f"\nWorker pool results:")
        print(f"  Startup: {startup:.1f}s")
        print(f"  Mean gen time: {np.mean(worker_times):.2f}s")
        print(f"  Total (excl startup): {sum(worker_times):.1f}s")
        print(f"  Throughput: {len(population)/np.mean(worker_times):.1f} structures/s")
    
    if args.mode in ('subprocess', 'both'):
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Subprocess Spawning ({args.generations} gen)")
        print(f"{'='*60}")
        sub_times, _ = run_subprocess_benchmark(args.generations, population)
        print(f"\nSubprocess results:")
        print(f"  Mean gen time: {np.mean(sub_times):.2f}s")
        print(f"  Total: {sum(sub_times):.1f}s")
        print(f"  Throughput: {len(population)/np.mean(sub_times):.1f} structures/s")
    
    if args.mode == 'both':
        speedup = np.mean(sub_times) / np.mean(worker_times)
        print(f"\n{'='*60}")
        print(f"SPEEDUP: {speedup:.1f}x (worker pool vs subprocess)")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
