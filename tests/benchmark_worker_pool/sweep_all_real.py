#!/usr/bin/env python
"""Full benchmark sweep: worker pool vs subprocess for all models, all round counts.
All numbers are REAL measurements. Results saved to sweep_results.json.
"""
import sys
import os
import time
import tempfile
import subprocess
import json
import numpy as np
from ase import units
from ase.io import read, write

# ============ Configuration ============
POPULATION_FILE = "/mnt/nfs/RuMOx_catemc/Cu0.1Ru0.9Ox_pet_productionrun/bulk/Cu0.1Ru0.9Ox_pet_productionrun_bulk_input_pop.traj"
TARGET_METAL_SPECIES = ['Cu', 'Ru']
FMAX = 0.10
MAX_RELAX_STEPS = 50
NUM_REPLICAS = 3
GPU_DEVICES = [0]

MODEL_TYPE = os.environ.get("BENCH_MODEL", "mace")

MACE_MODEL = "/mnt/nfs/source/CatEMC_sandip/RuMOx_r2scan_ft_v4c_stagetwo.model"
NEQUIP_MODEL = "/mnt/nfs/RuMOx_allegro/NequIP-OAM-S-stage6-oracle.nequip.pt2"
PET_MODEL = "/mnt/nfs/source/CatEMC_sandip/petmad_ft_v4.pt"

ROUND_COUNTS = {
    'nequip': [3, 6, 9, 12, 15, 30, 60],
    'mace': [3, 6, 9, 12, 15],
    'pet': [3, 6, 9, 12, 15],
}

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.json")
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


def run_worker_pool(population, model_type):
    """Worker pool: model loaded once at startup, reused across rounds."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from catemc.worker_pool import WorkerPool

    factory_name = f'_bench_sweep_{model_type}_factory'
    factory_path = os.path.join(tempfile.gettempdir(), f'{factory_name}.py')

    if model_type == 'nequip':
        factory_code = f'''
import sys, os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
def create_calculators():
    from nequip.integrations.ase import NequIPCalculator
    calc = NequIPCalculator.from_compiled_model("{NEQUIP_MODEL}", device="cuda")
    return [calc]
'''
    elif model_type == 'mace':
        factory_code = f'''
def create_calculators():
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths="{MACE_MODEL}", device="cuda", default_dtype="float64")
    return [calc]
'''
    elif model_type == 'pet':
        factory_code = f'''
def create_calculators():
    from metatomic.torch.ase_calculator import MetatomicCalculator
    calc = MetatomicCalculator("{PET_MODEL}", extensions_directory=None, device="cuda")
    return [calc]
'''

    with open(factory_path, 'w') as f:
        f.write(factory_code)

    if tempfile.gettempdir() not in sys.path:
        sys.path.insert(0, tempfile.gettempdir())

    max_rounds = max(ROUND_COUNTS[model_type])

    t0 = time.time()
    pool = WorkerPool(
        calculator_factory_module=factory_name,
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

    round_times = []
    for i in range(max_rounds):
        tr = time.time()
        pool.evaluate(population, run_md=False, fix_indices=None)
        round_times.append(time.time() - tr)
        elapsed = time.time() - t0
        print(f"  WP round {i+1}/{max_rounds}: {round_times[-1]:.3f}s (cumulative: {elapsed:.1f}s)")

    pool.shutdown()
    total_time = time.time() - t0
    return startup_time, round_times, total_time


def run_subprocess_rounds(population, num_rounds, model_type):
    """Subprocess: fresh process per round (real overhead measurement)."""
    tmpdir = tempfile.mkdtemp(prefix=f'{model_type}_bench_')
    catemc_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    worker_script = os.path.join(tmpdir, 'worker.py')

    if model_type == 'nequip':
        model_import = f'''import sys, os
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
sys.path.insert(0, os.environ["CATEMC_PATH"])
from nequip.integrations.ase import NequIPCalculator
calc = NequIPCalculator.from_compiled_model("{NEQUIP_MODEL}", device="cuda")
'''
    elif model_type == 'mace':
        model_import = f'''import sys, os
sys.path.insert(0, os.environ["CATEMC_PATH"])
from mace.calculators import MACECalculator
calc = MACECalculator(model_paths="{MACE_MODEL}", device="cuda", default_dtype="float64")
'''
    elif model_type == 'pet':
        model_import = f'''import sys, os
sys.path.insert(0, os.environ["CATEMC_PATH"])
from metatomic.torch.ase_calculator import MetatomicCalculator
calc = MetatomicCalculator("{PET_MODEL}", extensions_directory=None, device="cuda")
'''

    with open(worker_script, 'w') as f:
        f.write(f'''import time
t0 = time.time()
{model_import}
from ase import units
from ase.io import read, write
from catemc.calculation_templates import calculate_bulk_fitness_inprocess

load_time = time.time() - t0

input_file = sys.argv[1]
output_file = sys.argv[2]

population = read(input_file, index=":")
if not isinstance(population, list):
    population = [population]

fitness_kwargs = dict(
    fmax_list=[{FMAX}],
    target_metal_species={TARGET_METAL_SPECIES},
    md_timestep=1 * units.fs,
    md_taut=0.5 * 1000 * units.fs,
    md_steps=50,
    md_write_interval=10,
    max_relax_steps={MAX_RELAX_STEPS},
    neighbor_cutoff=3.5,
    raw_score_cutoff=-10000.,
    temperature_key='temperature',
    replica_key='replica',
)

results = []
for atoms in population:
    r = calculate_bulk_fitness_inprocess(
        atoms, calculators=[calc], run_md=False, fix_indices=None,
        **fitness_kwargs
    )
    results.append(r)

write(output_file, results)
with open(output_file + ".time", "w") as tf:
    tf.write(f"{{load_time:.4f}}")
''')

    input_file = os.path.join(tmpdir, 'input_pop.traj')
    write(input_file, population)

    env = os.environ.copy()
    env['CATEMC_PATH'] = catemc_path

    round_times = []
    total_t0 = time.time()

    for i in range(num_rounds):
        output_file = os.path.join(tmpdir, f'output_{i}.traj')
        tr = time.time()
        proc = subprocess.run(
            [sys.executable, worker_script, input_file, output_file],
            env=env, capture_output=True, text=True
        )
        rt = time.time() - tr
        round_times.append(rt)
        if proc.returncode != 0:
            print(f"  SP round {i+1}/{num_rounds}: {rt:.2f}s FAILED!")
            print(f"  stderr: {proc.stderr[-500:]}")
            return None, None
        else:
            print(f"  SP round {i+1}/{num_rounds}: {rt:.2f}s OK")

    total_time = time.time() - total_t0

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return round_times, total_time


def main():
    model_type = MODEL_TYPE
    print("=" * 70)
    print(f"BENCHMARK SWEEP: {model_type.upper()}")
    print(f"Rounds to test: {ROUND_COUNTS[model_type]}")
    print("=" * 70)

    population = load_population()
    print(f"Loaded {len(population)} structures ({len(population[0])} atoms each)")

    # --- Worker Pool ---
    max_rounds = max(ROUND_COUNTS[model_type])
    print(f"\n{'='*40}")
    print(f"WORKER POOL — running {max_rounds} rounds")
    print(f"{'='*40}")
    wp_startup, wp_round_times, wp_total = run_worker_pool(population, model_type)
    print(f"\nWP startup: {wp_startup:.2f}s")
    print(f"WP avg round (excl first): {np.mean(wp_round_times[1:]):.3f}s")

    # --- Subprocess ---
    max_sp_rounds = max(ROUND_COUNTS[model_type])
    print(f"\n{'='*40}")
    print(f"SUBPROCESS — running {max_sp_rounds} rounds")
    print(f"{'='*40}")
    sp_round_times, sp_total = run_subprocess_rounds(population, max_sp_rounds, model_type)

    if sp_round_times is None:
        print("SUBPROCESS FAILED — aborting")
        return 1

    # --- Compile results ---
    print(f"\n{'='*70}")
    print(f"RESULTS: {model_type.upper()}")
    print(f"{'='*70}")
    print(f"{'Rounds':<8} {'WP Total (s)':<14} {'SP Total (s)':<14} {'Speedup':<10}")
    print("-" * 46)

    results = {}
    for n in ROUND_COUNTS[model_type]:
        wp_t = wp_startup + sum(wp_round_times[:n])
        sp_t = sum(sp_round_times[:n])
        speedup = sp_t / wp_t
        print(f"{n:<8} {wp_t:<14.2f} {sp_t:<14.2f} {speedup:<10.2f}x")
        results[str(n)] = {
            'wp_total': round(wp_t, 2),
            'sp_total': round(sp_t, 2),
            'speedup': round(speedup, 2),
        }

    results['metadata'] = {
        'wp_startup': round(wp_startup, 2),
        'wp_avg_round_excl_first': round(float(np.mean(wp_round_times[1:])), 3),
        'sp_avg_round': round(float(np.mean(sp_round_times)), 2),
        'wp_round_times': [round(t, 3) for t in wp_round_times],
        'sp_round_times': [round(t, 2) for t in sp_round_times],
    }

    # Load existing results and merge
    all_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)

    all_results[model_type] = results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
