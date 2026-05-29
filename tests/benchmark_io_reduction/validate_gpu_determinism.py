"""Run OLD code 5 times with identical parameters, compare DBs to prove GPU non-determinism."""
import os, sys, time, shutil, importlib, importlib.util
import numpy as np

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
sys.path.insert(0, '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC')
sys.path.insert(0, '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk')

from ase.db import connect
from ase import units
from ase.io import read
import random

N_RUNS = 5
N_GENS = 5
N_REPLICAS = 48
BASE_DIR = '/fs/cephfs/data/qm_inorganics/sandip/emc_validate_gpu_determinism'
INPUT_POP = '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk/Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj'
EMC_SRC = '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC/catemc'

FITNESS_KWARGS = {
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

def run_old_code(run_idx):
    """Run the OLD (backup) code once."""
    run_dir = f"{BASE_DIR}/run_{run_idx}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Load old code from .bak
    bak_path = f"{EMC_SRC}/emc.py.bak"
    tmp_path = f"{EMC_SRC}/_emc_old_determ_{run_idx}.py"
    shutil.copy2(bak_path, tmp_path)
    # Fix relative imports
    with open(tmp_path, "r") as f:
        code = f.read()
    code = code.replace("from .emc_population", "from catemc.emc_population")
    code = code.replace("from .emc_operators", "from catemc.emc_operators")
    code = code.replace("from .worker_pool", "from catemc.worker_pool")
    with open(tmp_path, "w") as f:
        f.write(code)
    
    spec = importlib.util.spec_from_file_location(f"_emc_old_determ_{run_idx}", tmp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OldEMC = mod.EMC
    
    # Load input population
    input_pop = read(INPUT_POP, index=':')
    
    # Same seed every time
    random.seed(42)
    np.random.seed(42)
    
    from catemc.worker_pool import WorkerPool
    
    # Re-use the same worker pool across runs (it's already started)
    global _worker_pool
    if '_worker_pool' not in globals() or _worker_pool is None:
        _worker_pool = WorkerPool(
            calculator_factory_module='calc_factory_nequip',
            calculator_factory_func='create_calculators',
            fitness_kwargs=FITNESS_KWARGS,
            num_workers=1,
            gpu_devices=[0],
            startup_timeout=120.0,
        )
        print(f"WorkerPool: 1 worker ready")
    
    emc = OldEMC(
        name='determ_test',
        calculation_dir=run_dir,
        input_pop=input_pop,
        worker_pool=_worker_pool,
        metal_element_pool=['Cu', 'Ru'],
        nonmetal_element_pool=None,
        exchange_generation=1,
        intraswap_rate=1.0,
    )
    
    t0 = time.time()
    emc.run(total_steps=N_GENS)
    elapsed = time.time() - t0
    print(f"  Run {run_idx}: {elapsed:.1f}s ({elapsed/N_GENS:.2f}s/gen)")
    
    # Cleanup temp module
    os.remove(tmp_path)
    return f"{run_dir}/determ_test.db"


def compare_dbs(db_paths):
    """Compare all pairs of DBs."""
    print(f"\n{'='*60}")
    print(f"=== Comparing {len(db_paths)} DBs from identical OLD code runs ===")
    print(f"{'='*60}")
    
    dbs = [connect(p) for p in db_paths]
    counts = [db.count(relaxed=1) for db in dbs]
    print(f"\nRow counts: {counts}")
    print(f"All same count: {'YES' if len(set(counts)) == 1 else 'NO'}")
    
    # Get all rows from each DB
    all_rows = [list(db.select(relaxed=1, sort='id')) for db in dbs]
    
    # Pairwise comparison
    print(f"\nPairwise max |raw_score| diffs:")
    print(f"{'':>8}", end='')
    for j in range(len(db_paths)):
        print(f"  run_{j:>2}", end='')
    print()
    
    pair_diffs = {}
    for i in range(len(db_paths)):
        print(f"run_{i:>2}  ", end='')
        for j in range(len(db_paths)):
            if j <= i:
                print(f"{'---':>7}", end='')
            else:
                diffs = []
                for ri, rj in zip(all_rows[i], all_rows[j]):
                    ei = ri.key_value_pairs.get('raw_score', 0)
                    ej = rj.key_value_pairs.get('raw_score', 0)
                    diffs.append(abs(ei - ej))
                max_d = max(diffs)
                mean_d = np.mean(diffs)
                pair_diffs[(i,j)] = (max_d, mean_d)
                print(f"  {max_d:5.2f}", end='')
        print()
    
    print(f"\nPairwise mean |raw_score| diffs:")
    print(f"{'':>8}", end='')
    for j in range(len(db_paths)):
        print(f"  run_{j:>2}", end='')
    print()
    for i in range(len(db_paths)):
        print(f"run_{i:>2}  ", end='')
        for j in range(len(db_paths)):
            if j <= i:
                print(f"{'---':>7}", end='')
            else:
                _, mean_d = pair_diffs[(i,j)]
                print(f"  {mean_d:5.3f}", end='')
        print()
    
    # Check early gens (first 96 rows = gens 0-1)
    print(f"\nEarly gens (0-1) max diff between all pairs:")
    early_max = 0
    for i in range(len(db_paths)):
        for j in range(i+1, len(db_paths)):
            for ri, rj in zip(all_rows[i][:96], all_rows[j][:96]):
                ei = ri.key_value_pairs.get('raw_score', 0)
                ej = rj.key_value_pairs.get('raw_score', 0)
                early_max = max(early_max, abs(ei - ej))
    print(f"  {early_max:.2e} eV")
    
    # Summary
    all_max = max(v[0] for v in pair_diffs.values())
    all_mean = np.mean([v[1] for v in pair_diffs.values()])
    mean_energy = np.mean([abs(r.key_value_pairs.get('raw_score', 0)) for r in all_rows[0]])
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Mean energy magnitude: {mean_energy:.1f} eV")
    print(f"  Max diff across all pairs: {all_max:.2e} eV ({all_max/mean_energy*100:.4f}% rel)")
    print(f"  Mean diff across all pairs: {all_mean:.2e} eV ({all_mean/mean_energy*100:.4f}% rel)")
    print(f"  Early gens max diff: {early_max:.2e} eV")
    if all_max > 1e-4:
        print(f"\n  CONCLUSION: GPU non-determinism CONFIRMED")
        print(f"  (Same code, same seed, same GPU → different energies)")
    else:
        print(f"\n  CONCLUSION: Results are deterministic (unexpected)")
    print(f"{'='*60}")


def main():
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    os.makedirs(BASE_DIR, exist_ok=True)
    
    db_paths = []
    for i in range(N_RUNS):
        print(f"\n{'='*50}")
        print(f"Running OLD code: run {i+1}/{N_RUNS}")
        print(f"{'='*50}")
        path = run_old_code(i)
        db_paths.append(path)
    
    # Shutdown worker pool
    global _worker_pool
    if _worker_pool is not None:
        _worker_pool.shutdown()
        _worker_pool = None
    
    compare_dbs(db_paths)


if __name__ == '__main__':
    main()
