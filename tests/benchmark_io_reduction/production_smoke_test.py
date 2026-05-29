"""Quick smoke test: run the production EMC setup for 5 steps, verify DB and traj output."""
import sys
import os
import shutil
import time

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
sys.path.insert(0, '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC')
sys.path.insert(0, '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk')

from catemc.geemc import EMC
from catemc.worker_pool import WorkerPool
from ase.io import read
from ase.db import connect
from ase import units


def main():
    WORK_DIR = '/fs/cephfs/data/qm_inorganics/sandip/emc_prod_smoke_test'
    INPUT_POP = '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk/Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj'
    SYSTEM_NAME = 'Cu0.1Ru0.9Ox_nequip_productionrun_bulk'
    NUM_STEPS = 5
    NUM_WORKERS = 1
    TRAJ_WRITE_INTERVAL = 1

    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

    input_pop = read(INPUT_POP, ':')
    print(f"Input: {len(input_pop)} replicas, {len(input_pop[0])} atoms each")

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

    print(f"Starting WorkerPool with {NUM_WORKERS} worker(s)...")
    pool = WorkerPool(
        calculator_factory_module='calc_factory_nequip',
        calculator_factory_func='create_calculators',
        calculator_factory_kwargs={},
        fitness_func_module='catemc.calculation_templates',
        fitness_func_name='calculate_bulk_fitness_inprocess',
        fitness_kwargs=fitness_kwargs,
        num_workers=NUM_WORKERS,
        gpu_devices=[0],
        startup_timeout=120,
    )
    print("WorkerPool ready.")

    t0 = time.perf_counter()
    emc = EMC(SYSTEM_NAME,
              calculation_dir=WORK_DIR,
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
              procs_per_device=NUM_WORKERS,
              worker_pool=pool,
              traj_write_interval=TRAJ_WRITE_INTERVAL)

    emc.run(total_steps=NUM_STEPS)
    elapsed = time.perf_counter() - t0
    pool.shutdown()

    # Validate outputs
    print(f"\n{'='*60}")
    print(f"SMOKE TEST RESULTS ({elapsed:.1f}s total)")
    print(f"{'='*60}")

    db_path = f'{WORK_DIR}/{SYSTEM_NAME}.db'
    db = connect(db_path)
    n_relaxed = db.count(relaxed=1)
    n_unrelaxed = db.count(relaxed=0)
    print(f"DB: {n_relaxed} relaxed, {n_unrelaxed} unrelaxed")
    expected_relaxed = len(input_pop) * (NUM_STEPS + 1)
    print(f"Expected relaxed: {expected_relaxed}")
    db_pass = n_relaxed == expected_relaxed

    traj_dir = f'{WORK_DIR}/tmp_{SYSTEM_NAME}'
    traj_count = len([f for f in os.listdir(traj_dir) if f.startswith('emc_') and f.endswith('.traj')])
    print(f"Traj files: {traj_count} (expected {len(input_pop)})")
    traj_pass = traj_count == len(input_pop)

    from ase.io import read as ase_read
    traj0 = ase_read(f'{traj_dir}/emc_0.traj', ':')
    print(f"Traj 0 length: {len(traj0)} frames (expected {NUM_STEPS + 1})")
    traj_len_pass = len(traj0) == NUM_STEPS + 1

    all_pass = db_pass and traj_pass and traj_len_pass
    print(f"\nDB count:    {'PASS' if db_pass else 'FAIL'}")
    print(f"Traj files:  {'PASS' if traj_pass else 'FAIL'}")
    print(f"Traj length: {'PASS' if traj_len_pass else 'FAIL'}")
    print(f"{'='*60}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
