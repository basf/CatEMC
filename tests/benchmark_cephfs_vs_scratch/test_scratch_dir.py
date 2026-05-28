"""Quick functional test of the new scratch_dir parameter.

Runs 20 gens with 24 workers, DB on /mnt/scratch via scratch_dir param.
Verifies:
  - DB is created on scratch during run
  - DB is synced back to CephFS at end
  - Performance matches previous ~1.6 s/gen
"""
import sys
import os
import shutil
import time

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
sys.path.insert(0, '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catemc.geemc import EMC
from catemc.worker_pool import WorkerPool
from ase.io import read
from ase import units

INPUT_POP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj')
NUM_STEPS = 20
NUM_WORKERS = 24
CEPHFS_DIR = '/fs/cephfs/data/qm_inorganics/sandip/emc_scratch_dir_test'
SCRATCH_DIR = '/mnt/scratch/emc_scratch_dir_test'


def main():
    os.makedirs(CEPHFS_DIR, exist_ok=True)

    input_pop = read(INPUT_POP, index=':')
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

    pool = WorkerPool(
        calculator_factory_module='calc_factory_nequip',
        calculator_factory_func='create_calculators',
        calculator_factory_kwargs={},
        fitness_func_module='catemc.calculation_templates',
        fitness_func_name='calculate_bulk_fitness_inprocess',
        fitness_kwargs=fitness_kwargs,
        num_workers=NUM_WORKERS,
        gpu_devices=[0],
        startup_timeout=600.0,
    )
    print(f"Worker pool ready: {NUM_WORKERS} workers")

    system_name = 'scratch_dir_test'
    emc = EMC(system_name,
              calculation_dir=CEPHFS_DIR,
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
              traj_write_interval=1,
              scratch_dir=SCRATCH_DIR)

    print(f"\ncalculation_dir (CephFS): {CEPHFS_DIR}")
    print(f"scratch_dir (NVMe):       {SCRATCH_DIR}")
    print(f"scratch_dir param:        {emc.scratch_dir}")

    t0 = time.time()
    emc.run(total_steps=NUM_STEPS)
    elapsed = time.time() - t0

    pool.shutdown()

    # Verify results
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {elapsed:.1f}s total, {elapsed/NUM_STEPS:.2f} s/gen")
    print(f"{'=' * 60}")

    # Check DB was synced back to CephFS
    db_on_cephfs = os.path.join(CEPHFS_DIR, f'{system_name}.db')
    if os.path.isfile(db_on_cephfs):
        size = os.path.getsize(db_on_cephfs) / 1024
        print(f"✓ DB synced to CephFS: {db_on_cephfs} ({size:.1f} KB)")
    else:
        print(f"✗ DB NOT found on CephFS: {db_on_cephfs}")

    # Check scratch
    db_on_scratch = os.path.join(SCRATCH_DIR, f'{system_name}.db')
    if os.path.isfile(db_on_scratch):
        size = os.path.getsize(db_on_scratch) / 1024
        print(f"  DB still on scratch: {db_on_scratch} ({size:.1f} KB)")
    else:
        print(f"  DB not on scratch (already cleaned?)")

    # Check traj files on CephFS
    traj_files = [f for f in os.listdir(CEPHFS_DIR) if f.endswith('.traj')]
    print(f"  Traj files on CephFS: {len(traj_files)}")

    print(f"\nExpected ~1.6 s/gen. Got {elapsed/NUM_STEPS:.2f} s/gen.")


if __name__ == '__main__':
    main()
