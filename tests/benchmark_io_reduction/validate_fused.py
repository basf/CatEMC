#!/usr/bin/env python
"""
Fused validation: runs OLD and NEW DB writing logic side-by-side within the
same EMC run, writing to two separate databases. Both paths receive the EXACT
same relaxed candidates from the same WorkerPool call, eliminating GPU
non-determinism. Any difference proves a code bug.
"""
import sys
import os
import copy
import time
import shutil
import random
import numpy as np
from datetime import datetime

sys.path.insert(0, '/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC')

from ase.io import read, write
from ase.db import connect
from ase.ga.data import DataConnection, PrepareDB
from catemc.emc_population import EMCPopulation
from catemc.emc_operators import EMCCutSpliceCrossover, EMCIntraSwap
from catemc.worker_pool import WorkerPool

# Configuration
INPUT_POP = '/fs/cephfs/data/qm_inorganics/tiago/emc_oer/Cu0.1Ru0.9Ox_nequip_productionrun/bulk/Cu0.1Ru0.9Ox_nequip_productionrun_bulk_input_pop.traj'
MODEL_PATH = '/fs/cephfs/data/qm_inorganics/allegro-nequip/NequIP-OAM-S-stage6-h100-plain.nequip.pt2'
WORK_DIR = '/fs/cephfs/data/qm_inorganics/sandip/emc_validate_fused'
NUM_STEPS = 10
NUM_WORKERS = 1

TEMPERATURE_KEY = 'temperature'
REPLICA_KEY = 'replica'
ACCEPTANCE_KEY = 'accepted'
MD_GENERATION = 10
EXCHANGE_GENERATION = 1
CROSSOVER_RATE = 0.0
INTRASWAP_RATE = 1.0
METAL_ELEMENT_POOL = ['Cu', 'Ru', 'X']
RAW_SCORE_CUTOFF = -10000.0

FITNESS_KWARGS = {
    'fmax_list': [0.10],
    'target_metal_species': ['Cu', 'Ru', 'X'],
    'md_timestep': 1.0,
    'md_taut': 500.0,
    'md_steps': 200,
    'md_write_interval': 10,
    'max_relax_steps': 100,
    'neighbor_cutoff': 3.5,
    'raw_score_cutoff': -10000.,
    'temperature_key': 'temperature',
    'replica_key': 'replica',
}


def create_db(db_path, input_pop, pop_size):
    if os.path.exists(db_path):
        os.remove(db_path)
    asedb = PrepareDB(db_path, population_size=pop_size)
    for atoms in input_pop:
        kwargs = {
            TEMPERATURE_KEY: atoms.info['key_value_pairs'][TEMPERATURE_KEY],
            REPLICA_KEY: atoms.info['key_value_pairs'][REPLICA_KEY],
            ACCEPTANCE_KEY: 0
        }
        asedb.add_unrelaxed_candidate(atoms,
                                      emc_step=atoms.info['key_value_pairs']['emc_step'],
                                      data=atoms.info.get('data', {}),
                                      **kwargs)
    return DataConnection(db_path)


def get_pop(dc, pop_size):
    return EMCPopulation(
        data_connection=dc,
        population_size=pop_size,
        logfile='/dev/null',
        temperature_key=TEMPERATURE_KEY,
        replica_key=REPLICA_KEY,
        acceptance_key=ACCEPTANCE_KEY,
        category_key=None,
        verbose=False
    )


def update_pop_inplace(pop, relaxed_candidates):
    """NEW logic: update population in-memory."""
    for cand in relaxed_candidates:
        if cand.info['key_value_pairs'][ACCEPTANCE_KEY]:
            rep = cand.info['key_value_pairs'][REPLICA_KEY]
            for i, a in enumerate(pop.pop):
                if a.info['key_value_pairs'][REPLICA_KEY] == rep:
                    pop.pop[i] = cand
                    break
    pop.current_fitness = pop.__get_fitness__(pop.pop)


def compare_populations(pop_old, pop_new, step, label=""):
    old_by_rep = {a.info['key_value_pairs'][REPLICA_KEY]: a for a in pop_old.pop}
    new_by_rep = {a.info['key_value_pairs'][REPLICA_KEY]: a for a in pop_new.pop}
    assert set(old_by_rep.keys()) == set(new_by_rep.keys()), \
        f"Step {step} {label}: replica sets differ!"

    max_e_diff = 0.0
    max_pos_diff = 0.0
    mismatches = []

    for rep in sorted(old_by_rep.keys()):
        a_old = old_by_rep[rep]
        a_new = new_by_rep[rep]
        e_old = a_old.info['key_value_pairs'].get('raw_score', 0)
        e_new = a_new.info['key_value_pairs'].get('raw_score', 0)
        e_diff = abs(e_old - e_new)
        max_e_diff = max(max_e_diff, e_diff)
        pos_diff = np.max(np.abs(a_old.get_positions() - a_new.get_positions()))
        max_pos_diff = max(max_pos_diff, pos_diff)
        if a_old.get_chemical_symbols() != a_new.get_chemical_symbols():
            mismatches.append(f"  rep {rep}: symbols differ!")
        acc_old = a_old.info['key_value_pairs'][ACCEPTANCE_KEY]
        acc_new = a_new.info['key_value_pairs'][ACCEPTANCE_KEY]
        if acc_old != acc_new:
            mismatches.append(f"  rep {rep}: acceptance {acc_old} vs {acc_new}")
        if e_diff > 1e-10:
            mismatches.append(f"  rep {rep}: energy diff {e_diff:.2e} eV")
        if pos_diff > 1e-10:
            mismatches.append(f"  rep {rep}: position diff {pos_diff:.2e} A")
    return max_e_diff, max_pos_diff, mismatches


def main():
    print(f"{'='*70}")
    print(f"FUSED VALIDATION: OLD vs NEW DB logic (same relaxed candidates)")
    print(f"{'='*70}")
    print(f"Steps: {NUM_STEPS}, Workers: {NUM_WORKERS}")
    print(f"Input: {INPUT_POP}")
    print()

    os.makedirs(WORK_DIR, exist_ok=True)
    old_dir = os.path.join(WORK_DIR, 'old_logic')
    new_dir = os.path.join(WORK_DIR, 'new_logic')
    for d in [old_dir, new_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    input_pop = read(INPUT_POP, ':')
    pop_size = len(input_pop)
    print(f"Population: {pop_size} replicas, {len(input_pop[0])} atoms each")

    db_old_path = os.path.join(old_dir, 'fused.db')
    db_new_path = os.path.join(new_dir, 'fused.db')
    dc_old = create_db(db_old_path, input_pop, pop_size)
    dc_new = create_db(db_new_path, input_pop, pop_size)

    print(f"\nStarting WorkerPool with {NUM_WORKERS} worker(s)...")
    wp = WorkerPool(
        calculator_factory_module='calc_factory_nequip',
        calculator_factory_func='create_calculators',
        calculator_factory_kwargs={},
        fitness_func_module='catemc.calculation_templates',
        fitness_func_name='calculate_bulk_fitness_inprocess',
        fitness_kwargs=FITNESS_KWARGS,
        num_workers=NUM_WORKERS,
        gpu_devices=[0],
        startup_timeout=120,
    )
    print("WorkerPool ready.\n")

    # Step 0: initial relaxation
    print("Step 0: Initial relaxation...")
    pop_old = get_pop(dc_old, pop_size)
    pop_new = get_pop(dc_new, pop_size)

    unrelaxed = dc_old.get_all_unrelaxed_candidates()
    relaxed_candidates = wp.evaluate(unrelaxed, run_md=True, fix_indices=None)
    for atoms in relaxed_candidates:
        atoms.info['key_value_pairs'][ACCEPTANCE_KEY] = 1

    # OLD
    dc_old.add_more_relaxed_candidates(copy.deepcopy(relaxed_candidates))
    pop_old.update()
    # NEW
    dc_new.add_more_relaxed_candidates(copy.deepcopy(relaxed_candidates))
    pop_new.update()

    max_e, max_pos, issues = compare_populations(pop_old, pop_new, 0, "init")
    status = "PASS" if max_e < 1e-10 and max_pos < 1e-10 else "FAIL"
    print(f"  Step 0: max_e_diff={max_e:.2e}, max_pos_diff={max_pos:.2e} -> {status}")

    pending_db_new = []
    rates_sum = INTRASWAP_RATE + CROSSOVER_RATE
    intraswap_rate = INTRASWAP_RATE / rates_sum
    crossover_rate = CROSSOVER_RATE / rates_sum

    print(f"\n{'='*70}")
    print(f"Running {NUM_STEPS} EMC steps with side-by-side comparison...")
    print(f"{'='*70}")

    all_pass = True
    for step in range(1, NUM_STEPS + 1):
        t0 = time.perf_counter()

        # MD generation
        if step % MD_GENERATION == 0:
            unrelaxed_md = list(pop_old.pop)
            relaxed_md = wp.evaluate(unrelaxed_md, run_md=True, fix_indices=None)
            for atoms in relaxed_md:
                atoms.info['key_value_pairs'][ACCEPTANCE_KEY] = 1
            dc_old.add_more_relaxed_candidates(copy.deepcopy(relaxed_md))
            pop_old.update()
            update_pop_inplace(pop_new, copy.deepcopy(relaxed_md))
            pending_db_new.extend(copy.deepcopy(relaxed_md))

        # Mutation/Crossover — use pop_old parents (both pops should be identical)
        n_crossover_pairs = round((pop_size * crossover_rate) / 2)
        all_parents_old = pop_old.get_all_parents(n_crossover_pairs, n_interswap_pairs=0)

        offspring_cache = {}
        for parents in all_parents_old:
            if len(parents) == 2:
                operator = EMCCutSpliceCrossover(
                    allowed_indices=None, elements=METAL_ELEMENT_POOL, keep_composition=True)
                offsprings, desc = operator.get_new_individual(parents)
                for idx in range(2):
                    rep = parents[idx].info['key_value_pairs'][REPLICA_KEY]
                    offsprings[idx].info['key_value_pairs'][TEMPERATURE_KEY] = \
                        parents[idx].info['key_value_pairs'][TEMPERATURE_KEY]
                    offsprings[idx].info['key_value_pairs'][REPLICA_KEY] = rep
                    offspring_cache[rep] = offsprings[idx]
            elif len(parents) == 1:
                rep = parents[0].info['key_value_pairs'][REPLICA_KEY]
                operator = EMCIntraSwap(
                    allowed_indices=None, element_pools=METAL_ELEMENT_POOL)
                offspring, desc = operator.get_new_individual(parents)
                offspring.info['key_value_pairs'][TEMPERATURE_KEY] = \
                    parents[0].info['key_value_pairs'][TEMPERATURE_KEY]
                offspring.info['key_value_pairs'][REPLICA_KEY] = rep
                offspring_cache[rep] = offspring

        replicas = sorted(offspring_cache.keys())
        unrelaxed_offsprings = [offspring_cache[rep] for rep in replicas]
        relaxed_offsprings = wp.evaluate(unrelaxed_offsprings, run_md=False, fix_indices=None)

        # Evaluate acceptance
        relaxed_candidates = []
        for parents in all_parents_old:
            if len(parents) == 2:
                rep1 = parents[0].info['key_value_pairs'][REPLICA_KEY]
                rep2 = parents[1].info['key_value_pairs'][REPLICA_KEY]
                r1 = relaxed_offsprings[replicas.index(rep1)]
                r2 = relaxed_offsprings[replicas.index(rep2)]
                r1.info['data']['operation'] = parents[0].info['data']['for_operation']
                r2.info['data']['operation'] = parents[1].info['data']['for_operation']
                r1.info['key_value_pairs']['emc_step'] = step
                r2.info['key_value_pairs']['emc_step'] = step
                relaxed_candidates += pop_old.evaluate_offsprings(
                    parents, [r1, r2], pressure=None, raw_score_cutoff=RAW_SCORE_CUTOFF)
            elif len(parents) == 1:
                rep = parents[0].info['key_value_pairs'][REPLICA_KEY]
                r = relaxed_offsprings[replicas.index(rep)]
                r.info['data']['operation'] = parents[0].info['data']['for_operation']
                r.info['key_value_pairs']['emc_step'] = step
                relaxed_candidates += pop_old.evaluate_offsprings(
                    parents, [r], pressure=None, raw_score_cutoff=RAW_SCORE_CUTOFF)

        # Apply to BOTH paths with same candidates
        candidates_for_old = copy.deepcopy(relaxed_candidates)
        candidates_for_new = copy.deepcopy(relaxed_candidates)

        # OLD: write immediately + pop.update()
        dc_old.add_more_relaxed_candidates(candidates_for_old)
        pop_old.update()

        # NEW: in-memory update + batch queue
        update_pop_inplace(pop_new, candidates_for_new)
        pending_db_new.extend(candidates_for_new)

        # Flush new DB every step for comparison
        dc_new.add_more_relaxed_candidates(pending_db_new)
        pending_db_new = []
        pop_new_from_db = get_pop(dc_new, pop_size)

        # Exchange (sync BOTH np.random and random states)
        if EXCHANGE_GENERATION > 0 and step % EXCHANGE_GENERATION == 0:
            np_state = np.random.get_state()
            py_state = random.getstate()
            pop_old.exchange(keep_composition=True)
            np.random.set_state(np_state)
            random.setstate(py_state)
            pop_new.exchange(keep_composition=True)

        # Compare
        max_e, max_pos, issues = compare_populations(pop_old, pop_new, step, "in-memory")
        max_e_db, max_pos_db, issues_db = compare_populations(pop_old, pop_new_from_db, step, "from-DB")

        elapsed = time.perf_counter() - t0
        status_mem = "PASS" if max_e < 1e-10 and max_pos < 1e-10 and not issues else "FAIL"
        status_db = "PASS" if max_e_db < 1e-10 and max_pos_db < 1e-10 and not issues_db else "FAIL"
        if status_mem != "PASS" or status_db != "PASS":
            all_pass = False
        print(f"  Step {step:3d}: in-mem={status_mem} (e={max_e:.2e}, pos={max_pos:.2e}) | "
              f"DB-read={status_db} (e={max_e_db:.2e}, pos={max_pos_db:.2e}) | {elapsed:.1f}s")
        if issues:
            for iss in issues[:3]:
                print(f"    MEM: {iss}")
        if issues_db:
            for iss in issues_db[:3]:
                print(f"    DB:  {iss}")

    # Final DB comparison
    print(f"\n{'='*70}")
    print("FINAL DB COMPARISON")
    print(f"{'='*70}")
    db_old = connect(db_old_path)
    db_new = connect(db_new_path)
    n_old = db_old.count(relaxed=1)
    n_new = db_new.count(relaxed=1)
    print(f"  Old DB: {n_old} relaxed rows")
    print(f"  New DB: {n_new} relaxed rows")
    print(f"  Count match: {'PASS' if n_old == n_new else 'FAIL'}")

    rows_old = sorted(db_old.select(relaxed=1), key=lambda r: (r.key_value_pairs['emc_step'], r.key_value_pairs[REPLICA_KEY]))
    rows_new = sorted(db_new.select(relaxed=1), key=lambda r: (r.key_value_pairs['emc_step'], r.key_value_pairs[REPLICA_KEY]))
    max_final_e = 0.0
    for ro, rn in zip(rows_old, rows_new):
        e_diff = abs(ro.key_value_pairs['raw_score'] - rn.key_value_pairs['raw_score'])
        max_final_e = max(max_final_e, e_diff)
    print(f"  Max energy diff across all DB rows: {max_final_e:.2e} eV")

    print(f"\n{'='*70}")
    if all_pass and max_final_e < 1e-10:
        print("RESULT: ALL STEPS PASS - OLD and NEW logic produce IDENTICAL results")
    else:
        print("RESULT: MISMATCH DETECTED - investigate above failures")
    print(f"{'='*70}")
    wp.shutdown()


if __name__ == '__main__':
    main()
