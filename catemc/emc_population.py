#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implementation for evolutionary Monte Carlo (EMC) population"""
from .gc_utils import *
from collections import defaultdict, Counter
from functools import partial
from copy import deepcopy
from ase.ga.population import Population
from ase.ga import get_raw_score
from ase.parallel import parprint
from ase.io.jsonio import encode, decode
from ase.units import kB
from scipy.special import logsumexp
import numpy as np
import warnings
import random


def log_prefactor_value(prefactor):

    if prefactor is None or prefactor <= 0:
        return -np.inf
    return np.log(prefactor)


def metropolis_accept_log(ln_ratio):

    return (ln_ratio >= 0.) or (np.log(np.random.uniform()) < ln_ratio)


class EMCPopulation(Population):
    """The EMC population instance that evolves based on
    the Metropolis-Hastings algorithm individually.

    Parameters
    ----------
    population_size: int
        Size of the EMC population.

    """

    def __init__(self, data_connection, population_size, logfile=None,
                 temperature_key='temperature', replica_key='replica',
                 acceptance_key='accepted', category_key=None, 
                 use_extinct=False, store_delta=True, verbose=True, 
                 rng=np.random):
        # The current fitness is set at each update of the population
        self.current_fitness = None
        self.comparator = None
        self.tk = temperature_key
        self.rk = replica_key
        self.ak = acceptance_key
        self.ck = category_key
        self.store_delta = store_delta
        self.verbose = verbose

        Population.__init__(self, data_connection, population_size,
                            self.comparator, logfile, use_extinct, rng=rng)

    def __get_fitness__(self, candidates):

        return [get_raw_score(a) for a in candidates]

    def __initialize_pop__(self):
        # Get all relaxed candidates from the database
        if self.use_extinct:
            entries = self.dc.c.select('relaxed=1,extinct=0', sort='age')
        else:
            entries = self.dc.c.select('relaxed=1', sort='age')

        all_cand = []
        for v in entries:
            c = self.dc.get_atoms(id=v.id)
            c.info['confid'] = v.gaid
            c.info['relax_id'] = v.id
            if 'data' not in c.info:
                c.info['data'] = {}
            all_cand.append(c)

        if len(all_cand) > 0:
            vf = lambda x: x.info['key_value_pairs'][self.rk]
            all_sorted = sorted(all_cand, key=vf)

            # Fill up the population with the self.pop_size most stable
            # unique candidates.
            i = 0
            while i < len(all_sorted) and len(self.pop) < self.pop_size:
                c = all_sorted[i]
                i += 1
                if not c.info['key_value_pairs'][self.ak]:
                    continue
                c_vf = vf(c)
                eq = False
                for a in self.pop:
                    a_vf = vf(a)
                    # Only run comparator if the variable_function (vf)
                    # returns the same. If it returns something different the
                    # candidates are inherently different.
                    # This is done to speed up.
                    if a_vf == c_vf:
                        eq = True
                        break
                if not eq:
                    self.pop.append(c)
        self.pop.sort(key=lambda x: x.info['key_value_pairs'][self.rk])
        # Restore accepted state-only moves as well as accepted structures.
        # Keep historical database rows unchanged: the current population is
        # a small checkpoint of row ids and state metadata.
        checkpoint = self.dc.c.metadata.get('catemc_population')
        if checkpoint is not None:
            by_id = {c.info['relax_id']: c for c in all_cand}
            saved = checkpoint['candidates']
            if all(item['id'] in by_id for item in saved):
                current = {}
                for item in saved:
                    c = deepcopy(by_id[item['id']])
                    c.info = decode(item['info'])
                    current[c.info['key_value_pairs'][self.rk]] = c
                for c in sorted(all_cand, key=lambda a: a.info['relax_id']):
                    if (c.info['relax_id'] > checkpoint['last_id'] and
                            c.info['key_value_pairs'][self.ak]):
                        current[c.info['key_value_pairs'][self.rk]] = c
                self.pop = self._sorted_by_replica(current.values())
        self.current_fitness = self.__get_fitness__(self.pop)
        self.all_cand = all_cand

    def _checkpoint_population(self):
        """Persist state labels/scores without overwriting historical samples."""
        if not hasattr(self, 'dc') or not self.pop:
            return
        if any('relax_id' not in c.info for c in self.pop):
            return
        metadata = self.dc.c.metadata
        metadata['catemc_population'] = {
            'last_id': max(c.info['relax_id'] for c in self.all_cand),
            'candidates': [{'id': c.info['relax_id'], 'info': encode(c.info)}
                           for c in self.pop]}
        self.dc.c.metadata = metadata

    def update(self):
        """ New candidates can be added to the database
            after the population object has been created.
            This method extracts these new candidates from the
            database and includes them in the population. """

        self.pop = []
        self.__initialize_pop__()

        self._checkpoint_population()
        self._write_log()

    def evaluate_offsprings(self, parents, offsprings, prefactor=1, volume=None, 
                            pressure=None, nonmetal_species=None, 
                            temperature_species_internal_map=None,
                            no_internal_indices=None, metal_species_probabilities=None, 
                            mutable_indices=None, raw_score_cutoff=-10000.):

        if len(parents) == 2:
            # Crossover or interswap
            # Will adjust replica and temperature if necessary
            assert len(offsprings) == 2
            # Offspring i belongs to parent/state i.  Sorting by energy here
            # changes the proposal and may use a potential from the wrong state.
            switch_to_crossover = False
            sorted_parents = parents
            sorted_oids = [0, 1]
            sorted_offsprings = offsprings
            df1 = get_raw_score(sorted_offsprings[0]) - get_raw_score(sorted_parents[0])
            df2 = get_raw_score(sorted_offsprings[1]) - get_raw_score(sorted_parents[1])
            rep1 = sorted_parents[0].info['key_value_pairs'][self.rk]
            rep2 = sorted_parents[1].info['key_value_pairs'][self.rk]
            T1 = sorted_parents[0].info['key_value_pairs'][self.tk]
            T2 = sorted_parents[1].info['key_value_pairs'][self.tk]
            offsprings[sorted_oids[0]].info['key_value_pairs'][self.rk] = rep1
            offsprings[sorted_oids[1]].info['key_value_pairs'][self.rk] = rep2
            offsprings[sorted_oids[0]].info['key_value_pairs'][self.tk] = T1
            offsprings[sorted_oids[1]].info['key_value_pairs'][self.tk] = T2
            if self.ck is not None:
                C1 = sorted_parents[0].info['key_value_pairs'][self.ck]
                C2 = sorted_parents[1].info['key_value_pairs'][self.ck]
                offsprings[sorted_oids[0]].info['key_value_pairs'][self.ck] = C1
                offsprings[sorted_oids[1]].info['key_value_pairs'][self.ck] = C2
            if 'mu' in sorted_parents[0].info['data']:
                offsprings[sorted_oids[0]].info['data']['mu'] = sorted_parents[0].info['data']['mu']
                offsprings[sorted_oids[1]].info['data']['mu'] = sorted_parents[1].info['data']['mu']
            if 'mu_window' in sorted_parents[0].info['data']:
                offsprings[sorted_oids[0]].info['data']['mu_window'] = sorted_parents[0].info['data']['mu_window']
                offsprings[sorted_oids[1]].info['data']['mu_window'] = sorted_parents[1].info['data']['mu_window']

            if any(o for o in offsprings if get_raw_score(o) <= raw_score_cutoff):
                prefactor = None
                accept = False
            else:
                proposal_log = offsprings[0].info['data'].get('mc_log_proposal_ratio')
                if proposal_log is not None:
                    prefactor = 1.
                if prefactor == 'gc':
                    if (parents[0].info['data']['for_operation'] == 'crossover') or switch_to_crossover:
                        prefactor = compute_crossover_prefactor(
                                sorted_parents[0], sorted_parents[1],
                                offsprings[sorted_oids[0]], offsprings[sorted_oids[1]],
                                nonmetal_species, temperature_species_internal_map,
                                no_internal_indices, temperature_key=self.tk)
                    elif parents[0].info['data']['for_operation'] == 'interswap':
                        prefactor = compute_interswap_prefactor(
                                sorted_parents[0], sorted_parents[1],
                                offsprings[sorted_oids[0]], offsprings[sorted_oids[1]])

                ln_prefactor = (log_prefactor_value(prefactor) if proposal_log is None
                                else proposal_log)
                if pressure is not None:
                    V11 = sorted_parents[0].info['key_value_pairs']['volume']
                    V12 = sorted_offsprings[0].info['key_value_pairs']['volume']
                    V21 = sorted_parents[1].info['key_value_pairs']['volume']
                    V22 = sorted_offsprings[1].info['key_value_pairs']['volume']
                    N11 = len([a for a in sorted_parents[0] if a.symbol != 'X'])
                    N12 = len([a for a in sorted_offsprings[0] if a.symbol != 'X'])
                    N21 = len([a for a in sorted_parents[1] if a.symbol != 'X'])
                    N22 = len([a for a in sorted_offsprings[1] if a.symbol != 'X'])
                    lnV = N12*np.log(V12) - N11*np.log(V11) + N22*np.log(V22) - N21*np.log(V21)
                    df1 -= pressure * (V12 - V11)
                    df2 -= pressure * (V22 - V21)
                    ln_prefactor += lnV
                ln_p_acc = ln_prefactor + df1 / (kB*T1) + df2 / (kB*T2)
                accept = metropolis_accept_log(ln_p_acc)
            if self.verbose:
                parprint(f"Replicas: {[rep1, rep2]}, " + 
                         f"parents: {[sorted_parents[0].get_chemical_formula(), sorted_parents[1].get_chemical_formula()]}, " +
                         f"offsprings: {[sorted_offsprings[0].get_chemical_formula(), sorted_offsprings[1].get_chemical_formula()]}, " +
                         f"operation: {offsprings[0].info['key_value_pairs']['origin']}, " + 
                         f"deltas: {df1} and {df2}, prefactor: {prefactor}.")
            if accept:
                offsprings[0].info['key_value_pairs'][self.ak] = 1
                offsprings[1].info['key_value_pairs'][self.ak] = 1
                if self.verbose:
                    parprint(f'Replicas {rep1} {rep2} inter move accepted!')
            else:
                offsprings[0].info['key_value_pairs'][self.ak] = 0
                offsprings[1].info['key_value_pairs'][self.ak] = 0
                if self.verbose:
                    parprint(f'Replicas {rep1} {rep2} inter move rejected!')
            if self.store_delta:
                offsprings[sorted_oids[0]].info['key_value_pairs']['delta'] = df1
                offsprings[sorted_oids[1]].info['key_value_pairs']['delta'] = df2
                if pressure is not None:
                    if prefactor is None:
                        offsprings[sorted_oids[0]].info['key_value_pairs']['volume_ratio'] = 1.
                        offsprings[sorted_oids[1]].info['key_value_pairs']['volume_ratio'] = 1.
                    else:
                        offsprings[sorted_oids[0]].info['key_value_pairs']['volume_ratio'] = V12 / V11
                        offsprings[sorted_oids[1]].info['key_value_pairs']['volume_ratio'] = V22 / V21

        elif len(parents) == 1:
            # Mutation
            df = get_raw_score(offsprings[0]) - get_raw_score(parents[0])
            rep = parents[0].info['key_value_pairs'][self.rk]
            offsprings[0].info['key_value_pairs'][self.rk] = rep
            T = parents[0].info['key_value_pairs'][self.tk]
            offsprings[0].info['key_value_pairs'][self.tk] = T
            if self.ck is not None:
                C = parents[0].info['key_value_pairs'][self.ck]
                offsprings[0].info['key_value_pairs'][self.ck] = C
            if any(o for o in offsprings if get_raw_score(o) <= raw_score_cutoff): 
                prefactor = None
                accept = False
            else:
                proposal_log = offsprings[0].info['data'].get('mc_log_proposal_ratio')
                if proposal_log is not None:
                    prefactor = 1.
                if prefactor == 'gc':
                    prefactor = compute_intramove_prefactor(
                            parents[0], offsprings[0], 
                            volume, nonmetal_species, 
                            temperature_species_internal_map,
                            no_internal_indices,
                            metal_species_probabilities,
                            temperature_key=self.tk)
                ln_prefactor = (log_prefactor_value(prefactor) if proposal_log is None
                                else proposal_log)
                if pressure is not None:
                    V1 = parents[0].info['key_value_pairs']['volume']
                    V2 = offsprings[0].info['key_value_pairs']['volume']
                    N1 = len([a for a in parents[0] if a.symbol != 'X'])
                    N2 = len([a for a in offsprings[0] if a.symbol != 'X'])
                    lnV = N2*np.log(V2) - N1*np.log(V1)
                    df -= pressure * (V2 - V1)
                    ln_prefactor += lnV
                ln_p_acc = ln_prefactor + df / (kB*T)
                accept = metropolis_accept_log(ln_p_acc)
            if self.verbose:
                parprint(f"Replica: {rep}, parent: {parents[0].get_chemical_formula()}, " +
                         f"offspring: {offsprings[0].get_chemical_formula()}, " + 
                         f"operation: {offsprings[0].info['key_value_pairs']['origin']}, delta: {df}, " +
                         f"prefactor: {prefactor}.")
            if accept:
                offsprings[0].info['key_value_pairs'][self.ak] = 1
                if self.verbose:
                    parprint(f'Replica {rep} intra move accepted!')
            else:
                offsprings[0].info['key_value_pairs'][self.ak] = 0
                if self.verbose:
                    parprint(f'Replica {rep} intra move rejected!')
            if self.store_delta:
                offsprings[0].info['key_value_pairs']['delta'] = df
                if pressure is not None:
                    if prefactor is None:
                        offsprings[0].info['key_value_pairs']['volume_ratio'] = 1.
                    else:
                        offsprings[0].info['key_value_pairs']['volume_ratio'] = V2 / V1

        return offsprings

    def get_exchange_pairs_between_temperatures(self, keep_composition=False):

        temp_cand_dict = defaultdict(list)                                 
        for c in self.pop:
            temp_cand_dict[c.info['key_value_pairs'][self.tk]].append(c)
        dicts = [deepcopy(temp_cand_dict), deepcopy(temp_cand_dict)]
        keys = sorted(temp_cand_dict.keys())
        if len(keys) == 1:
            prioritize_odd = False
        else:
            prioritize_odd = random.choice([True, False])
        if prioritize_odd:
            first_temps = keys[1:]
            second_temps = keys
        else:
            first_temps = keys
            second_temps = keys[1:]
        change_strat = True
        for idx, temps in enumerate([first_temps, second_temps]):
            cand_pairs = []
            for T12 in [temps[i:i+2] for i in range(0, len(temps), 2)]:
                if len(T12) == 1:
                    cand_pairs += [(c,) for c in dicts[idx][T12[0]]]
                    continue
                T1, T2 = T12
                for c1 in dicts[idx][T1]:
                    if keep_composition:
                        formula1 = c1.get_chemical_formula()
                        valid_partner_ids = [j for j, c in enumerate(dicts[idx][T2]) if
                                             c.get_chemical_formula() == formula1]
                        if valid_partner_ids:
                            i_pop = random.choice(valid_partner_ids)
                            c2 = dicts[idx][T2].pop(i_pop)
                            cand_pairs.append((c1, c2))
                            if idx == 0:
                                change_strat = False
                        else:
                            cand_pairs.append((c1,))
                    else:
                        len2 = len(dicts[idx][T2])
                        if len2 == 0:
                            cand_pairs.append((c1,))
                        else:
                            c2 = dicts[idx][T2].pop(random.randrange(0, len2))
                            cand_pairs.append((c1, c2))
                            if idx == 0:
                                change_strat = False
                cand_pairs += [(c,) for c in dicts[idx][T2]]
            if (idx == 0) and (not change_strat):
                break
        if prioritize_odd and (not change_strat):
            cand_pairs += [(c,) for c in dicts[0][keys[0]]]
        elif (not prioritize_odd) and change_strat:
            cand_pairs += [(c,) for c in dicts[1][keys[0]]]

        if keep_composition and all([len(pair) != 2 for pair in cand_pairs]):
            cand_pairs = self.get_exchange_pairs_within_composition(temp_cand_dict)

        return cand_pairs

    def _sorted_by_replica(self, candidates):

        return sorted(candidates, key=lambda c: c.info['key_value_pairs'][self.rk])

    def _linear_exchange_pairs(self, candidates, offset=0):

        candidates = self._sorted_by_replica(candidates)
        pairs = []
        if offset == 1 and candidates:
            pairs.append((candidates[0],))
        i = offset
        while i + 1 < len(candidates):
            pairs.append((candidates[i], candidates[i + 1]))
            i += 2
        if i < len(candidates):
            pairs.append((candidates[i],))
        return pairs

    def _indexed_exchange_pairs(self, candidates1, candidates2, shift=0):

        candidates1 = self._sorted_by_replica(candidates1)
        candidates2 = self._sorted_by_replica(candidates2)
        used1 = set()
        used2 = set()
        pairs = []
        for i, c1 in enumerate(candidates1):
            j = i + shift
            if 0 <= j < len(candidates2):
                pairs.append((c1, candidates2[j]))
                used1.add(i)
                used2.add(j)
        pairs += [(c,) for i, c in enumerate(candidates1) if i not in used1]
        pairs += [(c,) for j, c in enumerate(candidates2) if j not in used2]
        return pairs

    def _composition_exchange_pairs(self, candidates):

        options = self._sorted_by_replica(candidates)
        pairs = []
        while options:
            c1 = options.pop(0)
            formula1 = c1.get_chemical_formula()
            valid_partner_ids = [i for i, c in enumerate(options)
                                 if c.get_chemical_formula() == formula1]
            if valid_partner_ids:
                i2 = random.choice(valid_partner_ids)
                pairs.append((c1, options.pop(i2)))
            else:
                pairs.append((c1,))
        return pairs

    def _check_exchange_pairs(self, cand_pairs):

        expected = sorted(c.info['key_value_pairs'][self.rk] for c in self.pop)
        seen = []
        for pair in cand_pairs:
            if len(pair) not in (1, 2):
                raise RuntimeError('Exchange pair must contain one or two replicas')
            seen.extend(c.info['key_value_pairs'][self.rk] for c in pair)
        if sorted(seen) != expected:
            raise RuntimeError(f'Exchange pair coverage mismatch: expected {expected}, got {sorted(seen)}')

    def get_exchange_pairs_between_replicas(self, keep_composition=False):

        temp_cand_dict = defaultdict(list)
        for c in self.pop:
            temp_cand_dict[c.info['key_value_pairs'][self.tk]].append(c)
        keys = sorted(temp_cand_dict.keys())

        if keep_composition:
            cand_pairs = self._composition_exchange_pairs(self.pop)
            if all(len(pair) != 2 for pair in cand_pairs):
                cand_pairs = self.get_exchange_pairs_within_composition(temp_cand_dict)
            self._check_exchange_pairs(cand_pairs)
            return cand_pairs

        if len(keys) == 1:
            offset = random.randrange(2) if len(temp_cand_dict[keys[0]]) > 2 else 0
            cand_pairs = self._linear_exchange_pairs(temp_cand_dict[keys[0]], offset=offset)
            self._check_exchange_pairs(cand_pairs)
            return cand_pairs

        strat = random.choice(['within_temperature', 'same_condition', 'cross_left', 'cross_right'])
        cand_pairs = []
        if strat == 'within_temperature':
            for T in keys:
                offset = random.randrange(2) if len(temp_cand_dict[T]) > 2 else 0
                cand_pairs += self._linear_exchange_pairs(temp_cand_dict[T], offset=offset)
        else:
            offset = random.randrange(2) if len(keys) > 2 else 0
            key_pairs = []
            if offset == 1 and keys:
                key_pairs.append((keys[0],))
            i = offset
            while i + 1 < len(keys):
                key_pairs.append((keys[i], keys[i + 1]))
                i += 2
            if i < len(keys):
                key_pairs.append((keys[i],))
            for pair in key_pairs:
                if len(pair) == 1:
                    T = pair[0]
                    cand_pairs += self._linear_exchange_pairs(temp_cand_dict[T], offset=0)
                    continue
                T1, T2 = pair
                if strat == 'same_condition':
                    cand_pairs += self._indexed_exchange_pairs(temp_cand_dict[T1], temp_cand_dict[T2], shift=0)
                elif strat == 'cross_left':
                    cand_pairs += self._indexed_exchange_pairs(temp_cand_dict[T1], temp_cand_dict[T2], shift=-1)
                elif strat == 'cross_right':
                    cand_pairs += self._indexed_exchange_pairs(temp_cand_dict[T1], temp_cand_dict[T2], shift=1)

        self._check_exchange_pairs(cand_pairs)
        return cand_pairs

    def get_exchange_pairs_within_composition(self, temp_cand_dict=None):

        if temp_cand_dict is None:
            temp_cand_dict = defaultdict(list)
            for c in self.pop:
                temp_cand_dict[c.info['key_value_pairs'][self.tk]].append(c)

        keys = sorted(temp_cand_dict.keys())
        cand_pairs = []
        for T1 in keys:
            for c1 in temp_cand_dict[T1]:
                found_pair = False
                formula1 = c1.get_chemical_formula()
                for T2 in keys:
                    if T2 <= T1:
                        continue
                    valid_partner_ids = [j for j, c in enumerate(temp_cand_dict[T2]) if
                                         c.get_chemical_formula() == formula1]
                    if valid_partner_ids:
                        i_pop = random.choice(valid_partner_ids)
                        c2 = temp_cand_dict[T2].pop(i_pop)
                        cand_pairs.append((c1, c2))
                        found_pair = True
                        break
                if not found_pair:
                    cand_pairs.append((c1,))

        return cand_pairs

    def exchange(self, keep_composition=False, cand_pairs=None,
                 exchange_within_temperature=False, 
                 prefactor=1, element_column_map=None, 
                 temperature_species_internal_map=None,
                 no_internal_indices=None, state_fitness=None, pressure=None):

        if cand_pairs is None:
            if exchange_within_temperature:
                cand_pairs = self.get_exchange_pairs_between_replicas(keep_composition)
            else:
                cand_pairs = self.get_exchange_pairs_between_temperatures(keep_composition)

        new_pop = []
        for c12 in cand_pairs:
            if len(c12) == 1:
                c = c12[0]
                if self.verbose:
                    rep = c.info['key_value_pairs'][self.rk]
                    parprint(f'Replica {rep} exchange skipped: no valid partner in this exchange sweep.')
                new_pop += list(c12)
                continue
            c1, c2 = c12
            rep1 = c1.info['key_value_pairs'][self.rk]
            rep2 = c2.info['key_value_pairs'][self.rk]
            f1 = get_raw_score(c1)
            f2 = get_raw_score(c2)
            T1 = c1.info['key_value_pairs'][self.tk]
            T2 = c2.info['key_value_pairs'][self.tk]
            if state_fitness is not None:
                mu1 = c1.info['data'].get('mu')
                mu2 = c2.info['data'].get('mu')
                f1 = state_fitness(c1, T1, mu1)
                f2 = state_fitness(c2, T2, mu2)
                f12 = state_fitness(c1, T2, mu2)
                f21 = state_fitness(c2, T1, mu1)
                c1.info['key_value_pairs']['raw_score'] = f1
                c2.info['key_value_pairs']['raw_score'] = f2
            if prefactor == 'gc':
                ln_prefactor = compute_exchange_prefactor(
                        c1, c2, element_column_map, 
                        temperature_species_internal_map,
                        no_internal_indices,
                        temperature_key=self.tk,
                        exponent=0. if state_fitness is not None else None,
                        return_log=True)
            else:
                ln_prefactor = log_prefactor_value(prefactor)
            df = f1 - f2
            if state_fitness is not None:
                ln_p_acc = ln_prefactor + (f21-f1)/(kB*T1) + (f12-f2)/(kB*T2)
            else:
                # Standalone use with a common metal reservoir.  GCEMC passes
                # state_fitness to include all temperature-dependent terms.
                shift1 = shift2 = 0.
                if prefactor == 'gc' and element_column_map:
                    for elem, col in element_column_map.items():
                        dmu = c2.info['data']['mu'][col] - c1.info['data']['mu'][col]
                        shift1 += sum(a.symbol == elem for a in c1) * dmu
                        shift2 -= sum(a.symbol == elem for a in c2) * dmu
                f12, f21 = f1 + shift1, f2 + shift2
                if pressure is not None:
                    df -= pressure * (c1.info['key_value_pairs']['volume'] -
                                      c2.info['key_value_pairs']['volume'])
                ln_p_acc = ln_prefactor + (1/(kB*T2)-1/(kB*T1))*df
            accept = metropolis_accept_log(ln_p_acc)

            if self.verbose:
                parprint(f"Replicas: {[rep1, rep2]}, " + 
                         f"Formulas: {[c1.get_chemical_formula(), c2.get_chemical_formula()]}, " +
                         f"operation: Exchange, delta: {df}, log prefactor: {ln_prefactor}.")
            if accept:
                c1.info['key_value_pairs'][self.rk], c2.info['key_value_pairs'][self.rk] = \
                        c2.info['key_value_pairs'][self.rk], c1.info['key_value_pairs'][self.rk]
                c1.info['key_value_pairs'][self.tk], c2.info['key_value_pairs'][self.tk] = \
                        c2.info['key_value_pairs'][self.tk], c1.info['key_value_pairs'][self.tk]
                if self.ck is not None:
                    c1.info['key_value_pairs'][self.ck], c2.info['key_value_pairs'][self.ck] = \
                        c2.info['key_value_pairs'][self.ck], c1.info['key_value_pairs'][self.ck]
                if 'mu' in c1.info['data']:
                    c1.info['data']['mu'], c2.info['data']['mu'] = \
                            c2.info['data']['mu'], c1.info['data']['mu']
                if 'mu_window' in c1.info['data']:
                    c1.info['data']['mu_window'], c2.info['data']['mu_window'] = \
                            c2.info['data']['mu_window'], c1.info['data']['mu_window']
                c1.info['key_value_pairs']['raw_score'] = f12
                c2.info['key_value_pairs']['raw_score'] = f21
                if self.verbose:
                    parprint(f'Replicas {rep1} and {rep2} exchange move accepted!')
            else:
                if self.verbose:
                    parprint(f'Replicas {rep1} and {rep2} exchange move rejected!')
            new_pop += [c1, c2]

        new_pop.sort(key=lambda x: x.info['key_value_pairs'][self.rk])
        self.pop = new_pop
        self.current_fitness = self.__get_fitness__(self.pop)
        self._checkpoint_population()

    def mu_walk(self, element_column_map, multiple_try=True,
                reaction_conditions=None, free_energy_differences=None, tol=5e-3,
                state_fitness=None):

        new_pop = []
        for c in self.pop:
            rep = c.info['key_value_pairs'][self.rk]
            T = c.info['key_value_pairs'][self.tk]
            mu = c.info['data']['mu']
            mu_window = c.info['data']['mu_window']
            mask = np.all(mu_window==mu, axis=1)
            if np.count_nonzero(mask) != 1:
                raise ValueError('The current chemical potential must occur once in its window.')
            valid_rows = np.where(~mask)[0]
            if not len(valid_rows):
                new_pop.append(c)
                continue
            old_fitness = (get_raw_score(c) if state_fitness is None else
                           state_fitness(c, T, mu))
            c.info['key_value_pairs']['raw_score'] = old_fitness

            def trial_fitness(trial_mu):
                if state_fitness is not None:
                    return state_fitness(c, T, trial_mu)
                return old_fitness + sum(sum(a.symbol == k for a in c) *
                                        (trial_mu[col] - mu[col])
                                        for k, col in element_column_map.items())

            if multiple_try:
                # multiple-try Metropolis
                # compute exponent/log-weight for every candidate in valid_rows
                exps = []
                for idx in valid_rows:
                    trial_mu_c = mu_window[idx]
                    exponent_c = (trial_fitness(trial_mu_c) - old_fitness) / (kB * T)
                    if free_energy_differences is not None:
                        cond = np.append(mu, T)
                        grid_id = np.where(
                                np.all(np.abs(reaction_conditions - cond) < tol, axis=1))[0]
                        trial_cond = np.append(trial_mu_c, T)
                        trial_grid_id = np.where(
                                np.all(np.abs(reaction_conditions - trial_cond) < tol, axis=1))[0]
                        df = free_energy_differences[grid_id, trial_grid_id]
                        if not np.isscalar(df):
                            df = df[0]
                        exponent_c += df
                    # exponent_c is ln(w_j / w_current)
                    exps.append(exponent_c)

                # numerically-stable selection: convert to probs ~ exp(exps)
                exps = np.asarray(exps, dtype=float)
                M = np.max(exps)
                weights = np.exp(exps - M)
                probs = weights / np.sum(weights)
                # pick one candidate (b) from forward set (weighted)
                chosen_pos = np.random.choice(len(valid_rows), p=probs)
                trial_idx = int(valid_rows[chosen_pos])
                trial_mu = mu_window[trial_idx]
                # MTM reverse/exponent construction
                exps_chosen = exps[chosen_pos]
                # forward log-sum = log(sum_j w_j) baseline canceled later
                log_sum_forward = logsumexp(exps)
                # reverse_exps = ln(w_u / w_selected) for u in reverse set
                reverse_exps = np.delete(exps, chosen_pos) - exps_chosen
                # include current state's term ln(w_current/w_selected) = -exps_chosen
                reverse_exps = np.append(reverse_exps, -exps_chosen)
                log_sum_reverse = logsumexp(reverse_exps)
                # final lnR as derived in the note
                lnR = log_sum_forward - log_sum_reverse - exps_chosen
                accept = metropolis_accept_log(lnR)
            else:
                # single-try Metropolis
                trial_mu = mu_window[np.random.choice(valid_rows)]
                exponent = (trial_fitness(trial_mu) - old_fitness) / (kB * T)
                if free_energy_differences is not None:
                    cond = np.append(mu, T)
                    grid_id = np.where(
                            np.all(np.abs(reaction_conditions - cond) < tol, axis=1))[0]
                    trial_cond = np.append(trial_mu, T)
                    trial_grid_id = np.where(
                            np.all(np.abs(reaction_conditions - trial_cond) < tol, axis=1))[0]
                    df = free_energy_differences[grid_id, trial_grid_id]
                    if not np.isscalar(df):
                        df = df[0]
                    exponent += df
                accept = metropolis_accept_log(exponent)

            if accept:
                c.info['key_value_pairs']['raw_score'] = trial_fitness(trial_mu)
                c.info['data']['mu'] = trial_mu
                if self.verbose:
                    parprint(f'Replica {rep} mu-walk move accepted!')
            else:
                if self.verbose:
                    parprint(f'Replica {rep} mu-walk move rejected!')
            new_pop.append(c)

        self.pop = new_pop
        self.current_fitness = self.__get_fitness__(self.pop)
        self._checkpoint_population()

    def get_all_parents(self, n_crossover_pairs=1, n_interswap_pairs=0,
                        crossover_by_boltzmann_first=False,        
                        interswap_within_temperature=False,
                        interswap_within_neighbors=False,
                        interswap_within_key=None):
        """Returns all paired candidates for pairing employing the
        roulete wheel selection scheme described in R.L. Johnston 
        Dalton Transactions, Vol. 22, No. 22. (2003), pp. 4193-4207"""

        if len(self.pop) < 2:
            self.update()

        if len(self.pop) < 2:
            return None

        T = np.array([c.info['key_value_pairs'][self.tk] for c in self.pop])
        # optional grouping key array
        if interswap_within_key is not None:
            K = np.array([c.info['key_value_pairs'][interswap_within_key] for c in self.pop])
        else:
            K = None
        indices = np.arange(len(self.pop))
        all_parents = []
        if interswap_within_neighbors:
            # neighboring pairs (0,1), (2,3), ... in the current replica order 
            used = []
            pairs_added = 0
            for pos in range(0, len(indices) - 1, 2):
                i_idx = indices[pos]
                j_idx = indices[pos + 1]
                if interswap_within_temperature and (T[i_idx] != T[j_idx]):
                    continue
                if (interswap_within_key is not None) and (K[i_idx] != K[j_idx]):
                    continue
                c1 = self.pop[i_idx]
                c2 = self.pop[j_idx]
                c1.info['data']['for_operation'] = 'interswap'
                c2.info['data']['for_operation'] = 'interswap'
                all_parents.append((c1, c2))
                used.extend([i_idx, j_idx])
                pairs_added += 1
                if pairs_added >= n_interswap_pairs:
                    break
            if used:
                indices = np.setdiff1d(indices, np.array(used, dtype=int))
        else:
            # random-selection interswap
            for _ in range(n_interswap_pairs):
                if len(indices) < 2:
                    break
                i = np.random.choice(np.arange(len(indices)))
                c1 = self.pop[indices[i]]
                tmp_indices = np.delete(indices, i)
                if interswap_within_temperature or (interswap_within_key is not None):
                    if interswap_within_temperature:
                        T1 = c1.info['key_value_pairs'][self.tk] 
                    else:
                        T1 = None
                    if interswap_within_key is not None:
                        K1 = c1.info['key_value_pairs'][interswap_within_key]
                    else:
                        K1 = None
                    options = []
                    for k, idx in enumerate(tmp_indices):
                        ok = True
                        if interswap_within_temperature:
                            if self.pop[idx].info['key_value_pairs'][self.tk] != T1:
                                ok = False
                        if interswap_within_key is not None:
                            if self.pop[idx].info['key_value_pairs'][interswap_within_key] != K1:
                                ok = False
                        if ok:
                            options.append(k)
                    if not options:
                        indices = tmp_indices
                        continue
                    indices = tmp_indices
                    j = np.random.choice(options)
                else:
                    indices = tmp_indices
                    j = np.random.choice(np.arange(len(indices)))
                c2 = self.pop[indices[j]]
                indices = np.delete(indices, j)
                c1.info['data']['for_operation'] = 'interswap'
                c2.info['data']['for_operation'] = 'interswap'
                all_parents.append((c1, c2))
 
        if crossover_by_boltzmann_first:
            weights = np.array(self.current_fitness)[indices] / (kB*T[indices]) 
            weights /= np.sum(weights)
            for _ in range(n_crossover_pairs):
                if len(indices) < 2:
                    break
                i = np.random.choice(np.arange(len(indices)), p=weights)
                c1 = self.pop[indices[i]]
                c1.info['data']['for_operation'] = 'crossover'
                indices = np.delete(indices, i)
                weights = np.delete(weights, i)
                weights /= np.sum(weights)
                j = np.random.choice(np.arange(len(indices)))
                c2 = self.pop[indices[j]]
                c2.info['data']['for_operation'] = 'crossover'
                indices = np.delete(indices, j)
                weights = np.delete(weights, j)
                weights /= np.sum(weights)
                all_parents.append((c1, c2))
        else:
            for _ in range(n_crossover_pairs):
                if len(indices) < 2:
                    break
                i = np.random.choice(np.arange(len(indices)))
                c1 = self.pop[indices[i]]
                c1.info['data']['for_operation'] = 'crossover'
                indices = np.delete(indices, i)
                j = np.random.choice(np.arange(len(indices)))
                c2 = self.pop[indices[j]]
                c2.info['data']['for_operation'] = 'crossover'
                indices = np.delete(indices, j)
                all_parents.append((c1, c2))

        other_parents = []
        for idx in indices:
            c = self.pop[idx]
            c.info['data']['for_operation'] = 'intramove'
            other_parents.append((c,))
        all_parents += other_parents
        all_parents.sort(key=lambda x: x[0].info['key_value_pairs'][self.rk])

        return all_parents
