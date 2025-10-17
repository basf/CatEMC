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
from ase.units import kB
from scipy.special import logsumexp
import numpy as np
import warnings
import random


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
                 acceptance_key='accepted', use_extinct=False, 
                 store_delta=True, verbose=True, rng=np.random):
        # The current fitness is set at each update of the population
        self.current_fitness = None
        self.comparator = None
        self.tk = temperature_key
        self.rk = replica_key
        self.ak = acceptance_key
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
        self.current_fitness = self.__get_fitness__(self.pop)
        self.all_cand = all_cand

    def update(self):
        """ New candidates can be added to the database
            after the population object has been created.
            This method extracts these new candidates from the
            database and includes them in the population. """

        self.pop = []
        self.__initialize_pop__()

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
            switch_to_crossover = False
            if parents[0].info['data']['for_operation'] == 'crossover':
                sorted_parents = sorted(parents, key=get_raw_score, reverse=True)
                sorted_oids, sorted_offsprings = zip(*sorted(enumerate(offsprings),
                        key=lambda t: get_raw_score(t[1]), reverse=True))
            else:
                if mutable_indices is not None:
                    ctp = Counter([a.symbol for a in parents[0] if a.index not in mutable_indices])
                    cto = Counter([a.symbol for a in offsprings[0] if a.index not in mutable_indices])
                    if ctp != cto:
                        switch_to_crossover = True
                if switch_to_crossover:
                    warnings.warn(f"Interswap not possible for [p.info['key_value_pairs'][self.rk] for p in parents] " +
                                  f"due to non-mutable atom counts changed. Switch to crossover instead!")
                    sorted_parents = sorted(parents, key=get_raw_score, reverse=True)
                    sorted_oids, sorted_offsprings = zip(*sorted(enumerate(offsprings),
                            key=lambda t: get_raw_score(t[1]), reverse=True))
                else:
                    sorted_parents = parents
                    sorted_oids = [0, 1]
                    sorted_offsprings = offsprings
            rep1 = sorted_parents[0].info['key_value_pairs'][self.rk]
            rep2 = sorted_parents[1].info['key_value_pairs'][self.rk]
            df1 = get_raw_score(sorted_offsprings[0]) - get_raw_score(sorted_parents[0])
            df2 = get_raw_score(sorted_offsprings[1]) - get_raw_score(sorted_parents[1])
            T1 = sorted_parents[0].info['key_value_pairs'][self.tk]
            T2 = sorted_parents[1].info['key_value_pairs'][self.tk]
            offsprings[sorted_oids[0]].info['key_value_pairs'][self.rk] = rep1
            offsprings[sorted_oids[1]].info['key_value_pairs'][self.rk] = rep2
            offsprings[sorted_oids[0]].info['key_value_pairs'][self.tk] = T1
            offsprings[sorted_oids[1]].info['key_value_pairs'][self.tk] = T2
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

                if pressure is not None:
                    V11 = sorted_parents[0].get_volume()
                    V12 = sorted_offsprings[0].get_volume()
                    V21 = sorted_parents[1].get_volume()
                    V22 = sorted_offsprings[1].get_volume()
                    N11 = len([a for a in sorted_parents[0] if a.symbol != 'X'])
                    N12 = len([a for a in sorted_offsprings[0] if a.symbol != 'X'])
                    N21 = len([a for a in sorted_parents[1] if a.symbol != 'X'])
                    N22 = len([a for a in sorted_offsprings[1] if a.symbol != 'X'])
                    lnV = N12*np.log(V12) - N11*np.log(V11) + N22*np.log(V22) - N21*np.log(V21)
                    df1 += pressure * (V12 - V11)
                    df2 += pressure * (V22 - V21)
                    lnC = np.log(prefactor) + lnV
                    prefactor = np.exp(lnC)
                p_acc = min([1, prefactor * np.exp(df1 / (kB*T1) + df2/ (kB*T2))])
                accept = p_acc > np.random.uniform()
            if self.verbose:
                parprint(f"Replicas: {[rep1, rep2]}, " + 
                         f"parents: {[sorted_parents[0].get_chemical_formula(), sorted_parents[1].get_chemical_formula()]}, " +
                         f"offsprings: {[sorted_offsprings[0].get_chemical_formula(), sorted_offsprings[1].get_chemical_formula()]}, " +
                         f"operation: {offsprings[0].info['key_value_pairs']['origin']}, " + 
                         f"deltas: {df1} and {df2}, prefactor: {prefactor}.")
            if accept:
                offsprings[0].info['key_value_pairs'][self.ak] = 1
                offsprings[1].info['key_value_pairs'][self.ak] = 1
                parprint(f'Replicas {rep1} {rep2} inter move accepted!')
            else:
                offsprings[0].info['key_value_pairs'][self.ak] = 0
                offsprings[1].info['key_value_pairs'][self.ak] = 0
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
            rep = parents[0].info['key_value_pairs'][self.rk]
            df = get_raw_score(offsprings[0]) - get_raw_score(parents[0])
            T = parents[0].info['key_value_pairs'][self.tk]
            offsprings[0].info['key_value_pairs'][self.rk] = rep
            offsprings[0].info['key_value_pairs'][self.tk] = T
            if any(o for o in offsprings if get_raw_score(o) <= raw_score_cutoff): 
                prefactor = None
                accept = False
            else:
                if prefactor == 'gc':
                    prefactor = compute_intramove_prefactor(
                            parents[0], offsprings[0], 
                            volume, nonmetal_species, 
                            temperature_species_internal_map,
                            no_internal_indices,
                            metal_species_probabilities,
                            temperature_key=self.tk)
                if pressure is not None:
                    V1 = parents[0].get_volume()
                    V2 = offsprings[0].get_volume()
                    N1 = len([a for a in parents[0] if a.symbol != 'X'])
                    N2 = len([a for a in offsprings[0] if a.symbol != 'X'])
                    lnV =  N2*np.log(V2) - N1*np.log(V1)
                    df += pressure * (V2 - V1)
                    lnC = np.log(prefactor) + lnV
                    prefactor = np.exp(lnC)
                p_acc = min([1, prefactor * np.exp(df / (kB*T))])
                accept = p_acc > np.random.uniform()
            if self.verbose:
                parprint(f"Replica: {rep}, parent: {parents[0].get_chemical_formula()}, " +
                         f"offspring: {offsprings[0].get_chemical_formula()}, " + 
                         f"operation: {offsprings[0].info['key_value_pairs']['origin']}, delta: {df}, " +
                         f"prefactor: {prefactor}.")
            if accept:
                offsprings[0].info['key_value_pairs'][self.ak] = 1
                parprint(f'Replica {rep} intra move accepted!')
            else:
                offsprings[0].info['key_value_pairs'][self.ak] = 0
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

    def get_exchange_pairs_between_replicas(self, keep_composition=False):

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
                    temp_cands = dicts[idx][T12[0]]
                    cand_pairs += [tuple(temp_cands[i:i+2]) for i in 
                                   range(0, len(temp_cands), 2)]
                    if idx == 0:
                        change_strat = False
                    continue
                T1, T2 = T12
                if keep_composition:                                                                          
                    options = dicts[idx][T1] + dicts[idx][T2]
                    while options:
                        i1 = random.randrange(0, len(options))
                        c1 = options.pop(i1)
                        formula1 = c1.get_chemical_formula()
                        valid_partner_ids = [j for j, c in enumerate(options) if
                                             c.get_chemical_formula() == formula1]
                        if valid_partner_ids:
                            i2 = random.choice(valid_partner_ids)
                            c2 = options.pop(i2)
                            cand_pairs.append((c1, c2))
                            if idx == 0:
                                change_strat = False
                        else:
                            cand_pairs.append((c1,))
                else:
                    strat = random.choice(['vertical','horizontal','cross_left','cross_right'])
                    lst2, lst1 = sorted([dicts[idx][T1], dicts[idx][T2]], key=lambda x: len(x))
                    if strat == 'vertical':
                        for i1, c1 in enumerate(lst1):
                            try:
                                c2 = lst2[i1]
                                cand_pairs.append((c1, c2))
                            except:
                                cand_pairs.append((c1,))
                    elif strat == 'horizontal':
                        for T in [T1, T2]:
                            temp_cands = dicts[idx][T]
                            cand_pairs += [tuple(temp_cands[i:i+2]) for i in
                                           range(0, len(temp_cands), 2)]
                    elif strat == 'cross_left':
                        for i1, c1 in enumerate(lst1):
                            if i1 == 0:
                                cand_pairs.append((c1,))
                                if len(lst2) == len(lst1):
                                    c2 = lst2[-1]
                                    cand_pairs.append((c2,))
                            else:
                                try:
                                    c2 = lst2[i1-1]
                                    cand_pairs.append((c1, c2))
                                except:
                                    cand_pairs.append((c1,))
                    elif strat == 'cross_right':
                        for i1, c1 in enumerate(lst1):
                            if i1 == len(lst1) - 1:
                                cand_pairs.append((c1,))
                                c2 = lst2[0]
                                cand_pairs.append((c2,))
                            else:
                                try:
                                    c2 = lst2[i1+1]
                                    cand_pairs.append((c1, c2))
                                except:
                                    cand_pairs.append((c1,))
                    if idx == 0:
                        change_strat = False
            if (idx == 0) and (not change_strat):
                break
        if prioritize_odd and (not change_strat):
            cand_pairs += [(c,) for c in dicts[0][keys[0]]]
        elif (not prioritize_odd) and change_strat:
            cand_pairs += [(c,) for c in dicts[1][keys[0]]]

        if keep_composition and all([len(pair) != 2 for pair in cand_pairs]):
            cand_pairs = self.get_exchange_pairs_within_composition(temp_cand_dict)

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
                 exchange_within_temperature=False, prefactor=1,
                 reaction_conditions=None, element_column_map=None,
                 temperature_species_internal_map=None,
                 no_internal_indices=None):

        if cand_pairs is None:
            if exchange_within_temperature:
                cand_pairs = self.get_exchange_pairs_between_replicas(keep_composition)
            else:
                cand_pairs = self.get_exchange_pairs_between_temperatures(keep_composition)

        new_pop = []
        for c12 in cand_pairs:
            if len(c12) == 1:
                new_pop += list(c12)
                continue
            c1, c2 = c12
            rep1 = c1.info['key_value_pairs'][self.rk]
            rep2 = c2.info['key_value_pairs'][self.rk]
            f1 = get_raw_score(c1)
            f2 = get_raw_score(c2)
            T1 = c1.info['key_value_pairs'][self.tk]
            T2 = c2.info['key_value_pairs'][self.tk]
            if prefactor == 'gc':
                c12_prefactor = compute_exchange_prefactor(
                        c1, c2, reaction_conditions, 
                        element_column_map, 
                        temperature_species_internal_map,
                        no_internal_indices,
                        temperature_key=self.tk)
            else:
                c12_prefactor = prefactor
            df = f1 - f2
            p_acc = min([1, c12_prefactor * np.exp((1/(kB*T2)-1/(kB*T1))*df)])
            accept = p_acc > np.random.uniform()

            if self.verbose:
                parprint(f"Replicas: {[rep1, rep2]}, " + 
                         f"Formulas: {[c1.get_chemical_formula(), c2.get_chemical_formula()]}, " +
                         f"operation: exchange, delta: {df}, prefactor: {c12_prefactor}.")
            if accept:
                c1.info['key_value_pairs'][self.rk], c2.info['key_value_pairs'][self.rk] = \
                        c2.info['key_value_pairs'][self.rk], c1.info['key_value_pairs'][self.rk]
                c1.info['key_value_pairs'][self.tk], c2.info['key_value_pairs'][self.tk] = \
                        c2.info['key_value_pairs'][self.tk], c1.info['key_value_pairs'][self.tk]
                if 'mu' in c1.info['data']:
                    c1.info['data']['mu'], c2.info['data']['mu'] = \
                            c2.info['data']['mu'], c1.info['data']['mu']
                if 'mu_window' in c1.info['data']:
                    c1.info['data']['mu_window'], c2.info['data']['mu_window'] = \
                            c2.info['data']['mu_window'], c1.info['data']['mu_window']
                parprint(f'Replicas {rep1} and {rep2} exchange move accepted!')
            else:
                parprint(f'Replicas {rep1} and {rep2} exchange move rejected!')
            new_pop += [c1, c2]

        new_pop.sort(key=lambda x: x.info['key_value_pairs'][self.rk])
        self.pop = new_pop

    def mu_walk(self, element_column_map, multiple_try=True,
                reaction_conditions=None, free_energy_differences=None, tol=1e-3):

        new_pop = []
        for c in self.pop:
            rep = c.info['key_value_pairs'][self.rk]
            T = c.info['key_value_pairs'][self.tk]
            mu = c.info['data']['mu']
            mu_window = c.info['data']['mu_window']
            mask = np.all(mu_window==mu, axis=1)
            valid_rows = np.where(~mask)[0]

            if multiple_try:
                # multiple-try Metropolis
                # compute exponent/log-weight for every candidate in valid_rows
                exps = []
                for idx in valid_rows:
                    trial_mu_c = mu_window[idx]
                    exponent_c = 0.0
                    for k, col in element_column_map.items():
                        nk = len([a for a in c if a.symbol == k])
                        exponent_c += nk * (trial_mu_c[col] - mu[col])
                    exponent_c /= (kB * T)
                    if free_energy_differences is not None:
                        cond = np.append(mu, T)
                        grid_id = np.where(
                                np.all(np.abs(reaction_conditions - cond) < tol, axis=1))[0]
                        trial_cond = np.append(trial_mu_c, T)
                        trial_grid_id = np.where(
                                np.all(np.abs(reaction_conditions - trial_cond) < tol, axis=1))[0]
                        df = free_energy_differences[grid_id, trial_grid_id]
                        exponent_c += df
                    # exponent_c is ln(w_j / w_current)
                    exps.append(float(exponent_c))

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
                if lnR >= 0:
                    accept = True
                else:
                    accept = (np.random.uniform() < np.exp(lnR))
            else:
                # single-try Metropolis
                trial_mu = mu_window[np.random.choice(valid_rows)]
                exponent = 0.0
                for k, col in element_column_map.items():
                    nk = len([a for a in c if a.symbol == k])
                    exponent += nk * (trial_mu[col] - mu[col])
                exponent /= (kB * T)
                if free_energy_differences is not None:
                    cond = np.append(mu, T)
                    grid_id = np.where(
                            np.all(np.abs(reaction_conditions - cond) < tol, axis=1))[0]
                    trial_cond = np.append(trial_mu, T)
                    trial_grid_id = np.where(
                            np.all(np.abs(reaction_conditions - trial_cond) < tol, axis=1))[0]
                    df = free_energy_differences[grid_id, trial_grid_id]
                    exponent += df
                p_acc = min([1, np.exp(exponent)])
                accept = (p_acc > np.random.uniform())

            if accept:
                c.info['data']['mu'] = trial_mu
                parprint(f'Replica {rep} mu-walk move accepted!')
            else:
                parprint(f'Replica {rep} mu-walk move rejected!')
            new_pop.append(c)

        self.pop = new_pop

    def get_all_parents(self, n_crossover_pairs=1, n_interswap_pairs=0,
                        crossover_by_boltzmann_first=False,        
                        interswap_within_temperature=False):
        """Returns all paired candidates for pairing employing the
        roulete wheel selection scheme described in
        R.L. Johnston Dalton Transactions,
        Vol. 22, No. 22. (2003), pp. 4193-4207
        """

        if len(self.pop) < 2:
            self.update()

        if len(self.pop) < 2:
            return None

        T = np.array([c.info['key_value_pairs'][self.tk] for c in self.pop])
        indices = np.arange(len(self.pop))
        all_parents = []
        for _  in range(n_interswap_pairs):
            i = np.random.choice(np.arange(len(indices)))
            c1 = self.pop[indices[i]]
            tmp_indices = np.delete(indices, i)
            if interswap_within_temperature:
                T1 = c1.info['key_value_pairs'][self.tk]
                options = [k for k, idx in enumerate(tmp_indices) if 
                           self.pop[idx].info['key_value_pairs'][self.tk] == T1]
                if not options:
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


class MixtureEMCPopulation(EMCPopulation):

    """The mixture EMC population instance that evolves based on
    Metropolis-Hastings algorithm as a whole.

    Parameters
    ----------
    population_size: int
        Size of the EMC population.

    num_tasks: int
        The number of tasks.

    """

    def __init__(self, data_connection, population_size, num_tasks, 
                 logfile=None, temperature_key='temperature', 
                 replica_key='replica', acceptance_key='accepted', 
                 use_extinct=False, store_delta=False, verbose=False, 
                 rng=np.random):
        # The current fitness is set at each update of the population
        self.current_fitness = None
        self.num_tasks = num_tasks

        EMCPopulation.__init__(self, data_connection, population_size, logfile, 
                               temperature_key, replica_key, acceptance_key, 
                               use_extinct, store_delta, verbose=verbose, rng=rng)

    def evaluate_offsprings(self, parents, offsprings, prefactor=1, volume=None,
                            nonmetal_species=None, temperature_species_internal_map=None,
                            no_internal_indices=None, metal_species_probabilities=None, 
                            mutable_indices=None, raw_score_cutoff=-10000.):

        if len(parents) == 2:
            # Crossover or interswap
            assert len(offsprings) == 2
            switch_to_crossover = False
            if mutable_indices is not None:
                ctp = Counter([a.symbol for a in parents[0] if a.index not in mutable_indices])
                cto = Counter([a.symbol for a in offsprings[0] if a.index not in mutable_indices])
                if ctp != cto:
                    switch_to_crossover = True

            T1 = parents[0].info['key_value_pairs'][self.tk]
            T2 = parents[1].info['key_value_pairs'][self.tk]
            rep1 = parents[0].info['key_value_pairs'][self.rk]
            rep2 = parents[1].info['key_value_pairs'][self.rk]
            pscores1 = parents[0].info['data']['raw_scores']
            pscores2 = parents[1].info['data']['raw_scores']
            oscores1 = offsprings[0].info['data']['raw_scores']
            oscores2 = offsprings[1].info['data']['raw_scores']
            # 4 cases: (P1,O1)/(P2,O2)/(P1,O2)/(P2,O1) being the fitter pair
            # 2 assignments: a: (P1,O1) + (P2,O2) / b: (P1,O2) + (P2,O1)
            if (parents[0].info['data']['for_operation'] == 'crossover') or switch_to_crossover:
                lnSp1 = logsumexp(pscores1 / (kB*T1)) # np.sum(np.exp(pscores1 / (kB*T1)))
                lnSp2 = logsumexp(pscores2 / (kB*T2)) # np.sum(np.exp(pscores2 / (kB*T2)))
                lnSo1a = logsumexp(oscores1 / (kB*T1)) # np.sum(np.exp(oscores1 / (kB*T1)))
                lnSo2a = logsumexp(oscores2 / (kB*T2)) # np.sum(np.exp(oscores2 / (kB*T2)))
                lnSo1b = logsumexp(oscores1 / (kB*T2)) # np.sum(np.exp(oscores1 / (kB*T2)))
                lnSo2b = logsumexp(oscores2 / (kB*T1)) # np.sum(np.exp(oscores2 / (kB*T1)))
                if lnSo1a + lnSo2a >= lnSo1b + lnSo2b:
                    lnSo1 = lnSo1b
                    lnSo2 = lnSo2b
                    assignment = 'a'
                else:
                    lnSo1 = lnSo1a
                    lnSo2 = lnSo2a
                    assignment = 'b'
            elif parents[0].info['data']['for_operation'] == 'interswap':
                lnSp1 = logsumexp(pscores1 / (kB*T1)) # np.sum(np.exp(pscores1 / kB*T1))
                lnSp2 = logsumexp(pscores2 / (kB*T2)) # np.sum(np.exp(pscores2 / kB*T2))
                lnSo1 = logsumexp(oscores1 / (kB*T1)) # np.sum(np.exp(oscores1 / kB*T1))
                lnSo2 = logsumexp(oscores2 / (kB*T2)) # np.sum(np.exp(oscores2 / kB*T2))
                assignment = 'a'

            if assignment == 'a':
                offsprings[0].info['key_value_pairs'][self.rk] = parents[0].info['key_value_pairs'][self.rk] 
                offsprings[0].info['key_value_pairs'][self.tk] = parents[0].info['key_value_pairs'][self.tk] 
                offsprings[1].info['key_value_pairs'][self.rk] = parents[1].info['key_value_pairs'][self.rk] 
                offsprings[1].info['key_value_pairs'][self.tk] = parents[1].info['key_value_pairs'][self.tk]
                if self.store_delta:
                    offsprings[0].info['data']['delta'] = oscores1 - pscores1
                    offsprings[1].info['data']['delta'] = oscores2 - pscores2
                if prefactor == 'gc':
                    if (parents[0].info['data']['for_operation'] == 'crossover') or switch_to_crossover:
                        prefactor = compute_crossover_prefactor(
                                parents[0], parents[1], offsprings[0], offsprings[1],
                                nonmetal_species, temperature_species_internal_map,
                                no_internal_indices, temperature_key=self.tk)
                    elif parents[0].info['data']['for_operation'] == 'interswap':
                        prefactor = compute_interswap_prefactor(
                                parents[0], parents[1], offsprings[0], offsprings[1])
            elif assignment == 'b':
                offsprings[0].info['key_value_pairs'][self.rk] = parents[1].info['key_value_pairs'][self.rk] 
                offsprings[0].info['key_value_pairs'][self.tk] = parents[1].info['key_value_pairs'][self.tk] 
                offsprings[1].info['key_value_pairs'][self.rk] = parents[0].info['key_value_pairs'][self.rk] 
                offsprings[1].info['key_value_pairs'][self.tk] = parents[0].info['key_value_pairs'][self.tk]
                if self.store_delta:
                    offsprings[0].info['data']['delta'] = oscores1 - pscores2
                    offsprings[1].info['data']['delta'] = oscores2 - pscores1
                if prefactor == 'gc':
                    prefactor = compute_crossover_prefactor(
                            parents[0], parents[1], offsprings[1], offsprings[0],
                            nonmetal_species, temperature_species_internal_map,
                            no_internal_indices, temperature_key=self.tk)

            if any(o for o in offsprings if get_raw_score(o) <= raw_score_cutoff):
                prefactor = None
                accept = False
            else:    
                p_acc = min([1, prefactor * np.exp(lnSo1 - lnSp1 + lnSo2 - lnSp2)])
                accept = p_acc > np.random.uniform()

            if self.verbose:
                if assignment == 'a':
                    parprint(f"Replicas: {[rep1, rep2]}, " + 
                             f"parents: {[parents[0].get_chemical_formula(), parents[1].get_chemical_formula()]}, " + 
                             f"offsprings: {[offsprings[0].get_chemical_formula(), offsprings[1].get_chemical_formula()]}, " +
                             f"operation: {offsprings[0].info['key_value_pairs']['origin']}, " + 
                             f"prefactor: {prefactor}.")
                elif assignment == 'b':
                    parprint(f"Replicas: {[rep1, rep2]}, " + 
                             f"parents: {[parents[0].get_chemical_formula(), parents[1].get_chemical_formula()]}, " + 
                             f"offsprings: {[offsprings[1].get_chemical_formula(), offsprings[0].get_chemical_formula()]}, " +
                             f"operation: {offsprings[0].info['key_value_pairs']['origin']}, " + 
                             f"prefactor: {prefactor}.")
            if accept:
                offsprings[0].info['key_value_pairs'][self.ak] = 1
                offsprings[1].info['key_value_pairs'][self.ak] = 1                
                parprint(f'Replicas {rep1} {rep2} inter move accepted!')
            else:
                offsprings[0].info['key_value_pairs'][self.ak] = 0
                offsprings[1].info['key_value_pairs'][self.ak] = 0
                parprint(f'Replicas {rep1} {rep2} inter move rejected!')

        elif len(parents) == 1:
            # Mutation
            rep = parents[0].info['key_value_pairs'][self.rk]
            T = parents[0].info['key_value_pairs'][self.tk]
            pscores = parents[0].info['data']['raw_scores']
            oscores = offsprings[0].info['data']['raw_scores']
            if any(o for o in offsprings if get_raw_score(o) <= raw_score_cutoff):
                prefactor = None
                accept = False
            else:    
                lnSp = logsumexp(pscores / (kB*T)) # np.sum(np.exp(pscores / (kB*T)))
                lnSo = logsumexp(oscores / (kB*T)) # np.sum(np.exp(oscores / (kB*T)))
                if prefactor == 'gc':
                    prefactor = compute_intramove_prefactor(
                            parents[0], offsprings[0], 
                            volume, nonmetal_species, 
                            temperature_species_internal_map,
                            no_internal_indices,
                            metal_species_probabilities,
                            temperature_key=self.tk)
             
                p_acc = min([1, prefactor * np.exp(lnSo - lnSp)])
                accept = p_acc > np.random.uniform()
            offsprings[0].info['key_value_pairs'][self.rk] = rep 
            offsprings[0].info['key_value_pairs'][self.tk] = T
            if self.store_delta:
                offsprings[0].info['data']['delta'] = oscores - pscores
            if self.verbose:
                parprint(f"Replica: {rep}, parent: {parents[0].get_chemical_formula()}, " +
                         f"offspring: {offsprings[0].get_chemical_formula()}, " + 
                         f"operation: {offsprings[0].info['key_value_pairs']['origin']}, " + 
                         f"prefactor: {prefactor}.")
            if accept:
                offsprings[0].info['key_value_pairs'][self.ak] = 1
                parprint(f'Replica {rep} intra move accepted!') 
            else:
                offsprings[0].info['key_value_pairs'][self.ak] = 0
                parprint(f'Replica {rep} intra move rejected!') 
                    
        return offsprings 

    def exchange(self, keep_composition=False, cand_pairs=None,
                 exchange_within_temperature=False, prefactor=1,
                 reaction_conditions=None, element_column_map=None,
                 temperature_species_internal_map=None, 
                 no_internal_indices=None):

        if cand_pairs is None:
            if exchange_within_temperature:
                cand_pairs = self.get_exchange_pairs_between_replicas(keep_composition)
            else:
                cand_pairs = self.get_exchange_pairs_between_temperatures(keep_composition)

        new_pop = []
        for c12 in cand_pairs:
            if len(c12) == 1:
                new_pop += list(c12)
                continue
            c1, c2 = c12
            rep1 = c1.info['key_value_pairs'][self.rk]
            rep2 = c2.info['key_value_pairs'][self.rk]
            T1 = c1.info['key_value_pairs'][self.tk]
            T2 = c2.info['key_value_pairs'][self.tk]
            f1 = c1.info['data']['raw_scores']
            f2 = c2.info['data']['raw_scores']
            if prefactor == 'gc':
                c12_prefactor = compute_exchange_prefactor(
                        c1, c2, reaction_conditions, 
                        element_column_map,
                        temperature_species_internal_map,
                        no_internal_indices,
                        temperature_key=self.tk,
                        exponent=0)
            else:
                c12_prefactor = prefactor

            lnS1T1 = logsumexp(f1 / (kB*T1))
            lnS1T2 = logsumexp(f1 / (kB*T2))
            lnS2T1 = logsumexp(f2 / (kB*T1))
            lnS2T2 = logsumexp(f2 / (kB*T2))
            p_acc = min([1, c12_prefactor * np.exp(lnS2T1 - lnS1T1 + lnS1T2 - lnS2T2)])
            accept = p_acc > np.random.uniform()

            if self.verbose:
                parprint(f"Replicas: {[rep1, rep2]}, " +
                         f"Formulas: {[c1.get_chemical_formula(), c2.get_chemical_formula()]}, " +
                         f"operation: exchange, prefactor: {c12_prefactor}.")
            if accept:
                c1.info['key_value_pairs'][self.rk], c2.info['key_value_pairs'][self.rk] = \
                        c2.info['key_value_pairs'][self.rk], c1.info['key_value_pairs'][self.rk]
                c1.info['key_value_pairs'][self.tk], c2.info['key_value_pairs'][self.tk] = \
                        c2.info['key_value_pairs'][self.tk], c1.info['key_value_pairs'][self.tk]
                parprint(f'Replicas {rep1} and {rep2} exchange move accepted!')
            else:
                parprint(f'Replicas {rep1} and {rep2} exchange move rejected!')
            new_pop += [c1, c2]

        new_pop.sort(key=lambda x: x.info['key_value_pairs'][self.rk])
        self.pop = new_pop
