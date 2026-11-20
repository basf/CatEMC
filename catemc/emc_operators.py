#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implementation for evolutionary Monte Carlo (EMC) operators"""
from ase.ga.offspring_creator import OffspringCreator
from acat.ga.particle_crossovers import SimpleCutSpliceCrossover
from acat.ga.adsorbate_operators import CatalystAdsorbateCrossover
from acat.ga.slab_operators import RandomSlabMutation, RandomSlabPermutation
from acat.ga.adsorbate_operators import (
        AddAdsorbate, 
        RemoveAdsorbate,
        MoveAdsorbate, 
        ReplaceAdsorbate,
        ReplaceAdsorbateSpecies)
from acat.build.action import remove_adsorbate_from_site
from collections import Counter
from copy import copy
import numpy as np
import random


class EMCCutSpliceCrossover(SimpleCutSpliceCrossover):

    def __init__(self, elements=None, 
                 keep_composition=True, 
                 allowed_indices=None):

        SimpleCutSpliceCrossover.__init__(self,
                elements=elements,
                keep_composition=keep_composition,
                allowed_indices=allowed_indices)
        self.descriptor = 'EMCCutSpliceCrossover'

    def get_new_individual(self, parents):
        """Cut-splice crossover with a symmetric proposal. ACAT restores the 
        composition of each offspring by changing randomly chosen atoms 
        after the two halves are joined, which has no reverse proposal with 
        the same probability (and can fail to restore the composition). 
        Here the sites on one side of a random plane are divided into 
        classes (a, b), where a and b are the elements of the first and the 
        second parent at that site. For every pair of elements, the same 
        number of sites is chosen at random from the classes (a, b) and 
        (b, a), and the two parents exchange their elements at the chosen 
        sites. Both compositions are preserved exactly, and the reverse 
        move chooses the same sites with the same probability, since the 
        size of each class does not change."""

        if not self.keep_composition:
            return super().get_new_individual(parents=parents, return_both=True)

        f0, m0 = parents
        if self.allowed_indices is None:
            indices = list(range(len(f0)))
        else:
            indices = list(self.allowed_indices)
        fsyms = f0.get_chemical_symbols()
        msyms = m0.get_chemical_symbols()

        # Random plane through the geometric center of the first parent. 
        # The center of mass is not used because it changes with the move
        e = np.array([random.gauss(0., 1.) for _ in range(3)])
        e /= np.linalg.norm(e)
        positions = f0.get_positions()[indices]
        side = np.dot(positions - positions.mean(axis=0), e) > 0.
        classes = {}
        for i, s in zip(indices, side):
            a, b = fsyms[i], msyms[i]
            if (not s) or (a == b):
                continue
            if (self.elements is not None) and (
            (a not in self.elements) or (b not in self.elements)):
                continue
            classes.setdefault((a, b), []).append(i)

        syms1, syms2 = list(fsyms), list(msyms)
        for (a, b) in sorted(classes.keys()):
            if (a > b) or ((b, a) not in classes):
                continue
            ids1, ids2 = classes[(a, b)], classes[(b, a)]
            k = min(len(ids1), len(ids2))
            for i in random.sample(ids1, k) + random.sample(ids2, k):
                syms1[i], syms2[i] = msyms[i], fsyms[i]

        indis = []
        for parent, syms in ((f0, syms1), (m0, syms2)):
            indi = parent.copy()
            indi.set_chemical_symbols(syms)
            indi = self.initialize_individual(parent, indi)
            indi.info['data']['parents'] = [i.info['confid'] for i in parents]
            indi.info['data']['operation'] = 'crossover'
            indis.append(self.finalize_individual(indi))
        parent_message = ':Parents {0} {1}'.format(f0.info['confid'],
                                                   m0.info['confid'])

        return (indis, self.descriptor + parent_message)


class EMCCatalystAdsorbateCrossover(CatalystAdsorbateCrossover):

    def __init__(self, group_by_sites=False, catalyst_indices=None):

        CatalystAdsorbateCrossover.__init__(self,
                group_by_sites=group_by_sites,
                catalyst_indices=catalyst_indices)
        self.descriptor = 'EMCCatalystAdsorbateCrossover'

    def get_new_individual(self, parents):

        return super().get_new_individual(parents=parents, return_both=True)


class EMCInterSwap(OffspringCreator):

    def __init__(self, element_pools=None,
                 element_probabilities=None,
                 allowed_indices=None,
                 distribution_correction_function=None,
                 filter_function=None):

        OffspringCreator.__init__(self, num_muts=1)

        self.descriptor = 'EMCInterSwap'
        self.element_pools = element_pools
        if self.element_pools is not None:
            if not any(isinstance(i, list) for i in self.element_pools):
                self.element_pools = [self.element_pools]
        self.element_probabilities = element_probabilities
        self.allowed_indices = allowed_indices
        self.dcf = distribution_correction_function
        self.ff = filter_function
        self.min_inputs = 2
        if self.dcf is not None or self.ff is not None:
            raise ValueError('Custom interswap callbacks require a reverse proposal probability.')

    def _eligible_indices(self, parent):

        idxs = list(range(len(parent)))
        if self.element_pools is not None:
            elem_pool = set(e for pool in self.element_pools for e in pool)
            idxs = [i for i in idxs if parent[i].symbol in elem_pool]
        if self.allowed_indices is not None:
            idxs = [i for i in idxs if i in self.allowed_indices]

        return idxs

    def find_possible_pairs(self, parents):
        """Return a list of viable (i1, i2) index pairs for interswap.
        In each pair, parents[0][i1].symbol != parents[1][i2].symbol"""

        if len(parents) != 2:
            return []
        f, m = parents
        options1 = self._eligible_indices(f)
        options2 = self._eligible_indices(m)
        pairs = [(i1, i2) for i1 in options1 for i2 in options2
                 if f[i1].symbol != m[i2].symbol and
                 (self.element_pools is None or any(
                     f[i1].symbol in pool and m[i2].symbol in pool
                     for pool in self.element_pools))]

        return pairs

    def can_interswap(self, parents):

        return len(self.find_possible_pairs(parents)) > 0

    def get_new_individual(self, parents):

        f, m = parents
        parent_message = ':Parents {0} {1}'.format(f.info['confid'], m.info['confid'])
        possible_pairs = self.find_possible_pairs(parents)
        if not possible_pairs:
            raise ValueError('No interswap possible between provided parents with given element_pools and allowed_indices')

        # Choose one feasible (i1, i2) pair. If element_probabilities provided,
        # weight pairs by product of probabilities of the two involved symbols.
        if self.element_probabilities is None:
            i1, i2 = random.choice(possible_pairs)
        else:
            weights = []
            for (i1_, i2_) in possible_pairs:
                s1 = f[i1_].symbol
                s2 = m[i2_].symbol
                w1 = self.element_probabilities.get(s1, 0.0)
                w2 = self.element_probabilities.get(s2, 0.0)
                weights.append(w1 * w2)
            weights = np.array(weights, dtype=float)
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.):
                raise ValueError('Interswap weights must be finite and nonnegative.')
            if weights.sum() == 0:
                # fallback to uniform if probabilities don't cover available symbols
                i1, i2 = random.choice(possible_pairs)
            else:
                p = weights / weights.sum()
                choice_idx = np.random.choice(len(possible_pairs), p=p)
                i1, i2 = possible_pairs[choice_idx]
        sym1 = f[i1].symbol
        sym2 = m[i2].symbol

        # Build mutation operators that swap sym1<->sym2 on the two parents
        op1 = EMCMutation(element_pools=self.element_pools,
                          allowed_indices=self.allowed_indices,
                          distribution_correction_function=self.dcf,
                          filter_function=self.ff,
                          from_element=sym1, to_element=sym2)
        op2 = EMCMutation(element_pools=self.element_pools,
                          allowed_indices=self.allowed_indices,
                          distribution_correction_function=self.dcf,
                          filter_function=self.ff,
                          from_element=sym2, to_element=sym1)

        indi1, _ = op1.get_new_individual([f])
        indi2, _ = op2.get_new_individual([m])

        reverse_pairs = self.find_possible_pairs([indi1, indi2])
        if self.element_probabilities is None:
            q_forward, q_reverse = 1. / len(possible_pairs), 1. / len(reverse_pairs)
        else:
            reverse_weight = sum(self.element_probabilities.get(indi1[i].symbol, 0.) *
                                 self.element_probabilities.get(indi2[j].symbol, 0.)
                                 for i, j in reverse_pairs)
            pair_weight = self.element_probabilities.get(sym1, 0.) * self.element_probabilities.get(sym2, 0.)
            q_forward = pair_weight / weights.sum() if weights.sum() > 0. else 1. / len(possible_pairs)
            q_reverse = pair_weight / reverse_weight if reverse_weight > 0. else 1. / len(reverse_pairs)
        log_ratio = np.log(q_reverse) - np.log(q_forward) if q_reverse > 0. else -np.inf
        for indi in (indi1, indi2):
            indi.info['data']['mc_log_proposal_ratio'] = float(log_ratio)

        indi1.info['data']['parents'] = [i.info['confid'] for i in parents]
        indi1.info['data']['operation'] = 'interswap'
        indi2.info['data']['parents'] = [i.info['confid'] for i in parents]
        indi2.info['data']['operation'] = 'interswap'

        return ([self.finalize_individual(indi1), self.finalize_individual(indi2)],
                self.descriptor + parent_message)


class EMCIntraSwap(RandomSlabPermutation):

    def __init__(self, element_pools=None,
                 allowed_indices=None,
                 distribution_correction_function=None,
                 filter_function=None,):

        RandomSlabPermutation.__init__(self, 
                verbose=False, 
                num_muts=1,
                element_pools=element_pools,
                allowed_indices=allowed_indices,
                distribution_correction_function=distribution_correction_function,
                filter_function=filter_function,
                )
                                        
        self.descriptor = 'EMCIntraSwap'

    def get_new_individual(self, parents):

        # ACAT changes allowed_indices during the call. Keep that change local.
        operator = copy(self)
        indi, msg = RandomSlabPermutation.get_new_individual(operator, parents)
        if indi is None:
            indi = self.initialize_individual(parents[0], parents[0])
            indi.info['data']['mc_log_proposal_ratio'] = -np.inf
            indi = self.finalize_individual(indi)

        indi.info['data']['parents'] = [i.info['confid'] for i in parents] 
        indi.info['data']['operation'] = 'intraswap'

        return (indi, msg)


class EMCMutation(RandomSlabMutation):

    def __init__(self, element_pools=None,
                 element_probabilities=None,
                 allowed_indices=None,
                 distribution_correction_function=None,
                 filter_function=None,
                 from_element=None,
                 to_element=None):

        RandomSlabMutation.__init__(self, 
                verbose=False, 
                num_muts=1,
                element_pools=element_pools,
                element_probabilities=element_probabilities,
                allowed_indices=allowed_indices,
                distribution_correction_function=distribution_correction_function,
                filter_function=filter_function,
                from_element=from_element,
                to_element=to_element)

        self.descriptor = 'EMCMutation'
        if distribution_correction_function is not None or filter_function is not None:
            raise ValueError('Custom mutation callbacks require a reverse proposal probability.')

    def get_new_individual(self, parents):

        # ACAT changes allowed_indices during the call. Keep that change local.
        operator = copy(self)
        indi, msg, mutated_elements = RandomSlabMutation.get_new_individual(
                operator, parents, return_mutated_elements=True)
        indices = list(range(len(indi))) if operator.allowed_indices is None else operator.allowed_indices
        m, n = mutated_elements
        probabilities = []
        for atoms, source, target in ((parents[0][indices], m, n), (indi[indices], n, m)):
            if self.element_pools is None:
                options = set(atoms.symbols) - {source}
            else:
                options = {e for pool in self.element_pools if source in pool for e in pool if e != source}
            if target not in options:
                probability = 0.
            elif self.element_probabilities is None or any(e not in self.element_probabilities for e in options):
                probability = 1. / len(options)
            else:
                weights = np.array([self.element_probabilities[e] for e in options], dtype=float)
                if not np.all(np.isfinite(weights)) or np.any(weights < 0.) or weights.sum() <= 0.:
                    raise ValueError('Mutation weights must be finite, nonnegative and have positive sum.')
                probability = self.element_probabilities[target] / weights.sum()
            probabilities.append(probability)
        q_forward, q_reverse = probabilities
        # A directed mutation alone has no reverse. EMCInterSwap supplies the
        # complete ratio for its paired directed mutations below this level.
        if self.from_element is not None or self.to_element is not None:
            q_reverse = 0.
        indi.info['data']['mc_log_proposal_ratio'] = (
                float(np.log(q_reverse) - np.log(q_forward)) if q_reverse > 0. else -np.inf)

        indi.info['data']['parents'] = [i.info['confid'] for i in parents] 
        indi.info['data']['operation'] = 'mutation'
        indi.info['data']['mutated_elements'] = mutated_elements

        return (indi, msg)


def choose_fragment_type(adsorbate_species, species_probabilities=None):
    """Choose a fragment type with fixed probabilities that do not depend 
    on the current adlayer (uniform if no probabilities are provided)."""

    if (species_probabilities is None) or sum(
    species_probabilities[a] for a in adsorbate_species) == 0.:
        return random.choice(adsorbate_species)
    probs = [species_probabilities[a] for a in adsorbate_species]

    return random.choices(k=1, population=adsorbate_species, weights=probs)[0]


def is_insertable_fragment_type(species, adsorbate_species, species_probabilities=None):
    """Whether a fragment type can be selected by choose_fragment_type."""

    if species not in adsorbate_species:
        return False
    if (species_probabilities is None) or sum(
    species_probabilities[a] for a in adsorbate_species) == 0.:
        return True

    return species_probabilities[species] > 0.


def get_fragment_sites(hetero_site_list, adsorbate_species, fragmentation=True):
    """Return one (label, site index, remove_fragment) tuple per fragment in 
    the adlayer. The fragments are labelled in the same way as the fragment 
    bookkeeping that defines the fragment counts N_k."""

    fragments = {}
    for i, st in enumerate(hetero_site_list):
        if not st['occupied']:
            continue
        if st['adsorbate'] in adsorbate_species:
            label, indices, remove_fragment = st['adsorbate'], st['adsorbate_indices'], False
        elif fragmentation:
            label, indices, remove_fragment = st['fragment'], st['fragment_indices'], True
        else:
            continue
        key = tuple(sorted(indices))
        if key not in fragments:
            fragments[key] = (label, i, remove_fragment)

    return list(fragments.values())


class EMCAddAdsorbate(AddAdsorbate):
    """Fragment insertion for GC-EMC. The fragment type is chosen with 
    fixed probabilities, i.e. independently of the current adlayer, and
    EMCRemoveAdsorbate chooses the type of the fragment to remove with 
    the same probabilities. The type selection therefore cancels in the
    proposal ratio and the acceptance probability reduces to the 
    continuum reference expression with the 1/(N_k+1) factor. If the 
    insertion is not possible (maximum number of species or maximum 
    coverage reached, or no free site), the move is rejected instead of 
    being turned into a removal, which would have no reverse proposal."""

    def add_adsorbate(self, atoms, hetero_site_list, heights, 
                      adsorbate_species=None, 
                      min_adsorbate_distance=2., 
                      tilt_angle=0.):

        adsorbate = choose_fragment_type(self.adsorbate_species, 
                                         self.species_probabilities)
        if self.max_species is not None:
            occupied_specs = {st['adsorbate'] for st in hetero_site_list if st['occupied']
                              and (st['adsorbate'] in self.adsorbate_species)}
            if (adsorbate not in occupied_specs) and (len(occupied_specs) >= self.max_species):
                return False

        return AddAdsorbate.add_adsorbate(self, atoms, hetero_site_list, heights,
                                          [adsorbate], min_adsorbate_distance,
                                          tilt_angle=tilt_angle)

    def remove_adsorbate(self, atoms, hetero_site_list, return_site_index=False,
                         fragmentation=True):

        return atoms

    def get_new_individual(self, parents):

        indi, msg = AddAdsorbate.get_new_individual(self, parents)
        if len(indi) <= len(parents[0]):
            indi.info['data']['mc_log_proposal_ratio'] = -np.inf

        return (indi, msg)


class EMCRemoveAdsorbate(RemoveAdsorbate):
    """Fragment removal for GC-EMC. The fragment type is chosen first, with 
    the same fixed probabilities as in EMCAddAdsorbate, then one existing 
    fragment of that type is removed at random. This is the reverse of the
    insertion proposal assumed by the N_k factor in the removal acceptance 
    probability. If there is no fragment of the chosen type, the move is 
    rejected instead of being turned into an insertion."""

    def __init__(self, adsorbate_species, species_probabilities=None, **kwargs):

        RemoveAdsorbate.__init__(self, adsorbate_species, **kwargs)
        self.species_probabilities = species_probabilities

    def remove_adsorbate(self, atoms, hetero_site_list, return_site_index=False,
                         fragmentation=True):

        adsorbate = choose_fragment_type(self.adsorbate_species, 
                                         self.species_probabilities)
        candidates = [t for t in get_fragment_sites(hetero_site_list, 
                      self.adsorbate_species, fragmentation) if t[0] == adsorbate]
        if not candidates:
            return atoms
        _, i, remove_fragment = random.choice(candidates)
        remove_adsorbate_from_site(atoms, hetero_site_list[i], 
                                   remove_fragment=remove_fragment)

        return atoms

    def get_new_individual(self, parents):

        indi, msg = RemoveAdsorbate.get_new_individual(self, parents)
        if len(indi) >= len(parents[0]):
            indi.info['data']['mc_log_proposal_ratio'] = -np.inf

        return (indi, msg)


class _RejectIncompleteMove(object):
    """Reject a fragment displacement or mutation that could not be completed. 
    ACAT then inserts a fragment (empty adlayer) or leaves the fragment removed 
    (no valid site to put it back), and neither of them has a reverse proposal."""

    def remove_adsorbate(self, *args, **kwargs):

        removed = super().remove_adsorbate(*args, **kwargs)
        if removed is False:
            self._incomplete = True

        return removed

    def add_adsorbate(self, *args, **kwargs):

        added = super().add_adsorbate(*args, **kwargs)
        if added is False:
            self._incomplete = True

        return added

    def get_new_individual(self, parents):

        self._incomplete = False
        indi, msg = super().get_new_individual(parents)
        if self._incomplete:
            indi.info['data']['mc_log_proposal_ratio'] = -np.inf

        return (indi, msg)


class EMCMoveAdsorbate(_RejectIncompleteMove, MoveAdsorbate):
    pass


class EMCReplaceAdsorbate(_RejectIncompleteMove, ReplaceAdsorbate):
    """Fragment mutation for GC-EMC. The fragment type k to replace is chosen 
    first, with the same fixed probabilities as in EMCAddAdsorbate, then one 
    existing fragment of that type is chosen at random and replaced by a 
    fragment of a different type l. The proposal ratio is then N_k / (N_l + 1) 
    times the ratio of the type selection probabilities, which is what the 
    factorials in the acceptance probability assume (for a fragment chosen 
    uniformly among all fragments the N_k / (N_l + 1) factor would have to be 
    left out). The move is rejected if there is no fragment of the chosen 
    type, if no allowed new type exists, or if the new fragment cannot be 
    placed."""

    def remove_adsorbate(self, atoms, hetero_site_list, return_site_index=False,
                         fragmentation=True):

        adsorbate = choose_fragment_type(self.adsorbate_species,
                                         self.species_probabilities)
        fragments = get_fragment_sites(hetero_site_list, self.adsorbate_species, 
                                       fragmentation)
        candidates = [t for t in fragments if t[0] == adsorbate]
        if not candidates:
            self._incomplete = True
            return False
        chosen = random.choice(candidates)
        _, i, remove_fragment = chosen
        # Fragment types left in the adlayer after the removal
        self._remaining_types = {t[0] for t in fragments if t is not chosen}
        remove_adsorbate_from_site(atoms, hetero_site_list[i], 
                                   remove_fragment=remove_fragment)
        hetero_site_list[i]['fragment'] = adsorbate

        return i

    def add_adsorbate(self, atoms, hetero_site_list, heights, 
                      adsorbate_species=None, 
                      min_adsorbate_distance=2., 
                      tilt_angle=0.):

        if self._incomplete:
            return False
        # New types with the same selection probability as the replaced type, 
        # not exceeding the maximum number of species 
        options = [s for s in adsorbate_species if is_insertable_fragment_type(
                   s, self.adsorbate_species, self.species_probabilities)]
        if (self.max_species is not None) and (
        len(self._remaining_types) >= self.max_species):
            options = [s for s in options if s in self._remaining_types]
        if not options:
            self._incomplete = True
            return False

        return super().add_adsorbate(atoms, hetero_site_list, heights, 
                                     options, min_adsorbate_distance,
                                     tilt_angle=tilt_angle)


class EMCReplaceAdsorbateSpecies(ReplaceAdsorbateSpecies):
    """Converting all fragments of one type into another type can only be
    reversed by the same operator if the new type was absent before the move
    and the old type can be selected as a new type. All other outcomes (new 
    type already present, old type with zero selection probability, or an 
    insertion into an empty adlayer) have no reverse proposal and are rejected."""

    def get_new_individual(self, parents):

        f = parents[0]
        indi, msg = ReplaceAdsorbateSpecies.get_new_individual(self, parents)
        reversible = False
        if ('fragments' in f.info['data']) and ('fragments' in indi.info['data']):
            ct1 = Counter(f.info['data']['fragments'])
            ct2 = Counter(indi.info['data']['fragments'])
            lost, gained = ct1 - ct2, ct2 - ct1
            if (len(lost) == 1) and (len(gained) == 1) and (
            sum(lost.values()) == sum(gained.values())):
                to_spec = list(gained.keys())[0]
                from_spec = list(lost.keys())[0]
                if self.catalyst_indices is None:
                    n_lattice = 0
                else:
                    n_lattice = len([i for i in self.catalyst_indices 
                                     if (i < len(f)) and (f[i].symbol == to_spec)])
                reversible = (ct1[to_spec] - n_lattice == 0) and is_insertable_fragment_type(
                             from_spec, self.adsorbate_species, self.species_probabilities)
        if not reversible:
            indi.info['data']['mc_log_proposal_ratio'] = -np.inf

        return (indi, msg)
