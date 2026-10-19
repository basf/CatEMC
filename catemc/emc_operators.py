#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implementation for evolutionary Monte Carlo (EMC) operators"""
from ase.ga.offspring_creator import OffspringCreator
from acat.ga.particle_crossovers import SimpleCutSpliceCrossover
from acat.ga.adsorbate_operators import CatalystAdsorbateCrossover
from acat.ga.slab_operators import RandomSlabMutation, RandomSlabPermutation
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

        return super().get_new_individual(parents=parents, return_both=True)


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
            raise ValueError('EMCInterSwap requires an explicit reverse proposal; '
                             'custom correction/filter callbacks are not supported.')

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
        pairs = [(i1, i2) for i1 in options1 for i2 in options2 if f[i1].symbol != m[i2].symbol]

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
            if weights.sum() == 0:
                # fallback to uniform if probabilities don't cover available symbols
                i1, i2 = random.choice(possible_pairs)
            else:
                p = weights / weights.sum()
                choice_idx = np.random.choice(len(possible_pairs), p=p)
                i1, i2 = possible_pairs[choice_idx]
        sym1 = f[i1].symbol
        sym2 = m[i2].symbol

        # Exchange the selected indices.  For labelled site assignments the
        # target has no species-count factorials.  The complete proposal ratio
        # is the reverse/forward probability of this same index pair.
        indi1 = self.initialize_individual(f, f)
        indi2 = self.initialize_individual(m, m)
        indi1[i1].symbol, indi2[i2].symbol = sym2, sym1
        def pair_probability(a, b, pairs):
            if self.element_probabilities is None:
                return 1. / len(pairs)
            w = np.array([self.element_probabilities.get(a[i].symbol, 0.) *
                          self.element_probabilities.get(b[j].symbol, 0.)
                          for i, j in pairs], dtype=float)
            if not np.all(np.isfinite(w)) or np.any(w < 0.):
                raise ValueError('Interswap weights must be finite and nonnegative.')
            return (w[pairs.index((i1, i2))] / w.sum()
                    if w.sum() > 0. else 1. / len(pairs))
        reverse_pairs = self.find_possible_pairs([indi1, indi2])
        qf = pair_probability(f, m, possible_pairs)
        qr = pair_probability(indi1, indi2, reverse_pairs)
        log_ratio = np.log(qr) - np.log(qf) if qr > 0. else -np.inf
        for parent, child in zip(parents, [indi1, indi2]):
            if 'groups' in parent.info.get('data', {}):
                child.info['data']['groups'] = parent.info['data']['groups']
            child.info['data']['mc_log_proposal_ratio'] = float(log_ratio)

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

        indi, msg = super().get_new_individual(parents=parents)

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
            raise ValueError('EMCMutation requires an explicit reverse proposal; '
                             'custom correction/filter callbacks are not supported.')
        if from_element is not None or to_element is not None:
            raise ValueError('Directed mutations have no reverse move in this '
                             'operator; use EMCInterSwap for paired exchanges.')

    def _species_probability(self, source, target, atoms):
        """Actual conditional species proposal used by ACAT, including fallback."""
        if self.element_pools is None:
            options = set(atoms.get_chemical_symbols()) - {source}
        else:
            options = {e for pool in self.element_pools if source in pool
                       for e in pool if e != source}
        if target not in options:
            return 0.
        p = self.element_probabilities
        if p is None or any(e not in p for e in options):
            return 1. / len(options)
        weights = np.array([p[e] for e in options], dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.) or weights.sum() <= 0.:
            raise ValueError('Mutation weights must be finite, nonnegative and have positive sum.')
        return p[target] / weights.sum()

    def get_new_individual(self, parents):

        indi, msg, mutated_elements = super().get_new_individual(
                parents=parents, return_mutated_elements=True)

        indices = (list(range(len(indi))) if self.allowed_indices is None
                   else self.allowed_indices)
        m, n = mutated_elements
        qf = self._species_probability(m, n, parents[0][indices])
        qr = self._species_probability(n, m, indi[indices])
        # The uniform atom-index probabilities cancel.  Raw composition
        # weights are normalized within the actual accessible species pool.
        indi.info['data']['mc_log_proposal_ratio'] = (
                float(np.log(qr) - np.log(qf)) if qr > 0. else -np.inf)

        indi.info['data']['parents'] = [i.info['confid'] for i in parents] 
        indi.info['data']['operation'] = 'mutation'
        indi.info['data']['mutated_elements'] = mutated_elements

        return (indi, msg)
