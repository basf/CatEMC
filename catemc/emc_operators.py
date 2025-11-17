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

    def get_new_individual(self, parents):

        indi, msg, mutated_elements = super().get_new_individual(
                parents=parents, return_mutated_elements=True)

        indi.info['data']['parents'] = [i.info['confid'] for i in parents] 
        indi.info['data']['operation'] = 'mutation'
        indi.info['data']['mutated_elements'] = mutated_elements

        return (indi, msg)
