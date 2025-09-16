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
        
    def get_new_individual(self, parents):
        f, m = parents 
        parent_message = ':Parents {0} {1}'.format(f.info['confid'],
                                                   m.info['confid'])
        options1 = list(range(len(f)))
        if self.element_pools is not None:
            element_pool = list({e for pool in self.element_pools for e in pool})
            options1 = [i for i in options1 if f[i].symbol in element_pool]
        if self.allowed_indices is not None:
            options1 = [i for i in options1 if i in self.allowed_indices]
        if self.element_probabilities is None:
            i1 = random.choice(options1) 
            sym1 = f[i1].symbol
        else:
            sym_set = random.choice([s for s in set(f.symbols[options1])])
            composition_rsv = {k: v for k, v in self.element_probabilities.items() if k in sym_set}
            composition_rsv_p = np.array(list(composition_rsv.values()))
            sym1 = np.random.choice(np.array(list(composition_rsv.keys())), 
                                    p=composition_rsv_p/composition_rsv_p.sum())

        options2 = [i for i in range(len(m)) if m[i].symbol != sym1]
        if self.element_pools is not None:
            element_pool = list({e for pool in self.element_pools for e in pool})
            options2 = [i for i in options2 if m[i].symbol in element_pool]
        if self.allowed_indices is not None:
            options2 = [i for i in options2 if i in self.allowed_indices]
        if self.element_probabilities is None:
            i2 = random.choice(options2)
            sym2 = m[i2].symbol
        else:
            sym_set = random.choice([s for s in set(m.symbols[options2])])
            composition_rsv = {k: v for k, v in self.element_probabilities.items() if k in sym_set}
            composition_rsv_p = np.array(list(composition_rsv.values()))
            sym2 = np.random.choice(np.array(list(composition_rsv.keys())), 
                                    p=composition_rsv_p/composition_rsv_p.sum())

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
        indi, _ = op1.get_new_individual([f])
        indi2, _ = op2.get_new_individual([m])

        indi.info['data']['parents'] = [i.info['confid'] for i in parents] 
        indi.info['data']['operation'] = 'interswap'
        indi2.info['data']['parents'] = [i.info['confid'] for i in parents]
        indi2.info['data']['operation'] = 'intraswap'

        return ([self.finalize_individual(indi), self.finalize_individual(indi2)],
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
