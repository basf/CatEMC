#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Algorithm for Gibbs ensemble evolutionary Monte Carlo (GE-EMC) run"""
from .emc_operators import (
        EMCCutSpliceCrossover,
        EMCInterSwap,
        EMCIntraSwap)
from .emc import EMC
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.ga.data import DataConnection, PrepareDB
from ase.parallel import parprint
import numpy as np
import warnings
import random
import shutil
import time
import os


class GEEMC(EMC):

    """
    Main algorithm for Gibbs ensemble evolutionary Monte Carlo (GE-EMC).

    Parameters
    ----------
    name: str
        Name of the target system.

    calculation_dir: str
        Path of the calculation directory.

    calculation_script: str
        The name of the Python script for calculating the fitness of each
        configuration. The script must be inside calculation_dir.

    input_pop: list of ase.Atoms objects
        Input population for the EMC run.

    fix_indices: list of ints, default None
        A list of atom indices that should be immutable and fixed in positions
        throughout the EMC run. Nothing is fixed by default.

    trim_generation: int, default 100
        To trim database (with backup) every N steps. The database file would be
        too large and adding new data would be too slow without trimming.

    pressure: float, default None
        The common pressure target to perform NPT GE-EMC MD and cell relaxation.
        If this is not set, run NVT GE-EMC without changing cell volumes.

    md_generation: int, default 10
        To perform molecular dynamics every N steps.

    exchange_generation: int, defatul 1
        To perform replica exchange every N steps.

    bias_starting_step: int, default None
        From which step should a the interswap operation be biased towards uniform
        metal species selection instead of uniform metal atom index selection.
        Select by metal atom index if this is not set.

    crossover_rate: float, default 0.
        The rate of performing crossover operations.

    metal_interswap_rate: float, default 0.
        The rate of performing metal atom interswap operations.

    nonmetal_interswap_rate: float, default 0.
        The rate of performing non-metal atom (including vacancy 'X') interswap 
        operations.

    intraswap_rate: float, default 0.
        The rate of performing intraswap operations in the catalyst slab.

    metal_element_pool: list of strs, default None
        The list of target metal elements (including vacancy 'X') that should be 
        mutable during the EMC run.

    nonmetal_element_pool: list of strs, default None
        The list of target non-metal elements (including vacancy 'X') that should 
        be mutable during the EMC run.

    mutable_metal_indices: list of ints, default None
        The list of metal atom (or 'X' which represents vacancy) indices that are 
        allowed to mutate during the EMC run. Default None means all metal indices 
        are mutable.

    mutable_nonmetal_indices: list of ints, default None
        The list of non-metal (or 'X' which represents vacancy) indices that are
        allowed to mutate during the EMC run. Default None means all non-metal 
        indices are mutable.

    raw_score_cutoff: float, default -10000.
        The fitness score for candidates that are should not be considered in the 
        sampling. This could be structures that are unphysical / failed to relax by 
        the calculator.

    temperature_key: str, default 'temperature'
        Keyword for the temperature stored in each configuration.

    replica_key: str, default 'replica'
        Keyword for the replica number stored in each configuration.

    acceptance_key: str, default 'accepted'
        Keyword for indicating being accepted or not in each configuration

    interswap_within_key: str, default None
        The key in atoms.info['key_value_pair'] that categorizes replicas into
        different groups, so that interswap is only performed within each group.
        Useful for e.g. simulating multiple bulk compositions in a single run.

    gpu_devices: list of ints, default None
        The list of GPU devices to use. Default is to use a single GPU/CPU device.

    procs_per_device: int, default 1
        The number of processes to run in parallel on each GPU/CPU device. Defaul is 
        to run in serial.

    batching: bool, default False
        Whether to run batch relaxation and MD using torch-sim. Default is False.

    """

    def __init__(self, name,
            calculation_dir,
            calculation_script,
            input_pop,
            fix_indices=None,
            trim_generation=100,
            pressure=None,
            md_generation=10,
            exchange_generation=1,
            bias_starting_step=None,
            crossover_rate=0.,
            metal_interswap_rate=0.,
            nonmetal_interswap_rate=0.,
            intraswap_rate=0.,
            metal_element_pool=None,
            nonmetal_element_pool=None,
            mutable_metal_indices=None,
            mutable_nonmetal_indices=None,
            raw_score_cutoff=-10000.,
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted',
            interswap_within_key=None,
            gpu_devices=1,
            procs_per_device=10,
            batching=False,
            verbose=True):

        super().__init__(
                name=name, 
                calculation_dir=calculation_dir,
                calculation_script=calculation_script,
                input_pop=input_pop,
                fix_indices=fix_indices,
                trim_generation=trim_generation,
                md_generation=md_generation,
                exchange_generation=exchange_generation,
                metal_element_pool=metal_element_pool,
                nonmetal_element_pool=nonmetal_element_pool,
                mutable_metal_indices=mutable_metal_indices,
                mutable_nonmetal_indices=mutable_nonmetal_indices,
                raw_score_cutoff=raw_score_cutoff,
                temperature_key=temperature_key,
                replica_key=replica_key,
                acceptance_key=acceptance_key,
                gpu_devices=gpu_devices,
                procs_per_device=procs_per_device,
                batching=batching,
                verbose=verbose)

        self.pressure = pressure
        self.bias_starting_step = bias_starting_step
        self.crossover_rate = crossover_rate
        self.metal_interswap_rate = metal_interswap_rate
        self.nonmetal_interswap_rate = nonmetal_interswap_rate
        self.intraswap_rate = intraswap_rate
        self.interswap_within_key = interswap_within_key
        
    def run(self, starting_step=1, total_steps=10000, restart_with_trim=False):

        """                                    
        Parameters
        ----------
        starting_step: int, default 1
            (Re)starting from a certain EMC step.
                                               
        total_steps: int, default 10000
            The total number of EMC steps that the algorithm should run.

        restart_with_trim: bool, default False
            Whether to restart while trimming all data generated after
            the starting_step that has been generated before.

        """

        restart = True
        self.pop_size = len(self.input_pop)
        # Connect to the ase db
        asedb_path = self.calculation_dir + '/{}.db'.format(self.name)
        if not os.path.isfile(asedb_path):
            restart = False
            asedb = PrepareDB(asedb_path, population_size=self.pop_size)
            for atoms in self.input_pop:
                if 'key_value_pairs' not in atoms.info:
                    atoms.info['key_value_pairs'] = {}
                if 'data' not in atoms.info:
                    atoms.info['data'] = {}
                kwargs = {self.temperature_key: atoms.info['key_value_pairs'][self.temperature_key], 
                          self.replica_key: atoms.info['key_value_pairs'][self.replica_key],
                          self.acceptance_key: 0}
                asedb.add_unrelaxed_candidate(atoms, 
                        emc_step=atoms.info['key_value_pairs']['emc_step'],
                        data=atoms.info['data'],
                        **kwargs)
        if restart_with_trim:
            self.connect_data_with_trim(starting_step=starting_step)
        else:
            self.dc = DataConnection(asedb_path)
        if not os.path.isdir(f'{self.calculation_dir}/tmp_{self.name}'):
            os.mkdir(f'{self.calculation_dir}/tmp_{self.name}')
        if not restart:
            unrelaxed_candidates = self.dc.get_all_unrelaxed_candidates()
            relaxed_candidates = self.get_relaxed_candidates(unrelaxed_candidates, run_md=True, batching=self.batching)
            for atoms in relaxed_candidates:                  
                atoms.info['key_value_pairs'][self.acceptance_key] = 1
        else:
            relaxed_candidates = []

        self.ntask = 1
        self.pop = self.get_pop()                                             
        if self.trim_generation == -1:
            parprint('Restarting from the final population of the previously-converged EMC run. ' + 
                     'All other previously-stored structures are discarded from the ASE database.')
            self.trim_asedb()
            self.pop = self.get_pop() 
        # Update the population to allow new candidates to enter
        self.dc.add_more_relaxed_candidates(relaxed_candidates)
        self.pop.update()
        if not restart:
            for atoms in self.pop.pop:
                #atoms = atoms[[a.index for a in atoms if a.symbol != 'X']]
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = Trajectory(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', 'a')
                traj.write(atoms)
        if restart_with_trim:
            for atoms in self.pop.pop:
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = read(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', index=f':{starting_step}')
                write(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', traj)
        rates_sum = self.intraswap_rate + self.crossover_rate + self.metal_interswap_rate + self.nonmetal_interswap_rate 
        if rates_sum != 1:
            self.intraswap_rate /= rates_sum
            self.crossover_rate /= rates_sum
            self.metal_interswap_rate /= rates_sum
            self.nonmetal_interswap_rate /= rates_sum

        start_time = time.perf_counter()
        # Below is the iterative part of the algorithm
        gen_num = starting_step
        gi = 0
        while gen_num + gi <= total_steps:
            if (gen_num + gi) % self.trim_generation == 0:
                self.trim_asedb()
                self.pop = self.get_pop()
                self.pop.current_fitness = self.pop.__get_fitness__(self.pop.pop)
                parprint('ASE database trimed at generation {0}. Restart from current population.'.format(gen_num + gi))
            parprint('Creating and evaluating generation {0}'.format(gen_num + gi))
            
            if (gen_num + gi) % self.md_generation == 0:
                unrelaxed_candidates = self.pop.pop
                relaxed_candidates = self.get_relaxed_candidates(unrelaxed_candidates, run_md=True, batching=self.batching)
                for atoms in relaxed_candidates:
                    atoms.info['key_value_pairs'][self.acceptance_key] = 1
                # Update the population to allow new candidates to enter
                self.dc.add_more_relaxed_candidates(relaxed_candidates) 
                self.pop.update()

            n_crossover_pairs = round((self.pop_size * self.crossover_rate) / 2)
            n_interswap_pairs = round((self.pop_size * (self.metal_interswap_rate + self.nonmetal_interswap_rate)) / 2)
            all_parents = self.pop.get_all_parents(n_crossover_pairs, n_interswap_pairs,
                                                   interswap_within_temperature=True,
                                                   interswap_within_key=self.interswap_within_key)
            for parents in all_parents:
                if len(parents) == 2:
                    switch_to_interswap = False
                    if parents[0].info['data']['for_operation'] == 'crossover':
                        if self.nonmetal_element_pool is None:
                            element_pool = self.metal_element_pool
                            allowed_indices = self.mutable_metal_indices
                        else:
                            element_pool = self.metal_element_pool + self.nonmetal_element_pool
                            if (self.mutable_metal_indices is None) or (self.mutable_nonmetal_indices is None):
                                allowed_indices = None
                            else:
                                allowed_indices = self.mutable_metal_indices + self.mutable_nonmetal_indices
                        try:
                            operator = EMCCutSpliceCrossover(allowed_indices=allowed_indices, elements=element_pool, keep_composition=True)
                            offsprings, desc = operator.get_new_individual(parents)
                        except:
                            warnings.warn('CutSpliceCrossover failed, fallback to interswap.')
                            switch_to_interswap = True
                    if (parents[0].info['data']['for_operation'] == 'interswap') or switch_to_interswap:
                        if self.nonmetal_element_pool is None:
                            do_metal = True
                        else:
                            do_metal = random.choices(population=[True, False], weights=[self.metal_interswap_rate, self.nonmetal_interswap_rate])[0]
                        if do_metal and (self.bias_starting_step is not None) and (gen_num + gi >= self.bias_starting_step):
                            operator = EMCInterSwap(
                                    allowed_indices=self.mutable_metal_indices, 
                                    element_pools=self.metal_element_pool,
                                    element_probabilities={s: 1/len(self.metal_element_pool) for s in self.metal_element_pool})
                        elif do_metal:
                            operator = EMCInterSwap(allowed_indices=self.mutable_metal_indices, element_pools=self.metal_element_pool)
                        else:
                            operator = EMCInterSwap(allowed_indices=self.mutable_nonmetal_indices, element_pools=self.nonmetal_element_pool)
                        if operator.can_interswap(parents):
                            offsprings, desc = operator.get_new_individual(parents)
                        else:
                            # interswap impossible with the current element_pools/allowed_indices -> do two intraswaps
                            warnings.warn('EMCInterSwap failed, fall back to double intraswaps.')
                            intra1 = EMCIntraSwap(allowed_indices=operator.allowed_indices, element_pools=operator.element_pools)
                            offspring1, _ = intra1.get_new_individual([parents[0]])
                            intra2 = EMCIntraSwap(allowed_indices=operator.allowed_indices, element_pools=operator.element_pools)
                            offspring2, _ = intra2.get_new_individual([parents[1]])
                            offsprings = [offspring1, offspring2]
                            desc = 'Double IntraSwap: Parents {0} {1}'.format(parents[0].info['confid'], parents[1].info['confid'])
                    for idx in range(2):
                        rep = parents[idx].info['key_value_pairs'][self.replica_key]
                        if os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
                            continue
                        if offsprings[idx] == parents[idx]:
                            warnings.warn(f'Offspring identical to parent. Something with replica {rep} went wrong')
                        offsprings[idx].info['key_value_pairs'][self.temperature_key] = parents[idx].info['key_value_pairs'][self.temperature_key]
                        offsprings[idx].info['key_value_pairs'][self.replica_key] = rep
                        write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj', offsprings[idx])

                elif len(parents) == 1:
                    rep = parents[0].info['key_value_pairs'][self.replica_key]
                    if os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
                        continue
                    if self.nonmetal_element_pool is None:
                        allowed_indices = self.mutable_metal_indices
                        element_pools = self.metal_element_pool
                    else:
                        if (self.mutable_metal_indices is None) or (self.mutable_nonmetal_indices is None):
                            allowed_indices = None
                        else:
                            allowed_indices = [self.mutable_metal_indices, self.mutable_nonmetal_indices]
                        element_pools = [self.metal_element_pool, self.nonmetal_element_pool]
                    operator = EMCIntraSwap(allowed_indices=allowed_indices, element_pools=element_pools)
                    offspring, desc = operator.get_new_individual(parents)
                    if offspring == parents[0]:
                        warnings.warn(f'Offspring identical to parent. Something with replica {rep} went wrong')
                    offspring.info['key_value_pairs'][self.temperature_key] = parents[0].info['key_value_pairs'][self.temperature_key]
                    offspring.info['key_value_pairs'][self.replica_key] = rep
                    write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj', offspring)

            replicas = sorted([c.info['key_value_pairs'][self.replica_key] for lst in all_parents for c in lst])
            unrelaxed_offsprings = [read(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj') for rep in replicas]
            relaxed_offsprings = self.get_relaxed_candidates(unrelaxed_offsprings, run_md=False, batching=self.batching)
            relaxed_candidates = []
            for parents in all_parents:
                if len(parents) == 2:
                    rep1 = parents[0].info['key_value_pairs'][self.replica_key]
                    rep2 = parents[1].info['key_value_pairs'][self.replica_key]
                    relaxed_offspring1 = relaxed_offsprings[replicas.index(rep1)]
                    relaxed_offspring2 = relaxed_offsprings[replicas.index(rep2)] 
                    relaxed_offspring1.info['data']['operation'] = parents[0].info['data']['for_operation']
                    relaxed_offspring2.info['data']['operation'] = parents[1].info['data']['for_operation']
                    relaxed_offspring1.info['key_value_pairs']['emc_step'] = gen_num + gi
                    relaxed_offspring2.info['key_value_pairs']['emc_step'] = gen_num + gi
                    if parents[0].info['data']['for_operation'] == 'crossover':
                        relaxed_candidates += self.pop.evaluate_offsprings(
                                parents, [relaxed_offspring1, relaxed_offspring2], 
                                pressure=self.pressure, raw_score_cutoff=self.raw_score_cutoff)
                    elif parents[0].info['data']['for_operation'] == 'interswap':
                        if (self.bias_starting_step is not None) and (gen_num + gi >= self.bias_starting_step):
                            prefactor = 'gc' 
                        else:
                            prefactor = 1
                        relaxed_candidates += self.pop.evaluate_offsprings(
                                parents, [relaxed_offspring1, relaxed_offspring2], 
                                pressure=self.pressure, prefactor=prefactor,
                                raw_score_cutoff=self.raw_score_cutoff)
                elif len(parents) == 1:
                    rep = parents[0].info['key_value_pairs'][self.replica_key]
                    relaxed_offspring = relaxed_offsprings[replicas.index(rep)]
                    relaxed_offspring.info['data']['operation'] = parents[0].info['data']['for_operation']
                    relaxed_offspring.info['key_value_pairs']['emc_step'] = gen_num + gi
                    relaxed_candidates += self.pop.evaluate_offsprings(
                            parents, [relaxed_offspring], pressure=self.pressure,
                            raw_score_cutoff=self.raw_score_cutoff)

            # Update the population to allow new candidates to enter
            self.dc.add_more_relaxed_candidates(relaxed_candidates)
            self.pop.update()
            if (self.exchange_generation is not None) and (self.exchange_generation > 0):
                if (gen_num + gi) % self.exchange_generation == 0:
                    self.pop.exchange(keep_composition=True)
            
            for atoms in self.pop.pop:
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = Trajectory(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', 'a')
                traj.write(atoms)

            gi += 1
