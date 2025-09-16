#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Algorithm for canonical evolutionary Monte Carlo (EMC) run"""
from .emc_population import EMCPopulation
from .emc_operators import (
        EMCCutSpliceCrossover,
        EMCIntraSwap)
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


class EMC(object):

    """
    Main algorithm for canonical evolutionary Monte Carlo (EMC).

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

    md_generation: int, default 10
        To perform molecular dynamics every N steps.

    exchange_generation: int, defatul 1
        To perform replica exchange every N steps.

    crossover_rate: float, default 0.
        The rate of performing crossover operations.

    intraswap_rate: float, default 0.
        The rate of performing intraswap operations in the catalyst slab.

    metal_element_pool: list of strs, default None
        The list of target metal elements that should be mutable during the EMC run.

    nonmetal_element_pool: list of strs, default None
        The list of target non-metal elements (including vacancy 'X') that should be 
        mutable during the EMC run.

    raw_score_cutoff: float, default -10000.
        The fitness score for candidates that are should not be considered in the sampling.
        This could be structures that are unphysical / failed to relax by the calculator.

    temperature_key: str, default 'temperature'
        Keyword for the temperature stored in each configuration.

    replica_key: str, default 'replica'
        Keyword for the replica number stored in each configuration.

    acceptance_key: str, default 'accepted'
        Keyword for indicating being accepted or not in each configuration

    gpu_devices: list of ints, default None
        The list of GPU devices to use. Default is to use a single GPU/CPU device.

    procs_per_device: int, default 1
        The number of processes to run in parallel on each GPU/CPU device. Defaul is to
        run in serial.

    """

    def __init__(self, name,
            calculation_dir,
            calculation_script,
            input_pop,
            fix_indices=None,
            trim_generation=100,
            md_generation=10,
            exchange_generation=1,
            crossover_rate=0.,
            intraswap_rate=0.,
            metal_element_pool=None,
            nonmetal_element_pool=None,
            raw_score_cutoff=-10000.,
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted',
            gpu_devices=1,
            procs_per_device=10,
            verbose=True):

        self.name = name
        self.calculation_dir = calculation_dir
        self.calculation_script = calculation_script
        self.input_pop = input_pop
        self.fix_indices = fix_indices
        self.trim_generation = trim_generation
        self.md_generation = md_generation
        self.exchange_generation = exchange_generation
        self.crossover_rate = crossover_rate
        self.intraswap_rate = intraswap_rate
        self.metal_element_pool = metal_element_pool
        self.nonmetal_element_pool = nonmetal_element_pool
        self.raw_score_cutoff = raw_score_cutoff
        self.temperature_key = temperature_key
        self.replica_key = replica_key
        self.acceptance_key = acceptance_key
        self.gpu_devices = gpu_devices
        self.procs_per_device = procs_per_device
        self.verbose = verbose

    def get_pop(self):

        # Define the logfile path
        log_path = self.calculation_dir + '/{}.log'.format(self.name)
        # Set the population
        pop = EMCPopulation(data_connection=self.dc,
                population_size=len(self.input_pop),
                logfile=log_path,
                temperature_key=self.temperature_key, 
                replica_key=self.replica_key,
                acceptance_key=self.acceptance_key,
                verbose=self.verbose)

        return pop

    def trim_asedb(self):

        asedb_path = self.calculation_dir + '/{}.db'.format(self.name)
        tmp_asedb = PrepareDB(asedb_path[:-3] + '_tmp.db', population_size=self.pop_size)
        current_pop = self.pop.get_current_population()
        i = 0
        while os.path.isfile(asedb_path[:-3] + '_bak{}.db'.format(i)):
            i += 1
        shutil.copy(asedb_path, asedb_path[:-3] + '_bak{}.db'.format(i))
        os.remove(asedb_path)
        shutil.move(asedb_path[:-3] + '_tmp.db', asedb_path)
        self.dc = DataConnection(asedb_path)
        self.dc.add_more_relaxed_candidates(current_pop)

    def get_relaxed_candidates(self, unrelaxed_candidates, run_md=False, add_args=None):

        replicas = [c.info['key_value_pairs'][self.replica_key] for c in unrelaxed_candidates]
        for i, atoms in enumerate(unrelaxed_candidates):
            rep = atoms.info['key_value_pairs'][self.replica_key]
            if os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
                continue
            write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj', atoms)
            if self.fix_indices is None:
                fix_indices_str = 'None'
            else:
                fix_indices_str = ','.join(map(str, self.fix_indices))
            if self.gpu_devices is None:
                gpu_devices = [0]
            else:
                gpu_devices = self.gpu_devices
            gpu_device = gpu_devices[(i//self.procs_per_device)%len(self.gpu_devices)]
            arg_str = f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj {int(run_md)} {gpu_device} {fix_indices_str}'
            if add_args is not None:
                for arg in add_args:
                    arg_str += f' {arg}'
            os.system(f'nohup python {self.calculation_dir}/{self.calculation_script} ' + arg_str + ' &')
            if (i % (len(gpu_devices)*self.procs_per_device) == (len(gpu_devices)*self.procs_per_device-1)
            ) or (i == len(unrelaxed_candidates) - 1):
                while not all(os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{replicas[j]}.traj') for j in range(i+1)):
                    time.sleep(5)
        relaxed_candidates = [read(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj', index=-1) for rep in replicas]
        for rep in replicas:
            os.remove(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj')
            os.remove(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj')

        return relaxed_candidates

    def run(self, starting_step=1, total_steps=10000):

        """                                    
        Parameters
        ----------
        starting_step: int, default 1
            (Re)starting from a certain EMC step.
                                               
        total_steps: int, default 10000
            The total number of EMC steps that the algorithm should run.

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
                if 'key_value_pairs' not in atoms.info:
                    atoms.info['data'] = {}
                kwargs = {self.temperature_key: atoms.info['key_value_pairs'][self.temperature_key], 
                          self.replica_key: atoms.info['key_value_pairs'][self.replica_key],
                          self.acceptance_key: 0}
                asedb.add_unrelaxed_candidate(atoms, 
                        emc_step=atoms.info['key_value_pairs']['emc_step'],
                        **kwargs)

        self.dc = DataConnection(asedb_path)
        unrelaxed_candidates = self.dc.get_all_unrelaxed_candidates()

        if not os.path.isdir(f'{self.calculation_dir}/tmp_{self.name}'):
            os.mkdir(f'{self.calculation_dir}/tmp_{self.name}')

        if not restart:
            relaxed_candidates = self.get_relaxed_candidates(unrelaxed_candidates, run_md=True)
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
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = Trajectory(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', 'a')
                traj.write(atoms)
        rates_sum = self.intraswap_rate + self.crossover_rate
        if rates_sum != 1:
            self.intraswap_rate /= rates_sum
            self.crossover_rate /= rates_sum

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
                relaxed_candidates = self.get_relaxed_candidates(unrelaxed_candidates, run_md=True)
                for atoms in relaxed_candidates:
                    atoms.info['key_value_pairs'][self.acceptance_key] = 1
                # Update the population to allow new candidates to enter
                self.dc.add_more_relaxed_candidates(relaxed_candidates) 
                self.pop.update()

            n_crossover_pairs = round((self.pop_size * self.crossover_rate) / 2)
            all_parents = self.pop.get_all_parents(n_crossover_pairs, n_interswap_pairs=0)
            for parents in all_parents:
                if len(parents) == 2:
                    if self.nonmetal_element_pool is None:
                        element_pool = self.metal_element_pool
                    else:
                        element_pool = self.metal_element_pool + self.nonmetal_element_pool
                    operator = EMCCutSpliceCrossover(elements=element_pool, keep_composition=True)
                    offsprings, desc = operator.get_new_individual(parents)
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
                        element_pools = self.metal_element_pool
                    else:
                        element_pools = [self.metal_element_pool, self.nonmetal_element_pool]
                    operator = EMCIntraSwap(element_pools=element_pools)
                    offspring, desc = operator.get_new_individual(parents)
                    if offspring == parents[0]:
                        warnings.warn(f'Offspring identical to parent. Something with replica {rep} went wrong')
                    offspring.info['key_value_pairs'][self.temperature_key] = parents[0].info['key_value_pairs'][self.temperature_key]
                    offspring.info['key_value_pairs'][self.replica_key] = rep
                    write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj', offspring)

            replicas = sorted([c.info['key_value_pairs'][self.replica_key] for lst in all_parents for c in lst])
            unrelaxed_offsprings = [read(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj') for rep in replicas]
            relaxed_offsprings = self.get_relaxed_candidates(unrelaxed_offsprings, run_md=False)
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
                    relaxed_candidates += self.pop.evaluate_offsprings(
                            parents, [relaxed_offspring1, relaxed_offspring2], 
                            raw_score_cutoff=self.raw_score_cutoff)
                elif len(parents) == 1:
                    rep = parents[0].info['key_value_pairs'][self.replica_key]
                    relaxed_offspring = relaxed_offsprings[replicas.index(rep)]
                    relaxed_offspring.info['data']['operation'] = parents[0].info['data']['for_operation']
                    relaxed_offspring.info['key_value_pairs']['emc_step'] = gen_num + gi
                    relaxed_candidates += self.pop.evaluate_offsprings(
                            parents, [relaxed_offspring], raw_score_cutoff=self.raw_score_cutoff)

            # Update the population to allow new candidates to enter
            self.dc.add_more_relaxed_candidates(relaxed_candidates)
            self.pop.update()
            if (gen_num + gi) % self.exchange_generation == 0:
                self.pop.exchange(keep_composition=True)
            
            for atoms in self.pop.pop:
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = Trajectory(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', 'a')
                traj.write(atoms)

            gi += 1
