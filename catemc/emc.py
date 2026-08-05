#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Algorithm for NVT/NPT evolutionary Monte Carlo (EMC) run"""
from .emc_population import EMCPopulation
from .emc_logger import EMCLogger
from .worker_pool import WorkerPool
from .calculation_templates import calculate_bulk_fitness
from .emc_operators import (
        EMCCutSpliceCrossover,
        EMCIntraSwap)
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.db import connect
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
    Main algorithm for standard NVT or NPT evolutionary Monte Carlo (EMC).

    Parameters
    ----------
    name: str
        Name of the target system.

    calculation_dir: str
        Path of the calculation directory.

    input_pop: list of ase.Atoms objects
        Input population for the EMC run.

    fix_indices: list of ints, default None
        A list of atom indices that should be immutable and fixed in positions
        throughout the EMC run. Nothing is fixed by default.

    trim_generation: int, default 100
        To trim database (with backup) every N steps. The database file would be
        too large and adding new data would be too slow without trimming.

    pressure: float, default None
        The common pressure target to perform NPT EMC MD and cell relaxation.
        If this is not set, run NVT EMC without changing cell volumes.

    md_generation: int, default 10
        To perform molecular dynamics every N steps.

    exchange_generation: int, defatul 1
        To perform replica exchange every N steps.

    crossover_rate: float, default 0.
        The rate of performing crossover operations.

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

    category_key: str, default None
        Keyword for a category that is specific for each replica and should be
        kept along with the replica number throughout the simulation.

    gpu_devices: list of ints, default None
        The list of GPU devices to use. Default is to use a single GPU/CPU device.

    procs_per_device: int, default 1
        Number of persistent workers assigned to each device. Each worker creates
        its own calculator sequence once and reuses it throughout the run.

    calculator_factory: callable
        Top-level function returning one calculator or a sequence of calculators.
        Each persistent worker calls this function once, then reuses the returned
        calculators for the complete simulation. A sequence is applied in order.

    calculator_factory_kwargs: dict, default None
        Keyword arguments passed to ``calculator_factory`` in every worker.

    fitness_function: callable, default None
        Function used to evaluate one structure. The bulk fitness template is used
        by default.

    fitness_kwargs: dict, default None
        Calculator, relaxation and MD keyword arguments passed to the fitness
        function. Parameters already defined by the EMC class are supplied
        automatically.

    worker_startup_timeout: float, default 300
        Maximum time in seconds to initialize all workers and calculators.

    worker_result_timeout: float, default 3600
        Maximum time in seconds to wait for the next completed evaluation.

    scratch_dir: str, default None
        Optional node-local directory for the active ASE database. Trimmed backup
        databases are copied to ``calculation_dir``.

    scratch_sync_interval: int, default None
        Copy the active scratch database to ``calculation_dir`` every N
        generations. ``None`` leaves synchronization to database trimming and
        the final synchronization.

    traj_write_interval: int, default 1
        Write one population frame per replica every N generations. ``None`` or
        a non-positive value disables trajectory writing.

    """

    def __init__(self, name,
            calculation_dir,
            input_pop,
            calculator_factory,
            fix_indices=None,
            trim_generation=100,
            pressure=None,
            md_generation=10,
            exchange_generation=1,
            crossover_rate=0.,
            intraswap_rate=0.,
            metal_element_pool=None,
            nonmetal_element_pool=None,
            mutable_metal_indices=None,
            mutable_nonmetal_indices=None,
            raw_score_cutoff=-10000.,
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted',
            category_key=None,
            gpu_devices=None,
            procs_per_device=1,
            verbose=True,
            calculator_factory_kwargs=None,
            fitness_function=None,
            fitness_kwargs=None,
            worker_startup_timeout=300.,
            worker_result_timeout=3600.,
            scratch_dir=None,
            scratch_sync_interval=None,
            traj_write_interval=1):

        self.name = name
        self.calculation_dir = calculation_dir
        self.input_pop = input_pop
        self.fix_indices = fix_indices
        self.trim_generation = trim_generation
        self.pressure = pressure
        self.md_generation = md_generation
        self.exchange_generation = exchange_generation
        self.crossover_rate = crossover_rate
        self.intraswap_rate = intraswap_rate
        self.metal_element_pool = metal_element_pool
        self.nonmetal_element_pool = nonmetal_element_pool
        self.mutable_metal_indices = mutable_metal_indices
        self.mutable_nonmetal_indices = mutable_nonmetal_indices
        self.raw_score_cutoff = raw_score_cutoff
        self.temperature_key = temperature_key
        self.replica_key = replica_key
        self.acceptance_key = acceptance_key
        self.category_key = category_key
        if not callable(calculator_factory):
            raise TypeError('calculator_factory must be callable')
        if fitness_function is not None and not callable(fitness_function):
            raise TypeError('fitness_function must be callable')
        if procs_per_device < 1:
            raise ValueError('procs_per_device must be at least 1')
        self.gpu_devices = ([None] if gpu_devices is None else
                            ([gpu_devices] if isinstance(gpu_devices, int)
                             else list(gpu_devices)))
        if len(self.gpu_devices) == 0:
            raise ValueError('gpu_devices cannot be empty')
        self.procs_per_device = procs_per_device
        self.verbose = verbose
        self.calculator_factory = calculator_factory
        self.calculator_factory_kwargs = ({} if calculator_factory_kwargs is None
                                          else dict(calculator_factory_kwargs))
        self.fitness_function = calculate_bulk_fitness if fitness_function is None else fitness_function
        self.fitness_kwargs = {} if fitness_kwargs is None else dict(fitness_kwargs)
        self.worker_startup_timeout = worker_startup_timeout
        self.worker_result_timeout = worker_result_timeout
        self.scratch_dir = scratch_dir
        self.scratch_sync_interval = scratch_sync_interval
        self.traj_write_interval = traj_write_interval
        self.worker_pool = None
        self.asedb_path = None
        self.persistent_asedb_path = None


    def uses_builtin_fitness(self):

        return self.fitness_function is calculate_bulk_fitness

    def get_fitness_context(self):

        return {
            'target_metal_species': [m for m in (self.metal_element_pool or [])
                                     if m != 'X'],
            'pressure': self.pressure,
            'fix_indices': self.fix_indices,
            'raw_score_cutoff': self.raw_score_cutoff,
            'temperature_key': self.temperature_key,
            'replica_key': self.replica_key,
            'work_dir': f'{self.calculation_dir}/tmp_{self.name}',
        }

    def get_reserved_fitness_kwargs(self):

        return {
            'target_metal_species', 'pressure', 'fix_indices',
            'raw_score_cutoff', 'temperature_key', 'replica_key',
            'work_dir', '_emc_context',
        }

    def make_worker_pool(self):

        n_workers = len(self.gpu_devices) * self.procs_per_device
        fitness_kwargs = dict(self.fitness_kwargs)
        if self.uses_builtin_fitness():
            duplicated = self.get_reserved_fitness_kwargs() & fitness_kwargs.keys()
            if duplicated:
                raise ValueError(
                    'Remove EMC-owned entries from fitness_kwargs: {}'.format(
                        ', '.join(sorted(duplicated))))
            fitness_kwargs['_emc_context'] = self.get_fitness_context()
        else:
            fitness_kwargs.setdefault('fix_indices', self.fix_indices)
            fitness_kwargs.setdefault(
                'work_dir', f'{self.calculation_dir}/tmp_{self.name}')
        return WorkerPool(
                calculator_factory=self.calculator_factory,
                fitness_function=self.fitness_function,
                calculator_factory_kwargs=self.calculator_factory_kwargs,
                fitness_kwargs=fitness_kwargs,
                num_workers=n_workers,
                gpu_devices=self.gpu_devices,
                startup_timeout=self.worker_startup_timeout,
                result_timeout=self.worker_result_timeout)

    def set_asedb_path(self):

        self.persistent_asedb_path = self.calculation_dir + '/{}.db'.format(self.name)
        if self.scratch_dir is None:
            self.asedb_path = self.persistent_asedb_path
            if not os.path.isfile(self.asedb_path):
                backup_path = self.get_latest_working_backup_path()
                if backup_path is not None:
                    self.restore_asedb_from_backup(backup_path, self.asedb_path)
            return
        os.makedirs(self.scratch_dir, exist_ok=True)
        self.asedb_path = os.path.join(self.scratch_dir, '{}.db'.format(self.name))
        if os.path.isfile(self.asedb_path):
            parprint('Using existing scratch ASE database: {}'.format(self.asedb_path))
        elif os.path.isfile(self.persistent_asedb_path):
            self.copy_asedb(self.persistent_asedb_path, self.asedb_path)
        else:
            backup_path = self.get_latest_working_backup_path()
            if backup_path is not None:
                self.restore_asedb_from_backup(backup_path, self.asedb_path)
                self.copy_asedb(self.asedb_path, self.persistent_asedb_path)
        parprint('Scratch ASE database: {}'.format(self.asedb_path))
        parprint('Persistent ASE database mirror: {}'.format(self.persistent_asedb_path))

    def copy_asedb(self, source, target):

        target_dir = os.path.dirname(os.path.abspath(target))
        os.makedirs(target_dir, exist_ok=True)
        shutil.copyfile(source, target)

    def get_backup_index(self, path):

        name = os.path.basename(path)
        prefix = '{}_bak'.format(self.name)
        suffix = '.db'
        if not name.startswith(prefix):
            return None
        if not name.endswith(suffix):
            return None
        index = name[len(prefix):-len(suffix)]
        if not index.isdigit():
            return None
        return int(index)

    def get_latest_working_backup_path(self):

        backup_paths = []
        if not os.path.isdir(self.calculation_dir):
            return None
        for name in os.listdir(self.calculation_dir):
            path = os.path.join(self.calculation_dir, name)
            index = self.get_backup_index(path)
            if index is not None and os.path.isfile(path):
                backup_paths.append((index, path))
        if not backup_paths:
            return None
        backup_paths.sort(key=lambda x: x[0])
        return backup_paths[-1][1]

    def get_asedb_history_paths(self):

        backup_paths = {}
        directories = [self.calculation_dir]
        if self.scratch_dir is not None:
            directories.append(self.scratch_dir)
        for directory in directories:
            if not os.path.isdir(directory):
                continue
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                index = self.get_backup_index(path)
                if index is not None and os.path.isfile(path):
                    backup_paths[index] = path
        paths = [backup_paths[index] for index in sorted(backup_paths)]
        active_path = self.asedb_path
        if active_path is None or not os.path.isfile(active_path):
            active_path = self.persistent_asedb_path
        if active_path is not None and os.path.isfile(active_path):
            if os.path.abspath(active_path) not in {os.path.abspath(path) for path in paths}:
                paths.append(active_path)
        return paths

    def restore_asedb_from_backup(self, backup_path, target_path):

        backup_dc = DataConnection(backup_path)
        backup_pop = EMCPopulation(
                data_connection=backup_dc,
                population_size=self.pop_size,
                logfile=None,
                temperature_key=self.temperature_key,
                replica_key=self.replica_key,
                acceptance_key=self.acceptance_key,
                category_key=self.category_key,
                verbose=False)
        backup_pop.update()
        current_pop = backup_pop.get_current_population()
        if len(current_pop) != self.pop_size:
            raise RuntimeError('Could not recover a complete population from {}'.format(backup_path))
        target_dir = os.path.dirname(os.path.abspath(target_path))
        os.makedirs(target_dir, exist_ok=True)
        if os.path.isfile(target_path):
            os.remove(target_path)
        PrepareDB(target_path, population_size=self.pop_size)
        dc = DataConnection(target_path)
        dc.add_more_relaxed_candidates(current_pop)
        latest_generation = self.get_latest_asedb_generation(target_path)
        parprint('Recovered generation {} from {}'.format(latest_generation, backup_path))

    def get_latest_asedb_generation(self, asedb_path):

        if not os.path.isfile(asedb_path):
            return None
        db = connect(asedb_path)
        generations = []
        for row in db.select('relaxed=1'):
            key_value_pairs = row.key_value_pairs
            if 'emc_step' in key_value_pairs:
                generations.append(key_value_pairs['emc_step'])
        if not generations:
            return None
        return max(generations)

    def get_starting_step(self, starting_step):

        if starting_step is not None:
            return starting_step
        latest_generation = self.get_latest_asedb_generation(self.asedb_path)
        if latest_generation is not None:
            return latest_generation + 1
        if os.path.isfile(f'{self.calculation_dir}/tmp_{self.name}/emc_0.traj'):
            atoms0 = read(f'{self.calculation_dir}/tmp_{self.name}/emc_0.traj', index=-1)
            return atoms0.info['key_value_pairs']['emc_step'] + 1
        return 1

    def sync_asedb(self):

        if self.scratch_dir is not None and os.path.isfile(self.asedb_path):
            self.copy_asedb(self.asedb_path, self.persistent_asedb_path)

    def sync_asedb_if_needed(self, generation):

        if self.scratch_sync_interval is None:
            return
        if self.scratch_sync_interval <= 0:
            return
        if generation % self.scratch_sync_interval == 0:
            self.sync_asedb()

    def get_persistent_backup_path(self, scratch_backup_path):

        backup_name = os.path.basename(scratch_backup_path)
        return os.path.join(self.calculation_dir, backup_name)

    def sync_backup_asedb(self, scratch_backup_path):

        if self.scratch_dir is not None and os.path.isfile(scratch_backup_path):
            persistent_backup_path = self.get_persistent_backup_path(scratch_backup_path)
            self.copy_asedb(scratch_backup_path, persistent_backup_path)

    def write_population_traj(self, generation):

        if self.traj_write_interval is None:
            return
        if self.traj_write_interval <= 0:
            return
        if generation % self.traj_write_interval != 0:
            return
        for atoms in self.pop.pop:
            rep = atoms.info['key_value_pairs'][self.replica_key]
            traj = Trajectory(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', 'a')
            traj.write(atoms)
            traj.close()

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
                category_key=self.category_key,
                verbose=self.verbose)

        return pop

    def trim_asedb(self):

        asedb_path = self.asedb_path
        tmp_asedb_path = asedb_path[:-3] + '_tmp.db'
        tmp_asedb = PrepareDB(tmp_asedb_path, population_size=self.pop_size)
        current_pop = self.pop.get_current_population()
        i = 0
        backup_path = asedb_path[:-3] + '_bak{}.db'.format(i)
        persistent_backup_path = self.get_persistent_backup_path(backup_path)
        while os.path.isfile(backup_path) or os.path.isfile(persistent_backup_path):
            i += 1
            backup_path = asedb_path[:-3] + '_bak{}.db'.format(i)
            persistent_backup_path = self.get_persistent_backup_path(backup_path)
        self.copy_asedb(asedb_path, backup_path)
        self.sync_backup_asedb(backup_path)
        os.remove(asedb_path)
        shutil.move(tmp_asedb_path, asedb_path)
        self.dc = DataConnection(asedb_path)
        self.dc.add_more_relaxed_candidates(current_pop)
        self.sync_asedb()

    def connect_data_with_trim(self, starting_step):

        asedb_path = self.asedb_path
        db = connect(asedb_path)
        relaxed_images = []
        unrelaxed_images = []
        for i, row in enumerate(db.select('relaxed<=1')):
            atoms = row.toatoms()
            atoms.info['key_value_pairs'] = row.key_value_pairs
            atoms.info['data'] = row.data
            if row.emc_step < starting_step:
                if row.relaxed:
                    relaxed_images.append(atoms)
                else:
                    unrelaxed_images.append(atoms)

        os.remove(asedb_path)
        asedb = PrepareDB(asedb_path, population_size=self.pop_size)
        for atoms in unrelaxed_images:
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
        self.dc = DataConnection(asedb_path)
        self.dc.add_more_relaxed_candidates(relaxed_images)

    def get_relaxed_candidates(self, unrelaxed_candidates, run_md=False):

        return self.worker_pool.evaluate(
                unrelaxed_candidates,
                run_md=run_md)

    def run(self, starting_step=None, total_steps=10000, restart_with_trim=False):

        """                                    
        Parameters
        ----------
        starting_step: int, default None 
            (Re)starting from a certain EMC step. Default None is to
            restart from the latest step assuming tmp_* folder exists.

        total_steps: int, default 10000
            The total number of EMC steps that the algorithm should run.

        restart_with_trim: bool, default False
            Whether to restart while trimming all data generated after
            the starting_step that has been generated before.

        """

        restart = True
        self.pop_size = len(self.input_pop)
        self.set_asedb_path()
        os.makedirs(f'{self.calculation_dir}/tmp_{self.name}', exist_ok=True)
        self.worker_pool = self.make_worker_pool()
        logger = EMCLogger(verbose=self.verbose, acceptance_key=self.acceptance_key)
        starting_step = self.get_starting_step(starting_step)
        # Connect to the ase db
        asedb_path = self.asedb_path
        if not os.path.isfile(asedb_path):
            restart = False
            asedb = PrepareDB(asedb_path, population_size=self.pop_size)
            for atoms in self.input_pop:
                assert 'key_value_pairs' in atoms.info
                if 'data' not in atoms.info:
                    atoms.info['data'] = {}
                kwargs = {self.temperature_key: atoms.info['key_value_pairs'][self.temperature_key], 
                          self.replica_key: atoms.info['key_value_pairs'][self.replica_key],
                          self.acceptance_key: 0}
                if self.category_key is not None:
                    kwargs[self.category_key] = atoms.info['key_value_pairs'][self.category_key]
                asedb.add_unrelaxed_candidate(atoms, 
                        emc_step=atoms.info['key_value_pairs']['emc_step'],
                        data=atoms.info['data'],
                        **kwargs)
        if restart_with_trim:
            self.connect_data_with_trim(starting_step=starting_step)
        else:
            self.dc = DataConnection(asedb_path)
        if not restart:
            unrelaxed_candidates = self.dc.get_all_unrelaxed_candidates()
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
            self.write_population_traj(0)
        self.sync_asedb()
        if restart_with_trim:
            for atoms in self.pop.pop:
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = read(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', index=f':{starting_step}')
                write(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', traj)
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
            unrelaxed_dict = {}
            for parents in all_parents:
                if len(parents) == 2:
                    if self.nonmetal_element_pool is None:
                        element_pool = self.metal_element_pool
                        allowed_indices = self.mutable_metal_indices
                    else:
                        element_pool = self.metal_element_pool + self.nonmetal_element_pool
                        if (self.mutable_metal_indices is None) or (self.mutable_nonmetal_indices is None):
                            allowed_indices = None
                        else:
                            allowed_indices = self.mutable_metal_indices + self.mutable_nonmetal_indices
                    operator = EMCCutSpliceCrossover(allowed_indices=allowed_indices, elements=element_pool, keep_composition=True)
                    offsprings, desc = operator.get_new_individual(parents)
                    for idx in range(2):
                        rep = parents[idx].info['key_value_pairs'][self.replica_key]
                        if offsprings[idx] == parents[idx]:
                            warnings.warn(f'Offspring identical to parent. Something with replica {rep} went wrong')
                        offsprings[idx].info['key_value_pairs'][self.temperature_key] = parents[idx].info['key_value_pairs'][self.temperature_key]
                        offsprings[idx].info['key_value_pairs'][self.replica_key] = rep
                        if self.category_key is not None:
                            offsprings[idx].info['key_value_pairs'][self.category_key] = \
                                    parents[idx].info['key_value_pairs'][self.category_key]
                        unrelaxed_dict[rep] = offsprings[idx]

                elif len(parents) == 1:
                    rep = parents[0].info['key_value_pairs'][self.replica_key]
                    do_metal = False
                    if self.nonmetal_element_pool is None:
                        do_metal = True
                    elif len([s for s in set(parents[0].symbols) if s in self.nonmetal_element_pool]) < 2:
                        do_metal = True
                    if do_metal:
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
                    if self.category_key is not None:
                        offspring.info['key_value_pairs'][self.category_key] = \
                                parents[0].info['key_value_pairs'][self.category_key]
                    unrelaxed_dict[rep] = offspring

            replicas = sorted([c.info['key_value_pairs'][self.replica_key] for lst in all_parents for c in lst])
            unrelaxed_offsprings = [unrelaxed_dict[rep] for rep in replicas]
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
                            pressure=self.pressure, raw_score_cutoff=self.raw_score_cutoff)
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
            logger.generation(gen_num + gi, relaxed_candidates, self.pop.pop)
            
            self.write_population_traj(gen_num + gi)
            self.sync_asedb_if_needed(gen_num + gi)

            gi += 1

        self.sync_asedb()
        self.worker_pool.close()
