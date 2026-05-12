#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Algorithm for canonical evolutionary Monte Carlo (EMC) run"""
from .emc_population import EMCPopulation
from .emc_operators import (
        EMCCutSpliceCrossover,
        EMCIntraSwap)
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.db import connect
from ase.ga.data import DataConnection, PrepareDB
from ase.parallel import parprint
import numpy as np
import subprocess
import warnings
import random
import shutil
import time
import sys
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
            gpu_devices=1,
            procs_per_device=10,
            batching=False,
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
        self.mutable_metal_indices = mutable_metal_indices
        self.mutable_nonmetal_indices = mutable_nonmetal_indices
        self.raw_score_cutoff = raw_score_cutoff
        self.temperature_key = temperature_key
        self.replica_key = replica_key
        self.acceptance_key = acceptance_key
        self.category_key = category_key
        self.gpu_devices = gpu_devices
        self.procs_per_device = procs_per_device
        self.batching = batching
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
                category_key=self.category_key,
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

    def connect_data_with_trim(self, starting_step):

        asedb_path = self.calculation_dir + '/{}.db'.format(self.name)
        db = connect(asedb_path)
        relaxed_images = []
        unrelaxed_iamges = []
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

    def get_relaxed_candidates(self, unrelaxed_candidates, run_md=False, add_args=None, batching=False):

        replicas = [c.info['key_value_pairs'][self.replica_key] for c in unrelaxed_candidates]
        if self.fix_indices is None:
            fix_indices_str = 'None'
        else:
            fix_indices_str = ','.join(map(str, self.fix_indices))
        if self.gpu_devices is None:
            gpu_devices = [0]
        else:
            gpu_devices = self.gpu_devices
        batch_images = []
        batch_start_index = None
        gpu_device = gpu_devices[0]
        procs = []
        for i, atoms in enumerate(unrelaxed_candidates):
            rep = atoms.info['key_value_pairs'][self.replica_key]
            if os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
                continue
            if batching:
                # Append current atoms and decide whether to flush the batch.
                device_for_current = gpu_devices[(i // self.procs_per_device) % len(gpu_devices)]
                if not batch_images:     
                    batch_start_index = i
                batch_images.append(atoms)
                is_last = (i == len(unrelaxed_candidates) - 1)
                device_for_next = None
                if not is_last:
                    device_for_next = gpu_devices[((i + 1) // self.procs_per_device) % len(gpu_devices)]
                if (len(batch_images) >= self.procs_per_device) or (device_for_next is not None and device_for_next != device_for_current) or is_last:
                    reps_str = '_'.join([str(a.info['key_value_pairs'][self.replica_key]) for a in batch_images])
                    write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{reps_str}.traj', batch_images)
                    arg_str = f'{self.calculation_dir}/tmp_{self.name}/atoms_{reps_str}.traj {int(run_md)} {device_for_current} {fix_indices_str}'
                    if add_args is not None:
                        for arg in add_args:
                            arg_str += f' {arg}'
                    # create a log file per job and launch with Popen so we can poll exit codes
                    logpath = f'{self.calculation_dir}/tmp_{self.name}/job_{reps_str if "reps_str" in locals() else rep}.log'
                    cmd = f'python {self.calculation_dir}/{self.calculation_script} {arg_str}'
                    p = subprocess.Popen(cmd, shell=True, stdout=open(logpath, 'ab'), stderr=subprocess.STDOUT, cwd=self.calculation_dir)
                    procs.append({'proc': p, 'reps': (reps_list if 'reps_list' in locals() else [rep]), 'log': logpath})
                    #os.system(f'nohup python {self.calculation_dir}/{self.calculation_script} ' + arg_str + ' &')
                    # reset batch_images for next batch
                    batch_images = []
                    batch_start_index = None 
            else:
                gpu_device = gpu_devices[(i//self.procs_per_device)%len(self.gpu_devices)]
                write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj', atoms)
                arg_str = f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj {int(run_md)} {gpu_device} {fix_indices_str}'
                if add_args is not None:
                    for arg in add_args:
                        arg_str += f' {arg}'
                # create a log file per job and launch with Popen so we can poll exit codes
                logpath = f'{self.calculation_dir}/tmp_{self.name}/job_{reps_str if "reps_str" in locals() else rep}.log'
                cmd = f'python {self.calculation_dir}/{self.calculation_script} {arg_str}'
                p = subprocess.Popen(cmd, shell=True, stdout=open(logpath, 'ab'), stderr=subprocess.STDOUT, cwd=self.calculation_dir)
                procs.append({'proc': p, 'reps': (reps_list if 'reps_list' in locals() else [rep]), 'log': logpath})
                #os.system(f'nohup python {self.calculation_dir}/{self.calculation_script} ' + arg_str + ' &')
            if (i % (len(gpu_devices)*self.procs_per_device) == (len(gpu_devices)*self.procs_per_device-1)
            ) or (i == len(unrelaxed_candidates) - 1):
                # wait until all expected relaxed files for this chunk exist, but also
                # check subprocess exit codes and fail fast if any returned non-zero.
                while True:
                    # check whether all expected output files exist for the chunk
                    files_ok = all(os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{r}.traj')
                                   for entry in procs for r in entry['reps'])
                    # check processes for failures
                    for entry in procs:
                        ret = entry['proc'].poll()
                        if ret is not None and ret != 0:
                            raise RuntimeError(f'Parallel job for replicas {entry["reps"]} failed with exit code {ret}. See log {entry["log"]}')
                    if files_ok:
                        break
                    time.sleep(5)
                # clear procs for the next chunk
                procs = []

        # flush any remaining partial batch that wasn't submitted
        if batching and batch_images:
            # decide device based on batch_start_index (the index where this batch started)
            device_for_current = gpu_devices[(batch_start_index // self.procs_per_device) % len(gpu_devices)]
            reps_str = '_'.join([str(a.info['key_value_pairs'][self.replica_key]) for a in batch_images])
            write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{reps_str}.traj', batch_images)
            arg_str = f'{self.calculation_dir}/tmp_{self.name}/atoms_{reps_str}.traj {int(run_md)} {device_for_current} {fix_indices_str}'
            if add_args is not None:
                for arg in add_args:
                    arg_str += f' {arg}'
            # create a log file per job and launch with Popen so we can poll exit codes
            logpath = f'{self.calculation_dir}/tmp_{self.name}/job_{reps_str if "reps_str" in locals() else rep}.log'
            cmd = f'python {self.calculation_dir}/{self.calculation_script} {arg_str}'
            p = subprocess.Popen(cmd, shell=True, stdout=open(logpath, 'ab'), stderr=subprocess.STDOUT, cwd=self.calculation_dir)
            procs.append({'proc': p, 'reps': (reps_list if 'reps_list' in locals() else [rep]), 'log': logpath})
            #os.system(f'nohup python {self.calculation_dir}/{self.calculation_script} ' + arg_str + ' &')
            batch_images = []
            batch_start_index = None

        relaxed_candidates = [read(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj', index=-1) for rep in replicas]
        for rep in replicas:
            os.remove(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj')
            if os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxation_{rep}.traj'):
                os.remove(f'{self.calculation_dir}/tmp_{self.name}/relaxation_{rep}.traj')

        return relaxed_candidates

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
        if starting_step is None:
            if os.path.isfile(f'{self.calculation_dir}/tmp_{self.name}/emc_0.traj'):
                atoms0 = read(f'{self.calculation_dir}/tmp_{self.name}/emc_0.traj', index=-1)
                starting_step = atoms0.info['key_value_pairs']['emc_step'] + 1
            else:
                starting_step = 1
        # Connect to the ase db
        asedb_path = self.calculation_dir + '/{}.db'.format(self.name)
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
                relaxed_candidates = self.get_relaxed_candidates(unrelaxed_candidates, run_md=True, batching=self.batching)
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
                        if os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
                            continue
                        if offsprings[idx] == parents[idx]:
                            warnings.warn(f'Offspring identical to parent. Something with replica {rep} went wrong')
                        offsprings[idx].info['key_value_pairs'][self.temperature_key] = parents[idx].info['key_value_pairs'][self.temperature_key]
                        offsprings[idx].info['key_value_pairs'][self.replica_key] = rep
                        if self.category_key is not None:
                            offsprings[idx].info['key_value_pairs'][self.category_key] = \
                                    parents[idx].info['key_value_pairs'][self.category_key]
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
                    if self.category_key is not None:
                        offspring.info['key_value_pairs'][self.category_key] = \
                                parents[0].info['key_value_pairs'][self.category_key]
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
            if (self.exchange_generation is not None) and (self.exchange_generation > 0):
                if (gen_num + gi) % self.exchange_generation == 0:
                    self.pop.exchange(keep_composition=True)
            
            for atoms in self.pop.pop:
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = Trajectory(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', 'a')
                traj.write(atoms)

            gi += 1
