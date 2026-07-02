#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Algorithm for NVT/NPT evolutionary Monte Carlo (EMC) run"""
from .emc_population import EMCPopulation
from .emc_logger import EMCLogger
from .worker_pool import WorkerPool
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
    Main algorithm for standard NVT or NPT evolutionary Monte Carlo (EMC).

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
            verbose=True,
            use_worker_pool=False,
            calculator_factory=None,
            calculator_factory_kwargs=None,
            fitness_kwargs=None,
            worker_startup_timeout=300.,
            worker_result_timeout=3600.,
            scratch_dir=None,
            scratch_sync_interval=None,
            traj_write_interval=1):

        self.name = name
        self.calculation_dir = calculation_dir
        self.calculation_script = calculation_script
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
        self.gpu_devices = [0] if gpu_devices is None else ([gpu_devices] if isinstance(gpu_devices, int) else gpu_devices)
        self.procs_per_device = procs_per_device
        self.batching = batching
        self.verbose = verbose
        self.use_worker_pool = use_worker_pool
        self.calculator_factory = calculator_factory
        self.calculator_factory_kwargs = {} if calculator_factory_kwargs is None else calculator_factory_kwargs
        self.fitness_kwargs = {} if fitness_kwargs is None else fitness_kwargs
        self.worker_startup_timeout = worker_startup_timeout
        self.worker_result_timeout = worker_result_timeout
        self.scratch_dir = scratch_dir
        self.scratch_sync_interval = scratch_sync_interval
        self.traj_write_interval = traj_write_interval
        self.worker_pool = None
        self.asedb_path = None
        self.persistent_asedb_path = None


    def get_fitness_function(self):

        return 'catemc.calculation_templates:calculate_bulk_fitness_inprocess'

    def get_calculator_factory(self):

        if self.calculator_factory is not None:
            return self.calculator_factory
        module_name = os.path.splitext(os.path.basename(self.calculation_script))[0]
        if self.calculation_dir not in sys.path:
            sys.path.insert(0, self.calculation_dir)
        return module_name + ':create_calculators'

    def get_worker_extra_kwargs(self):

        return {}

    def make_worker_pool(self):

        if not self.use_worker_pool:
            return None
        n_workers = len(self.gpu_devices) * self.procs_per_device
        fitness_kwargs = dict(self.fitness_kwargs)
        fitness_kwargs.update(self.get_worker_extra_kwargs())
        return WorkerPool(
                calculator_factory=self.get_calculator_factory(),
                fitness_function=self.get_fitness_function(),
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

    def get_relaxed_candidates(self, unrelaxed_candidates, run_md=False, add_args=None, batching=False):

        if self.worker_pool is not None:
            extra_kwargs = {'work_dir': f'{self.calculation_dir}/tmp_{self.name}'}
            return self.worker_pool.evaluate(unrelaxed_candidates, run_md=run_md,
                                             fix_indices=self.fix_indices,
                                             extra_kwargs=extra_kwargs)

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
            if (self.worker_pool is None) and os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
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
        self.set_asedb_path()
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
                relaxed_candidates = self.get_relaxed_candidates(unrelaxed_candidates, run_md=True, batching=self.batching)
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
                        if (self.worker_pool is None) and os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
                            continue
                        if offsprings[idx] == parents[idx]:
                            warnings.warn(f'Offspring identical to parent. Something with replica {rep} went wrong')
                        offsprings[idx].info['key_value_pairs'][self.temperature_key] = parents[idx].info['key_value_pairs'][self.temperature_key]
                        offsprings[idx].info['key_value_pairs'][self.replica_key] = rep
                        if self.category_key is not None:
                            offsprings[idx].info['key_value_pairs'][self.category_key] = \
                                    parents[idx].info['key_value_pairs'][self.category_key]
                        unrelaxed_dict[rep] = offsprings[idx]
                        if self.worker_pool is None:
                            write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj', offsprings[idx])

                elif len(parents) == 1:
                    rep = parents[0].info['key_value_pairs'][self.replica_key]
                    if (self.worker_pool is None) and os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
                        continue
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
                    if self.worker_pool is None:
                        write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj', offspring)

            replicas = sorted([c.info['key_value_pairs'][self.replica_key] for lst in all_parents for c in lst])
            if self.worker_pool is None:
                unrelaxed_offsprings = [read(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj') for rep in replicas]
            else:
                unrelaxed_offsprings = [unrelaxed_dict[rep] for rep in replicas]
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
        if self.worker_pool is not None:
            self.worker_pool.close()
