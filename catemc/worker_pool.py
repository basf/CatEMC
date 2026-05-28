#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persistent worker pool for CatEMC fitness evaluation.

Workers load the ML model once at startup and stay alive across generations,
eliminating the subprocess spawn + model reload overhead (~3s per call).

Usage:
    pool = WorkerPool(
        calculator_factory=my_calc_factory,
        num_workers=20,
        gpu_devices=[0],
    )
    results = pool.evaluate(atoms_list, run_md=False, fix_indices=None)
    pool.shutdown()
"""

import multiprocessing as mp
import traceback
import numpy as np
import os
import sys
import time
from ase import Atoms
from ase.io import read, write


# Sentinel to signal worker shutdown
_SHUTDOWN = "SHUTDOWN"


def _worker_loop(task_queue, result_queue, worker_id, gpu_device,
                 calculator_factory_module, calculator_factory_func,
                 calculator_factory_kwargs, fitness_func_module,
                 fitness_func_name, fitness_kwargs):
    """Main loop for a persistent worker process.
    
    Loads model once, then processes tasks from task_queue until shutdown.
    """
    try:
        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        
        # Import and call the calculator factory
        import importlib
        mod = importlib.import_module(calculator_factory_module)
        factory_fn = getattr(mod, calculator_factory_func)
        calculators = factory_fn(**calculator_factory_kwargs)
        
        # Import the fitness function
        fitness_mod = importlib.import_module(fitness_func_module)
        fitness_fn = getattr(fitness_mod, fitness_func_name)
        
        # Signal ready
        result_queue.put(("READY", worker_id, None))
        
        while True:
            task = task_queue.get()
            if task == _SHUTDOWN:
                break
            
            task_id, atoms_data, run_md, fix_indices, extra_kwargs = task
            try:
                # Reconstruct Atoms from serialized data
                atoms = atoms_data  # Already an Atoms object via mp
                
                # Merge fitness_kwargs with per-task overrides
                kwargs = dict(fitness_kwargs)
                kwargs['run_md'] = run_md
                kwargs['fix_indices'] = fix_indices
                if extra_kwargs:
                    kwargs.update(extra_kwargs)
                
                # Call the fitness function
                result_atoms = fitness_fn(
                    atoms,
                    calculators=calculators,
                    **kwargs
                )
                result_queue.put(("RESULT", task_id, result_atoms))
            except Exception as e:
                tb = traceback.format_exc()
                result_queue.put(("ERROR", task_id, f"{e}\n{tb}"))
    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put(("FATAL", worker_id, f"Worker {worker_id} died: {e}\n{tb}"))


class WorkerPool:
    """Pool of persistent worker processes for CatEMC fitness evaluation.
    
    Parameters
    ----------
    calculator_factory_module : str
        Python module containing the calculator factory function.
        
    calculator_factory_func : str
        Name of the factory function that returns calculator(s).
        
    calculator_factory_kwargs : dict
        Keyword arguments to pass to the factory function.
        
    fitness_func_module : str
        Module containing the fitness evaluation function.
        Default: 'catemc.calculation_templates'
        
    fitness_func_name : str
        Name of the fitness function.
        Default: 'calculate_bulk_fitness_inprocess'
        
    fitness_kwargs : dict
        Default keyword arguments for the fitness function.
        
    num_workers : int
        Number of worker processes to spawn.
        
    gpu_devices : list of int
        GPU device IDs to distribute workers across.
        
    startup_timeout : float
        Seconds to wait for each worker to signal ready.
    """
    
    def __init__(self,
                 calculator_factory_module,
                 calculator_factory_func,
                 calculator_factory_kwargs=None,
                 fitness_func_module='catemc.calculation_templates',
                 fitness_func_name='calculate_bulk_fitness_inprocess',
                 fitness_kwargs=None,
                 num_workers=10,
                 gpu_devices=None,
                 startup_timeout=120.0):
        
        self.num_workers = num_workers
        self.gpu_devices = gpu_devices or [0]
        self.calculator_factory_module = calculator_factory_module
        self.calculator_factory_func = calculator_factory_func
        self.calculator_factory_kwargs = calculator_factory_kwargs or {}
        self.fitness_func_module = fitness_func_module
        self.fitness_func_name = fitness_func_name
        self.fitness_kwargs = fitness_kwargs or {}
        
        # Use 'spawn' context to avoid CUDA fork issues
        ctx = mp.get_context('spawn')
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.workers = []
        
        _startup_start = time.time()
        # Spawn workers with staggered start to avoid CephFS metadata storm
        _stagger_delay = 0.5 if num_workers > 4 else 0.0
        for i in range(num_workers):
            gpu = self.gpu_devices[i % len(self.gpu_devices)]
            p = ctx.Process(
                target=_worker_loop,
                args=(
                    self.task_queue, self.result_queue, i, gpu,
                    calculator_factory_module, calculator_factory_func,
                    self.calculator_factory_kwargs, fitness_func_module,
                    fitness_func_name, self.fitness_kwargs,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
            if _stagger_delay and i < num_workers - 1:
                time.sleep(_stagger_delay)
        
        # Wait for all workers to signal ready
        ready_count = 0
        deadline = time.time() + startup_timeout
        while ready_count < num_workers:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Only {ready_count}/{num_workers} workers started "
                    f"within {startup_timeout}s timeout"
                )
            try:
                msg = self.result_queue.get(timeout=min(remaining, 10.0))
                if msg[0] == "READY":
                    ready_count += 1
                elif msg[0] == "FATAL":
                    raise RuntimeError(msg[2])
            except mp.queues.Empty:
                # Check if any worker died
                for w in self.workers:
                    if not w.is_alive():
                        raise RuntimeError(
                            f"Worker PID {w.pid} died during startup"
                        )
        
        print(f"WorkerPool: {num_workers} workers ready "
              f"(GPUs: {self.gpu_devices}, startup took {time.time() - _startup_start:.1f}s)")


    
    def evaluate(self, atoms_list, run_md=False, fix_indices=None, extra_kwargs=None):
        """Submit atoms for evaluation and collect results.
        
        Parameters
        ----------
        atoms_list : list of ase.Atoms
            Structures to evaluate. Each must have info['key_value_pairs']
            with 'replica' key.
            
        run_md : bool
            Whether to run MD on these structures.
            
        fix_indices : list of int or None
            Atom indices to fix.
            
        extra_kwargs : dict or None
            Additional kwargs to pass to fitness function.
            
        Returns
        -------
        list of ase.Atoms
            Relaxed/evaluated structures in same order as input.
        """
        n = len(atoms_list)
        if n == 0:
            return []
        
        # Submit all tasks
        for i, atoms in enumerate(atoms_list):
            task = (i, atoms, run_md, fix_indices, extra_kwargs)
            self.task_queue.put(task)
        
        # Collect results
        results = [None] * n
        errors = []
        collected = 0
        
        while collected < n:
            msg = self.result_queue.get(timeout=600.0)  # 10min timeout per task
            if msg[0] == "RESULT":
                task_id, result_atoms = msg[1], msg[2]
                results[task_id] = result_atoms
                collected += 1
            elif msg[0] == "ERROR":
                task_id, error_msg = msg[1], msg[2]
                errors.append((task_id, error_msg))
                collected += 1
            elif msg[0] == "FATAL":
                raise RuntimeError(f"Worker fatal error: {msg[2]}")
        
        if errors:
            error_strs = [f"  Task {tid}: {err}" for tid, err in errors]
            raise RuntimeError(
                f"{len(errors)}/{n} tasks failed:\n" + "\n".join(error_strs)
            )
        
        return results
    
    def shutdown(self):
        """Gracefully shut down all workers."""
        for _ in self.workers:
            self.task_queue.put(_SHUTDOWN)
        for w in self.workers:
            w.join(timeout=30)
            if w.is_alive():
                w.terminate()
        print("WorkerPool: all workers shut down")
    
    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
