#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing as mp
import os
import queue
import time
import traceback


def _worker_loop(task_queue, result_queue, worker_id, gpu_device,
                 calculator_factory, calculator_factory_kwargs,
                 fitness_function, fitness_kwargs):
    if gpu_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    try:
        calculators = calculator_factory(**calculator_factory_kwargs)
        if not isinstance(calculators, (list, tuple)):
            calculators = [calculators]
        if len(calculators) == 0:
            raise ValueError('calculator_factory returned no calculators')
        result_queue.put(('ready', worker_id, None))
    except Exception:
        result_queue.put(('fatal', worker_id, traceback.format_exc()))
        return

    while True:
        task = task_queue.get()
        if task is None:
            return

        task_id, atoms, run_md = task
        kwargs = dict(fitness_kwargs)
        kwargs['run_md'] = run_md
        try:
            relaxed = fitness_function(
                    atoms,
                    calculator_list=calculators,
                    **kwargs)
            result_queue.put(('result', task_id, relaxed))
        except Exception:
            result_queue.put(('error', task_id, traceback.format_exc()))


class WorkerPool(object):

    """Persistent processes for calculator-based fitness evaluation.

    Each worker creates its calculator sequence once and reuses it for every
    structure assigned to that process.

    Parameters
    ----------
    calculator_factory : callable
        Top-level function returning one ASE calculator or a sequence of ASE
        calculators. A sequence is applied in order by the fitness function.
    fitness_function : callable
        Top-level function accepting ``atoms``, ``calculator_list`` and the
        entries in ``fitness_kwargs``.
    calculator_factory_kwargs : dict, optional
        Keyword arguments passed to ``calculator_factory`` in every worker.
    fitness_kwargs : dict, optional
        Static keyword arguments passed to ``fitness_function`` for every task.
    num_workers : int, default 1
        Number of persistent worker processes.
    gpu_devices : sequence of int or None, optional
        Device identifiers assigned to workers round-robin. A ``None`` entry
        leaves ``CUDA_VISIBLE_DEVICES`` unchanged for that worker.
    startup_timeout : float, default 300
        Maximum time in seconds to wait for all workers to initialize.
    result_timeout : float, default 3600
        Maximum time in seconds to wait for the next task result.
    """

    def __init__(self,
                 calculator_factory,
                 fitness_function,
                 calculator_factory_kwargs=None,
                 fitness_kwargs=None,
                 num_workers=1,
                 gpu_devices=None,
                 startup_timeout=300.,
                 result_timeout=3600.):

        if not callable(calculator_factory):
            raise TypeError('calculator_factory must be callable')
        if not callable(fitness_function):
            raise TypeError('fitness_function must be callable')
        if num_workers < 1:
            raise ValueError('num_workers must be at least 1')

        self.num_workers = num_workers
        self.gpu_devices = [0] if gpu_devices is None else list(gpu_devices)
        if len(self.gpu_devices) == 0:
            raise ValueError('gpu_devices cannot be empty')
        self.result_timeout = result_timeout
        self.context = mp.get_context('spawn')
        self.task_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        self.workers = []
        self.closed = False

        calculator_factory_kwargs = (
                {} if calculator_factory_kwargs is None
                else dict(calculator_factory_kwargs))
        fitness_kwargs = {} if fitness_kwargs is None else dict(fitness_kwargs)

        for i in range(num_workers):
            gpu_device = self.gpu_devices[i % len(self.gpu_devices)]
            worker = self.context.Process(
                    target=_worker_loop,
                    args=(self.task_queue, self.result_queue, i, gpu_device,
                          calculator_factory, calculator_factory_kwargs,
                          fitness_function, fitness_kwargs),
                    daemon=True)
            worker.start()
            self.workers.append(worker)

        deadline = time.time() + startup_timeout
        ready = 0
        while ready < num_workers:
            timeout = deadline - time.time()
            if timeout <= 0:
                self.close()
                raise TimeoutError(
                        'Only {}/{} workers initialized within {} seconds'.format(
                            ready, num_workers, startup_timeout))
            try:
                msg = self.result_queue.get(timeout=timeout)
            except queue.Empty:
                self.close()
                raise TimeoutError(
                        'Only {}/{} workers initialized within {} seconds'.format(
                            ready, num_workers, startup_timeout))
            if msg[0] == 'ready':
                ready += 1
            elif msg[0] == 'fatal':
                self.close()
                raise RuntimeError(msg[2])

    def evaluate(self, atoms_list, run_md=False):
        """Evaluate structures and return results in input order.

        Parameters
        ----------
        atoms_list : sequence of ase.Atoms
            Structures to evaluate.
        run_md : bool, default False
            Whether the fitness function should perform its MD stage.
        Returns
        -------
        list of ase.Atoms
            Evaluated structures in the same order as ``atoms_list``.
        """

        if self.closed:
            raise RuntimeError('WorkerPool is closed')
        if len(atoms_list) == 0:
            return []

        for task_id, atoms in enumerate(atoms_list):
            self.task_queue.put((task_id, atoms, run_md))

        relaxed_atoms = [None] * len(atoms_list)
        n_done = 0
        while n_done < len(atoms_list):
            try:
                msg = self.result_queue.get(timeout=self.result_timeout)
            except queue.Empty:
                dead = [worker.pid for worker in self.workers
                        if not worker.is_alive()]
                if dead:
                    raise RuntimeError(
                            'Worker processes terminated before returning results: '
                            + ', '.join(map(str, dead)))
                raise TimeoutError(
                        'No worker result received within {} seconds'.format(
                            self.result_timeout))

            if msg[0] == 'result':
                relaxed_atoms[msg[1]] = msg[2]
                n_done += 1
            elif msg[0] == 'error':
                raise RuntimeError(
                        'Fitness evaluation failed for task {}:\n{}'.format(
                            msg[1], msg[2]))
            elif msg[0] == 'fatal':
                raise RuntimeError(msg[2])

        return relaxed_atoms

    def close(self):
        """Stop all worker processes."""

        if self.closed:
            return
        self.closed = True
        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=30.)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        self.task_queue.close()
        self.result_queue.close()
