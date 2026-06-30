#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import multiprocessing as mp
import os
import time


def _load_function(path):
    module_name, function_name = path.split(':')
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _worker_loop(task_queue, result_queue, worker_id, gpu_device,
                 calculator_factory, calculator_factory_kwargs,
                 fitness_function, fitness_kwargs):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    calculators = _load_function(calculator_factory)(**calculator_factory_kwargs)
    fitness = _load_function(fitness_function)
    result_queue.put(('ready', worker_id, None))

    while True:
        task = task_queue.get()
        if task is None:
            break
        task_id, atoms, run_md, fix_indices, extra_kwargs = task
        kwargs = dict(fitness_kwargs)
        kwargs.update(extra_kwargs)
        kwargs['run_md'] = run_md
        kwargs['fix_indices'] = fix_indices
        relaxed = fitness(atoms, calculator_list=calculators, **kwargs)
        result_queue.put(('result', task_id, relaxed))


class WorkerPool(object):

    def __init__(self,
                 calculator_factory,
                 fitness_function,
                 calculator_factory_kwargs=None,
                 fitness_kwargs=None,
                 num_workers=1,
                 gpu_devices=None,
                 startup_timeout=300.,
                 result_timeout=3600.):

        self.num_workers = num_workers
        self.gpu_devices = [0] if gpu_devices is None else gpu_devices
        self.result_timeout = result_timeout
        self.context = mp.get_context('spawn')
        self.task_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        self.workers = []

        if calculator_factory_kwargs is None:
            calculator_factory_kwargs = {}
        if fitness_kwargs is None:
            fitness_kwargs = {}

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
            msg = self.result_queue.get(timeout=max(1., deadline - time.time()))
            if msg[0] == 'ready':
                ready += 1

    def evaluate(self, atoms_list, run_md=False, fix_indices=None, extra_kwargs=None):

        if extra_kwargs is None:
            extra_kwargs = {}
        for i, atoms in enumerate(atoms_list):
            self.task_queue.put((i, atoms, run_md, fix_indices, extra_kwargs))

        relaxed_atoms = [None] * len(atoms_list)
        n_done = 0
        while n_done < len(atoms_list):
            msg = self.result_queue.get(timeout=self.result_timeout)
            if msg[0] == 'result':
                relaxed_atoms[msg[1]] = msg[2]
                n_done += 1

        return relaxed_atoms

    def close(self):

        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5.)
            if worker.is_alive():
                worker.terminate()
