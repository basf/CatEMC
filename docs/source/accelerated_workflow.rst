Performance and restarts
========================

Almost all the time of an EMC simulation is spent in the potential: every
step relaxes each replica once, and every ``md_generation`` steps a short
MD run is added. This page describes how CatEMC distributes that work,
which settings control it and how a long simulation survives the time
limit of a batch job.

How a step is executed
----------------------

.. raw:: html

   <div class="catemc-diagram">
   <svg viewBox="0 0 900 250" width="900" role="img"
        aria-label="The main process proposes one move per replica and sends the candidates to a pool of persistent worker processes. Each worker holds its own calculators on one device, relaxes the candidates it receives and returns them. The main process then accepts or rejects and writes the database.">
     <defs>
       <marker id="aw-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
         <path d="M0 0 L10 5 L0 10 z" class="d-arrow"/>
       </marker>
     </defs>
     <text class="d-muted d-small" x="10" y="20">MAIN PROCESS</text>
     <rect class="d-box" x="10" y="30" width="250" height="56" rx="10"/>
     <text class="d-text d-head" x="135" y="54" text-anchor="middle">1. Propose</text>
     <text class="d-muted d-small" x="135" y="72" text-anchor="middle">one move per replica</text>
     <rect class="d-box" x="10" y="182" width="250" height="56" rx="10"/>
     <text class="d-text d-head" x="135" y="206" text-anchor="middle">3. Accept or reject</text>
     <text class="d-muted d-small" x="135" y="224" text-anchor="middle">exchange, write the database</text>
     <path class="d-line" d="M135 180 L135 90" marker-end="url(#aw-a)"/>
     <text class="d-muted d-small" x="145" y="140">next step</text>

     <text class="d-muted d-small" x="420" y="20">PERSISTENT WORKERS</text>
     <rect class="d-box" x="420" y="30" width="470" height="208" rx="10"/>
     <text class="d-text d-head" x="655" y="56" text-anchor="middle">2. Evaluate: (MD,) relaxation, energy</text>
     <text class="d-muted d-small" x="436" y="88">GPU 0</text>
     <rect class="d-blue" x="486" y="72" width="88" height="26" rx="6"/>
     <rect class="d-blue" x="584" y="72" width="88" height="26" rx="6"/>
     <rect class="d-blue" x="682" y="72" width="88" height="26" rx="6"/>
     <rect class="d-blue" x="780" y="72" width="88" height="26" rx="6"/>
     <text class="d-muted d-small" x="436" y="132">GPU 1</text>
     <rect class="d-aqua" x="486" y="116" width="88" height="26" rx="6"/>
     <rect class="d-aqua" x="584" y="116" width="88" height="26" rx="6"/>
     <rect class="d-aqua" x="682" y="116" width="88" height="26" rx="6"/>
     <rect class="d-aqua" x="780" y="116" width="88" height="26" rx="6"/>
     <text class="d-muted d-small" x="655" y="174" text-anchor="middle">gpu_devices: [0, 1]   procs_per_device: 4</text>
     <text class="d-muted d-small" x="655" y="196" text-anchor="middle">every worker loads its calculators once</text>
     <text class="d-muted d-small" x="655" y="214" text-anchor="middle">and keeps them for the whole run</text>

     <path class="d-line" d="M262 50 L416 50" marker-end="url(#aw-a)"/>
     <text class="d-muted d-small" x="339" y="42" text-anchor="middle">candidates</text>
     <path class="d-line" d="M416 210 L264 210" marker-end="url(#aw-a)"/>
     <text class="d-muted d-small" x="340" y="202" text-anchor="middle">relaxed structures</text>
   </svg>
   </div>

The main process holds the population. In every step it proposes one move
per replica and hands the candidates to a pool of worker processes
(:class:`~catemc.worker_pool.WorkerPool`). A worker creates its
calculators once, when the run starts, and then evaluates one candidate
after the other. Loading a machine-learning potential takes several
seconds, which would add up over the many thousand evaluations of a run.
With persistent workers it is paid once per worker.

When all candidates are back, the main process applies the acceptance
rule of each move, attempts the exchanges, and writes the step to the
database.

Workers and devices
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Argument
     - Default
     - Meaning
   * - ``gpu_devices``
     - ``None``
     - list of GPU numbers. Every worker sees exactly one of them, through
       ``CUDA_VISIBLE_DEVICES``. With ``None`` the variable is left alone
       and the workers use the default device of the calculator, which is
       the right choice for a CPU run.
   * - ``procs_per_device``
     - 1
     - worker processes per device. The number of workers is
       ``len(gpu_devices) * procs_per_device``.
   * - ``worker_startup_timeout``
     - 300
     - seconds allowed for all workers to load their calculators
   * - ``worker_result_timeout``
     - 3600
     - seconds allowed for the next evaluation to come back

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: yaml

         emc:
           gpu_devices: [0, 1]
           procs_per_device: 16

      .. code-block:: console

         $ catemc run config.yaml -s emc.gpu_devices=[0,1,2,3]

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         emc = GEEMC(..., gpu_devices=[0, 1], procs_per_device=16)

**How many workers?** A step evaluates as many candidates as there are
replicas, so more workers than replicas are never busy. Below that limit,
several workers per GPU pay off as long as the GPU has memory and compute
to spare, because relaxations of a few hundred atoms do not saturate a
modern GPU on their own. The examples use 32 workers on one GPU. Start
there, watch ``nvidia-smi`` during the first steps, and reduce the number
if the memory fills up.

**One thread per worker.** Many processes that each start as many threads
as the node has cores slow each other down. Set the thread counts to one
before starting the run:

.. code-block:: bash

   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1

Rules for a Python run script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workers are started with the *spawn* method, which imports the run
script again in every worker. Two rules follow, and all example scripts
obey them:

.. code-block:: python

   def create_calculators():            # 1. a function at the top level of the module
       from mace.calculators.mace import MACECalculator
       return [MACECalculator(model_paths=MODEL_PATH, device='cuda')]


   def main():
       emc = GEEMC(..., calculator_factory=create_calculators)
       emc.run(total_steps=1000)


   if __name__ == '__main__':            # 2. the run is started under this guard
       main()

Import the potential inside ``create_calculators`` and not at the top of
the script. The main process then never loads it and does not occupy GPU
memory. On the command line both rules are taken care of.

The database
------------

Every evaluated structure, accepted or not, is written to the ASE
database ``{name}.db``. Two settings keep this file fast.

**Trimming.** A database that grows without limit becomes slow to write.
Every ``trim_generation`` steps the older steps are moved to a backup
file ``{name}_bak0.db``, ``{name}_bak1.db``, ... and the live database
keeps only what the simulation needs to continue. The analysis reads the
backup files and the live database together. Do not delete them.

**Scratch directory.** On a cluster the calculation directory is usually
on a network file system, where the many small writes of a database are
slow. ``scratch_dir`` puts the live database on a local disk of the node:

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Argument
     - Default
     - Meaning
   * - ``scratch_dir``
     - ``None``
     - folder for the live database. ``None`` writes it in the
       calculation directory.
   * - ``scratch_sync_interval``
     - ``None``
     - copy the live database to the calculation directory every this
       many steps. With ``None`` it is copied when the database is trimmed
       and when the run ends.
   * - ``overwrite_scratch_db``
     - ``False``
     - before starting, replace the databases in ``scratch_dir`` by those
       of the calculation directory
   * - ``traj_write_interval``
     - 1
     - append the population to the working trajectories in
       ``tmp_{name}/`` every this many steps. Production runs do not need
       every step there, since the database holds all samples.
       50 is a common choice.

The backup files are always copied to the calculation directory as soon
as they are written.

When a run starts with ``scratch_dir``, CatEMC looks for the live
database in this order: the scratch directory, the calculation directory,
the newest backup file. A database that is still on the scratch disk
therefore wins. If the scratch disk holds a stale database from an older
job, for example because the new job runs on the same node, set
``overwrite_scratch_db`` once so that the calculation directory is used.

Restarting
----------

``run(total_steps=N)`` continues an existing simulation from its last
completed step. A job that reached its time limit is resumed by
submitting the same script again, and a finished simulation is extended
by raising ``total_steps``.

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Situation
     - What to do
   * - The job hit its time limit
     - submit it again
   * - More steps are needed
     - raise ``total_steps`` and run again
   * - The job was killed while a scratch directory was in use
     - submit it again. At most the steps since the last copy are
       repeated. Set ``scratch_sync_interval`` to bound that number.
   * - The last part of the run should be repeated, for example after a
       node failure corrupted it
     - ``run(starting_step=S, total_steps=N, restart_with_trim=True)``
       deletes everything after step ``S`` and continues from there

A job script
------------

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: bash

         #!/bin/bash
         #SBATCH --gres=gpu:1
         #SBATCH --cpus-per-task=32
         #SBATCH --time=24:00:00
         set -e
         export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

         catemc run config.yaml \
             -s emc.gpu_devices=[0] -s emc.procs_per_device=32 \
             -s emc.scratch_dir=$TMPDIR/catemc -s emc.scratch_sync_interval=50 \
             -s emc.traj_write_interval=50

   .. tab-item:: Python
      :sync: python

      .. code-block:: bash

         #!/bin/bash
         #SBATCH --gres=gpu:1
         #SBATCH --cpus-per-task=32
         #SBATCH --time=24:00:00
         set -e
         export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

         python -u geemc_run.py > emc.out 2> emc.err

      ``python -u`` writes the output without buffering, so the progress
      of the run can be followed in ``emc.out``. The settings of the
      table above are arguments of the class in ``geemc_run.py``.

Output
------

With ``verbose=True`` every move is reported with the replicas involved,
the energy changes, the prefactor of the acceptance rule and the
decision, followed by the summary of the step:

.. code-block:: text

   Replicas: [1, 0], parents: ['Au3Cu8Ni8Pd7Pt6', 'Au3Cu8Ni8Pd7Pt6'], offsprings: ['Au3Cu9Ni7Pd7Pt6', 'Au3Cu7Ni9Pd7Pt6'], operation: EMCInterSwap, deltas: 0.178 and -0.156, prefactor: 1.0.
   Replicas 1 0 inter move accepted!
   Replica: 2, parent: Au3Cu8Ni8Pd7Pt6, offspring: Au3Cu8Ni8Pd7Pt6, operation: EMCIntraSwap, delta: -0.032, prefactor: 1.
   Replica 2 intra move accepted!
   ...
   Generation 10: acceptance 10/32 (0.312), best_raw_score 4810.01537109, elapsed 26.9 min

With ``verbose=False`` only the last line is printed.

The acceptance ratio of a step is the first thing to watch. Values of a
few percent in every step mean that the moves are too aggressive for the
temperatures of the run, or that many relaxations fail.

If something goes wrong
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Message or symptom
     - Cause and remedy
   * - ``Only m/n workers initialized within 300 seconds``
     - loading the potential in many workers at once takes longer than
       the limit. Raise ``worker_startup_timeout``.
   * - A traceback of the calculator right at the start
     - a worker could not create its calculators: wrong model path, or
       not enough GPU memory for this number of workers. Reduce
       ``procs_per_device``.
   * - ``No worker result received within 3600 seconds``
     - one evaluation takes longer than the limit, typically a large
       cell on a CPU. Raise ``worker_result_timeout``, or shorten the MD
       run.
   * - ``Worker processes terminated before returning results``
     - a worker was killed, usually by the out-of-memory handler of the
       node. Reduce ``procs_per_device`` or request more memory.
   * - Nearly every candidate is rejected
     - relaxations do not converge within ``max_relax_steps``, or the
       potential is used outside its training domain. Set
       ``write_relax_traj`` in the fitness settings and look at the
       relaxation trajectories in ``tmp_{name}/``.
   * - The run starts from step 1 although databases exist
     - ``name`` or the calculation directory changed. Both identify the
       simulation.
