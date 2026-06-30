Accelerated workflow
====================

The accelerated CatEMC workflow keeps the input-script style of the original
package while avoiding repeated calculator initialization and excessive I/O.

Persistent worker pool
----------------------

Set ``use_worker_pool=True`` in ``GEEMC`` or ``GCEMC``. The EMC driver starts a
pool of worker processes. Each worker imports the calculation script, calls
``create_calculators()`` once, and then reuses that calculator list for every
candidate assigned to the worker.

The calculation script should define two functions:

.. code-block:: python

   def create_calculators():
       return [calc1, calc2]

   def get_fitness_kwargs():
       return dict(...)

The run script passes ``fitness_kwargs=get_fitness_kwargs()`` into the EMC
constructor. Static surface objects such as ``adsorption_sites`` and
``slab_indices`` are held by the workers instead of being reconstructed for every
candidate.

Scratch database
----------------

``scratch_dir`` controls where the live ASE database is written.

.. code-block:: python

   scratch_dir = '/scratch/$USER/catemc_job'
   scratch_sync_interval = 100

When ``scratch_dir`` is ``None``, CatEMC writes directly in ``calculation_dir``.
When a path is provided, the live database is kept on scratch and synchronized
back to the run directory at clean shutdown. Setting ``scratch_sync_interval`` to
an integer performs additional periodic synchronizations.

Trajectory throttling
---------------------

``traj_write_interval`` controls how often the EMC trajectory is appended.
Production runs usually do not need every generation in the trajectory file:

.. code-block:: python

   traj_write_interval = 50

The database still stores the EMC samples used for restart and MBAR analysis.

Logging
-------

``verbose=True`` prints detailed per-move diagnostics and the compact generation
summary. ``verbose=False`` prints only the generation summary.

.. code-block:: text

   Generation 10: acceptance 10/32 (0.312), best_raw_score 4810.01537109, elapsed 26.9 min

Recommended launch environment
------------------------------

For many worker processes per GPU, use one CPU thread per process:

.. code-block:: bash

   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1

A typical surface command is:

.. code-block:: bash

   python -u gcemc_run.py 0 16 50 /scratch/$USER/catemc_gcemc 100 1

A typical bulk command is:

.. code-block:: bash

   python -u geemc_run.py 0 16 50 /scratch/$USER/catemc_geemc 100 1

The arguments are:

.. code-block:: text

   gpu_devices workers_per_gpu traj_write_interval scratch_dir scratch_sync_interval verbose
