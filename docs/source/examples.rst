Examples
========

The repository contains working examples under ``examples/`` and compact script
templates under ``scripts/``.

Script templates
----------------

``scripts/geemc_run.py``
   Accelerated bulk GE-EMC entry point.

``scripts/gcemc_run.py``
   Accelerated surface GC-EMC entry point.

``scripts/calculate_bulk_fitness.py``
   Bulk calculator and relaxation template.

``scripts/calculate_surface_fitness.py``
   Surface calculator and relaxation template.

``scripts/submit.sh``
   Batch-submission template using persistent workers, scratch database writing,
   trajectory throttling, and detailed logging.

Example systems
---------------

``examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1``
   HEA bulk and surface examples.

``examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5``
   HEO bulk and surface examples.

Adapting an example
-------------------

1. Prepare and inspect the initial population trajectory.
2. Update the calculator path and calculator constructor in the fitness script.
3. Check target species, composition maps, reference energies, and internal
   partition functions.
4. Choose ``gpu_devices`` and ``procs_per_device`` for the available hardware.
5. Set ``scratch_dir`` to node-local storage for production runs.
6. Start with a short run and inspect ``emc.out``, ``emc.err``, and the ASE
   database before launching a long campaign.
