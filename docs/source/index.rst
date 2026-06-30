CatEMC
======

CatEMC is a Python package for evolutionary Monte Carlo sampling of complex
bulk and surface catalyst materials. It combines machine-learning interatomic
potentials with EMC moves, replica exchange, Gibbs-ensemble composition exchange,
grand-canonical adsorbate/surface moves and MBAR analysis.

The current workflow uses persistent worker processes for accelerated fitness
evaluation. Each worker initializes its calculator stack once and reuses it
throughout the run, while optional scratch-database I/O and trajectory throttling
reduce filesystem overhead on HPC systems.

.. image:: ../../images/catemc_workflow.png
   :alt: CatEMC workflow
   :align: center
   :width: 760px

Main workflow
-------------

Start with :doc:`installation`, then read :doc:`accelerated_workflow` before running the bulk and surface tutorials. Bulk :doc:`bulk_geemc` provides metal chemical-potential differences that can be used by surface :doc:`surface_gcemc`. The sampled databases can then be processed with :doc:`analysis` and plotted with :doc:`visualization`.

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   installation
   concepts
   accelerated_workflow
   input_populations
   bulk_geemc
   surface_gcemc
   analysis
   visualization
   api
