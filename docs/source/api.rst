API reference
=============

The reference is generated from the docstrings of the package. The
tutorials explain how the pieces work together:
:doc:`input_populations`, :doc:`emc` and :doc:`analysis`.

The classes you call
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Object
     - Purpose
   * - :func:`catemc.generate.generate_population`
     - random bulk and surface structures at a composition
   * - :func:`catemc.generate.assign_mu_windows`
     - temperatures and μ-windows for a surface population
   * - :func:`catemc.generate.get_ls_elemental_mu`
     - elemental chemical potentials from those of gas molecules
   * - :class:`catemc.emc.EMC`
     - canonical ensemble
   * - :class:`catemc.geemc.GEEMC`
     - Gibbs ensemble
   * - :class:`catemc.gcemc.GCEMC`
     - grand-canonical ensemble
   * - :class:`catemc.analysis.GEEMCAnalysis`
     - short-range order and chemical-potential differences of a bulk run
   * - :class:`catemc.analysis.GCEMCAnalysis`
     - surface phase diagrams and free energies of a surface run
   * - :class:`catemc.repulsive.RepulsivePotential`
     - pre-relaxation calculator that removes close contacts

Structure generation
--------------------

.. autosummary::
   :toctree: generated
   :recursive:

   catemc.generate

Simulations
-----------

.. autosummary::
   :toctree: generated
   :recursive:

   catemc.emc
   catemc.geemc
   catemc.gcemc

Analysis
--------

.. autosummary::
   :toctree: generated
   :recursive:

   catemc.analysis

Command line
------------

.. autosummary::
   :toctree: generated
   :recursive:

   catemc.cli
   catemc.config

Building blocks
---------------

These modules are used by the classes above. You need them when you write
your own fitness function, calculator or move.

.. autosummary::
   :toctree: generated
   :recursive:

   catemc.calculation_templates
   catemc.repulsive
   catemc.optimizers
   catemc.worker_pool
   catemc.emc_population
   catemc.emc_operators
   catemc.gc_utils
