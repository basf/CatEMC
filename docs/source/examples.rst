Examples
========

Every simulation of the tutorials is in the ``examples`` folder of the
repository, once as Python scripts and once as a YAML file for the command
line. The two versions of a run use the same settings and write the same
files.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc
      :class-card: sd-card-hover

      ``examples/quickstart``

      A 32-atom CuNiPdPtAu alloy with the EMT calculator. Runs in a
      minute on a laptop.

   .. grid-item-card:: High-entropy alloy
      :link: ex-hea
      :link-type: ref
      :class-card: sd-card-hover

      ``examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1``

      Bulk and (111) surface under ammonia-decomposition conditions.

   .. grid-item-card:: High-entropy oxide
      :link: ex-heo
      :link-type: ref
      :class-card: sd-card-hover

      ``examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5``

      Spinel bulk with oxygen vacancies and (311) surface under
      H\ :sub:`2`-reduction conditions.

Run a surface example after the bulk example of the same material. The
surface run reads the chemical-potential differences of the metals that
the bulk analysis writes.

Quickstart
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Content
   * - ``quickstart.py``
     - population, Gibbs-ensemble run and analysis in one script
   * - ``config.yaml``
     - the same study for the command line. It is the file that
       ``catemc init gibbs`` writes.
   * - ``run_cli.sh``
     - the four commands

.. _ex-hea:

High-entropy alloy
------------------

Co\ :sub:`0.25`\ Mo\ :sub:`0.45`\ Fe\ :sub:`0.1`\ Ni\ :sub:`0.1`\ Cu\ :sub:`0.1`,
a catalyst for ammonia decomposition.

.. list-table:: ``examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk``
   :header-rows: 1
   :widths: 36 36 28

   * - File
     - Content
     - Tutorial
   * - ``generate_initial_population.py``
     - 48 random fcc replicas of 512 atoms at twelve temperatures
     - :doc:`input_populations`
   * - ``geemc_run.py``
     - Gibbs-ensemble run
     - :doc:`bulk_geemc`
   * - ``emc_run.py``
     - canonical run at fixed composition
     - :doc:`canonical_emc`
   * - ``mbar_analysis.py``
     - short-range order and chemical-potential differences
     - :doc:`analysis`
   * - ``config.yaml``, ``run_cli.sh``
     - command-line version of the Gibbs-ensemble study
     - :doc:`bulk_geemc`
   * - ``config_canonical.yaml``
     - command-line version of the canonical study
     - :doc:`canonical_emc`

.. list-table:: ``examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/111``
   :header-rows: 1
   :widths: 36 36 28

   * - File
     - Content
     - Tutorial
   * - ``generate_initial_population.py``
     - 56 slabs with N, H and NH adlayers, four temperatures, μ-windows
     - :doc:`input_populations`
   * - ``get_internal_partition_functions.py``
     - internal partition functions of the fragments
     - :doc:`surface_gcemc`
   * - ``gcemc_run.py``
     - grand-canonical run with N and H reservoirs
     - :doc:`surface_gcemc`
   * - ``mbar_analysis.py``
     - surface phase diagram and free energies at the early-stage and
       the equilibrium condition
     - :doc:`analysis`
   * - ``plot_*.py``
     - figures of the surface phase diagram
     - :doc:`visualization`
   * - ``config.yaml``, ``run_cli.sh``
     - command-line version of the whole study
     - :doc:`surface_gcemc`

.. _ex-heo:

High-entropy oxide
------------------

(Co\ :sub:`0.2`\ Ni\ :sub:`0.2`\ Mg\ :sub:`0.2`\ Zn\ :sub:`0.2`\ Mn\ :sub:`0.2`)Al\ :sub:`2`\ O\ :sub:`3.5`,
a spinel with one oxygen vacancy per eight oxygen sites.

.. list-table:: ``examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk``
   :header-rows: 1
   :widths: 36 36 28

   * - File
     - Content
     - Tutorial
   * - ``Al2CoO4.cif``
     - the spinel lattice that is decorated
     - :doc:`input_populations`
   * - ``generate_initial_population.py``
     - random cations and oxygen vacancies
     - :doc:`input_populations`
   * - ``geemc_run.py``
     - Gibbs-ensemble run at constant pressure. Metals and oxygen
       vacancies are exchanged between the replicas.
     - :doc:`bulk_geemc`
   * - ``emc_run.py``
     - canonical run at a fixed oxygen-vacancy concentration
     - :doc:`canonical_emc`
   * - ``mbar_analysis.py``
     - chemical-potential differences of the six metals
     - :doc:`analysis`
   * - ``config.yaml``, ``run_cli.sh``
     - command-line version of the Gibbs-ensemble study
     - :doc:`bulk_geemc`
   * - ``config_canonical.yaml``
     - command-line version of the canonical study
     - :doc:`canonical_emc`

The (311) folder contains two studies of the same surface. The files with
``OH`` in their name use O and H reservoirs (H\ :sub:`2` reduction, the
study of the tutorials). The files with ``COH`` add a carbon reservoir.

.. list-table:: ``examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/311``
   :header-rows: 1
   :widths: 36 36 28

   * - File
     - Content
     - Tutorial
   * - ``template_..._311_slab_mace_opt.traj``
     - the relaxed stoichiometric slab
     - :doc:`input_populations`
   * - ``generate_initial_population_OH.py``, ``..._COH.py``
     - slabs with three oxygen-vacancy concentrations and random
       adlayers, μ-windows
     - :doc:`input_populations`
   * - ``get_internal_partition_functions.py``
     - internal partition functions of the fragments
     - :doc:`surface_gcemc`
   * - ``gcemc_run_OH.py``, ``gcemc_run_COH.py``
     - grand-canonical runs
     - :doc:`surface_gcemc`
   * - ``mbar_analysis_OH.py``, ``mbar_analysis_COH.py``
     - surface phase diagrams, lattice-oxygen vacancies, free energies
     - :doc:`analysis`
   * - ``plot_*.py``
     - figures of the surface phase diagrams
     - :doc:`visualization`
   * - ``config_OH.yaml``, ``run_cli_OH.sh``, ``config_COH.yaml``,
       ``run_cli_COH.sh``
     - command-line versions of the two studies
     - :doc:`surface_gcemc`

Script templates
----------------

``scripts/geemc_run.py`` and ``scripts/gcemc_run.py`` are compact
templates of a bulk and a surface run in Python, with every argument of
:class:`~catemc.geemc.GEEMC` and :class:`~catemc.gcemc.GCEMC` written out.
Copy one of them to start a new Python study.

Before you run an example
-------------------------

The examples carry the settings of the published simulations, including
file locations of the machine they ran on. Three things have to be
adapted.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - What
     - Command line
     - Python
   * - Machine-learning potential
     - ``model_paths`` in the ``calculators`` section
     - ``MODEL_PATH`` at the top of the run script
   * - Calculation directory
     - ``directory`` (the default ``.`` is the folder of the YAML file)
     - ``dir_name`` in every script of the study
   * - GPUs and worker processes
     - ``emc.gpu_devices``, ``emc.procs_per_device``
     - ``gpu_devices``, ``procs_per_device``

The fine-tuned MACE models of the two materials are not part of the
repository. Any potential that describes the system can be used in their
place (:doc:`cli`).

To try an example before a production run, shorten it on the command line
without editing the file:

.. code-block:: console

   $ cd examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk
   $ catemc check config.yaml
   $ catemc generate config.yaml
   $ catemc run config.yaml -s run.total_steps=20 -s emc.procs_per_device=4

Starting your own system
------------------------

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      1. ``catemc init gibbs`` (or ``canonical``, ``grand_canonical``)
         writes a commented file.
      2. Set ``composition`` and ``template`` to your material and
         ``calculators`` to your potential.
      3. ``catemc check config.yaml`` after every edit.
      4. ``catemc generate config.yaml``, then look at the population
         with ``ase gui``.
      5. A short ``catemc run`` first. Check the acceptance ratios in
         the output before the long run.

   .. tab-item:: Python
      :sync: python

      1. Copy the four scripts of the example closest to your system.
      2. Change the template, the species and the composition in the
         generation script and inspect the population with ``ase gui``.
      3. Change ``create_calculators()`` and the element pools in the
         run script.
      4. For a surface, check the reference energies, the reaction
         conditions and the internal partition functions.
      5. A short run first. Check the acceptance ratios in the output
         before the long run.
