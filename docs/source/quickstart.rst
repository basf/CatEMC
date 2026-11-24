Quickstart
==========

This page runs a complete CatEMC study in about a minute: a Gibbs-ensemble
simulation of a small Cu\ :sub:`0.25`\ Ni\ :sub:`0.25`\ Pd\ :sub:`0.2`\ Pt\
:sub:`0.2`\ Au\ :sub:`0.1` alloy that gives the chemical-potential
differences between the five metals. It uses the EMT calculator that ships
with ASE, so it needs no GPU and no machine-learning potential.

.. note::

   EMT and a 32-atom cell make this example fast. The numbers are a toy
   result. The :doc:`tutorials` use production settings and the MACE-MPA-0
   foundation model.

The command-line version needs nothing but ``pip install catemc``. The files
of this page are also in ``examples/quickstart`` of the
`GitHub repository <https://github.com/basf/CatEMC>`_, and the Python version
is shown in full below.

Run it
------

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      Write a commented starting configuration and its template
      structure, then run the three stages:

      .. code-block:: console

         $ catemc init gibbs
         Wrote config.yaml. Provide the template structure named in population.template, edit the file, then run:
           catemc generate config.yaml
           catemc run config.yaml
           catemc analyze config.yaml
         $ ase build -x fcc -a 3.8 --cubic -r 2,2,2 Cu template.traj
         $ catemc generate config.yaml
         $ catemc run config.yaml
         $ catemc analyze config.yaml

      ``ase build`` is the command line of ASE. It writes the 32-site fcc
      cell that ``population.template`` names. Any structure file that ASE
      reads can be the template.

      ``catemc init`` also writes starting files for the other two ensembles:
      ``catemc init canonical`` and ``catemc init grand_canonical``.

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         $ cd examples/quickstart
         $ python quickstart.py

      .. literalinclude:: ../../examples/quickstart/quickstart.py
         :language: python
         :lines: 15-

What the three stages do
------------------------

Structure generation
^^^^^^^^^^^^^^^^^^^^

``catemc generate`` reads the template structure, decorates it randomly at
the requested composition, and gives every replica a temperature:

.. code-block:: console

   $ catemc generate config.yaml
   Random seed: 1
   Template: Cu32 (32 sites)
   Replica   0  T = 600 K  Au3Cu8Ni8Pd7Pt6
   Replica   1  T = 600 K  Au3Cu8Ni8Pd7Pt6
   Replica   2  T = 700 K  Au3Cu8Ni8Pd7Pt6
   Replica   3  T = 700 K  Au3Cu8Ni8Pd7Pt6
   Replica   4  T = 800 K  Au3Cu8Ni8Pd7Pt6
   Replica   5  T = 800 K  Au3Cu8Ni8Pd7Pt6
   Replica   6  T = 900 K  Au3Cu8Ni8Pd7Pt6
   Replica   7  T = 900 K  Au3Cu8Ni8Pd7Pt6
   Wrote 8 replicas to CuNiPdPtAu_bulk_input_pop.traj

The population has two replicas at each of four temperatures. Two replicas
per temperature is the minimum for the Gibbs ensemble, because replicas at
the same temperature exchange atoms with each other.

EMC simulation
^^^^^^^^^^^^^^

``catemc run`` starts the worker processes, evaluates the initial population
and performs 100 EMC steps. In every step each replica is changed by one
Monte Carlo move and relaxed, and every tenth step the replicas are also
propagated by a short molecular-dynamics run. The last line of each step
summarizes it:

.. code-block:: text

   Generation 100: acceptance 6/8 (0.750), best_raw_score 0.82903473, elapsed 0.7 min

Running the same command again continues from the last completed step, so a
longer simulation only needs a larger ``run.total_steps``:

.. code-block:: console

   $ catemc run config.yaml -s run.total_steps=300

Post-simulation analysis
^^^^^^^^^^^^^^^^^^^^^^^^

``catemc analyze`` reweights the samples of all temperatures with MBAR and
prints the chemical-potential differences
:math:`\dmu{A}{B} = \mu_\mathrm{B} - \mu_\mathrm{A}`, listed as ``A2B``, at the
simulated temperatures and at any other temperature you ask for (650 K here):

.. code-block:: console

   $ catemc analyze config.yaml
   Wrote the trajectory of replica 0.
   Wrote CuNiPdPtAu_bulk_mbar_diagnostics.pkl
   Wrote the Warren-Cowley short-range order parameters.
   Metal chemical-potential differences mu_B - mu_A (eV), listed as 'A2B':
     600 K: Cu2Ni 0.134, Cu2Pd -0.035, Cu2Pt -0.222, Cu2Au -0.047, Ni2Pd -0.170, ...
     650 K: Cu2Ni 0.133, Cu2Pd -0.038, Cu2Pt -0.223, Cu2Au -0.053, Ni2Pd -0.172, ...
     700 K: Cu2Ni 0.133, Cu2Pd -0.041, Cu2Pt -0.225, Cu2Au -0.059, Ni2Pd -0.174, ...
     800 K: Cu2Ni 0.132, Cu2Pd -0.046, Cu2Pt -0.229, Cu2Au -0.072, Ni2Pd -0.179, ...
     900 K: Cu2Ni 0.133, Cu2Pd -0.051, Cu2Pt -0.233, Cu2Au -0.085, Ni2Pd -0.185, ...
   Metal chemical potentials for chemical_potentials.metals of a surface run (eV):
     metals:
       Cu: {600: 0.0000, 650: 0.0000, 700: 0.0000, 800: 0.0000, 900: 0.0000}
       Ni: {600: 0.1344, 650: 0.1334, 700: 0.1327, 800: 0.1325, 900: 0.1332}
       Pd: {600: -0.0352, 650: -0.0382, 700: -0.0411, 800: -0.0462, 900: -0.0514}
       Pt: {600: -0.2218, 650: -0.2234, 700: -0.2252, 800: -0.2287, 900: -0.2334}
       Au: {600: -0.0473, 650: -0.0528, 700: -0.0590, 800: -0.0721, 900: -0.0854}
   Wrote CuNiPdPtAu_bulk_mbar_dmu_estimates.pkl

.. figure:: _static/figures/quickstart_dmu_light.png
   :figclass: only-light
   :width: 560px
   :align: center
   :alt: Chemical potentials of Ni, Pd, Pt and Au relative to Cu between 600 and 900 K

   Chemical potentials relative to Cu from the quickstart run (EMT, 32-atom
   cell, 100 steps).

.. figure:: _static/figures/quickstart_dmu_dark.png
   :figclass: only-dark
   :width: 560px
   :align: center
   :alt: Chemical potentials of Ni, Pd, Pt and Au relative to Cu between 600 and 900 K

   Chemical potentials relative to Cu from the quickstart run (EMT, 32-atom
   cell, 100 steps).

The ``metals`` block lists the same result as chemical potentials relative
to Cu. It is the ``chemical_potentials.metals`` entry of the configuration
file of a surface in contact with this bulk (:doc:`surface_gcemc`).

Your numbers will differ in the second decimal, because the worker processes
do not share one random-number stream. A run this short carries a statistical
error of about 0.01 to 0.02 eV.

The files that were written
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - File
     - Content
   * - ``template.traj``
     - the template structure, written by ``ase build`` (command line only)
   * - ``CuNiPdPtAu_bulk_input_pop.traj``
     - the initial population (command line only)
   * - ``CuNiPdPtAu_bulk.db``, ``..._bak0.db``, ...
     - ASE databases with every accepted and attempted structure. Old steps
       are moved to the backup files every ``trim_generation`` steps.
   * - ``CuNiPdPtAu_bulk.log``
     - the population after every step
   * - ``CuNiPdPtAu_bulk_mbar_dmu_estimates.pkl``
     - ``{temperature: {'A2B': mu_B - mu_A}}`` in eV
   * - ``CuNiPdPtAu_bulk_sro_data.pkl``
     - Warren-Cowley short-range order parameters per temperature
   * - ``CuNiPdPtAu_bulk_geemc_0.traj``
     - the trajectory of replica 0
   * - ``CuNiPdPtAu_bulk_mbar_*.pkl``
     - cached samples, reused by the next analysis

The configuration file
----------------------

Everything the three commands did is described by ``config.yaml``:

.. literalinclude:: ../../examples/quickstart/config.yaml
   :language: yaml
   :lines: 12-

Two commands help while you edit this file:

.. code-block:: console

   $ catemc check config.yaml                    # show what the file resolves to
   $ catemc run config.yaml -s emc.procs_per_device=4   # replace one entry

Where to go next
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Method
      :link: concepts
      :link-type: doc
      :class-card: sd-card-hover

      What the replicas, the moves and the reweighting do.

   .. grid-item-card:: Tutorials
      :link: tutorials
      :link-type: doc
      :class-card: sd-card-hover

      A high-entropy alloy and a high-entropy oxide, bulk and surface.

   .. grid-item-card:: Use your own potential
      :link: cli
      :link-type: doc
      :class-card: sd-card-hover

      Replace the ``calculators`` section by MACE or any ASE calculator.

   .. grid-item-card:: Every key of the file
      :link: config_reference
      :link-type: doc
      :class-card: sd-card-hover

      The complete reference of the configuration file.
