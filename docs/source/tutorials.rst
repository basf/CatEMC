Tutorials
=========

The tutorials follow two catalysts through a complete study, from the first
structure to the surface phase diagram. Both are taken from the CatEMC paper
and all input files are in the ``examples`` folder of the repository.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: High-entropy alloy (HEA)

      Co\ :sub:`0.25`\ Mo\ :sub:`0.45`\ Fe\ :sub:`0.1`\ Ni\ :sub:`0.1`\ Cu\ :sub:`0.1`

      * **Bulk**: fcc, 8×8×8 cell with 512 atoms, constant volume
      * **Surface**: (111), 6×6 cell with 6 layers (216 atoms)
      * **Reaction**: ammonia decomposition,
        2 NH\ :sub:`3` ⇌ N\ :sub:`2` + 3 H\ :sub:`2`
      * **Reservoirs**: N and H

      ``examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1``

   .. grid-item-card:: High-entropy oxide (HEO)

      (Co\ :sub:`0.2`\ Ni\ :sub:`0.2`\ Mg\ :sub:`0.2`\ Zn\ :sub:`0.2`\ Mn\ :sub:`0.2`)Al\ :sub:`2`\ O\ :sub:`3.5`

      * **Bulk**: spinel with oxygen vacancies, 2×2×2 cell, constant pressure
      * **Surface**: (311), 2×1 cell with 6 layers (304 atoms)
      * **Reaction**: H\ :sub:`2` reduction pretreatment
      * **Reservoirs**: O and H

      ``examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5``

The tutorials are organized by module, and every page treats both
catalysts. Each section has the Python version and the command-line version
of the same step, and the equations of the paper that the step implements.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tutorial
     - What you learn
   * - :doc:`input_populations`
     - Build the template lattice, decorate it at any composition, add
       vacancies and adsorbates, and assign temperatures and μ-windows.
   * - :doc:`emc`
     - How a simulation is set up: calculators, relaxation and MD settings,
       moves and their rates.
   * - :doc:`canonical_emc`
     - Fixed composition: equilibrium ordering of the HEA and HEO bulk,
       with a fixed oxygen-vacancy concentration for the HEO.
   * - :doc:`bulk_geemc`
     - Gibbs ensemble: chemical-potential differences between the metals
       of the HEA and HEO bulk.
   * - :doc:`surface_gcemc`
     - Grand-canonical ensemble: HEA(111) under ammonia-decomposition
       conditions and HEO(311) under H\ :sub:`2`-reduction conditions.
   * - :doc:`analysis`
     - MBAR reweighting: chemical potentials, short-range order, surface
       phase diagrams and free-energy surfaces.
   * - :doc:`visualization`
     - Turn the analysis files into figures.

The order of a study
--------------------

.. raw:: html

   <div class="catemc-diagram">
   <svg viewBox="0 0 900 190" width="900" role="img"
        aria-label="Order of a study: bulk Gibbs-ensemble run, its analysis gives the metal chemical-potential differences, which the surface grand-canonical run and its analysis need">
     <defs>
       <marker id="tu-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
         <path d="M0 0 L10 5 L0 10 z" class="d-arrow"/>
       </marker>
     </defs>
     <text class="d-muted d-small" x="10" y="20">BULK</text>
     <rect class="d-box" x="10" y="30" width="180" height="56" rx="10"/>
     <text class="d-text d-head" x="100" y="54" text-anchor="middle">Generate</text>
     <text class="d-muted d-small" x="100" y="72" text-anchor="middle">random bulk replicas</text>
     <rect class="d-box" x="230" y="30" width="180" height="56" rx="10"/>
     <text class="d-text d-head" x="320" y="54" text-anchor="middle">Gibbs-ensemble EMC</text>
     <text class="d-muted d-small" x="320" y="72" text-anchor="middle">GEEMC</text>
     <rect class="d-box" x="450" y="30" width="180" height="56" rx="10"/>
     <text class="d-text d-head" x="540" y="54" text-anchor="middle">Analyze</text>
     <text class="d-muted d-small" x="540" y="72" text-anchor="middle">Δμ of all metal pairs</text>
     <path class="d-line" d="M192 58 L226 58" marker-end="url(#tu-a)"/>
     <path class="d-line" d="M412 58 L446 58" marker-end="url(#tu-a)"/>
     <text class="d-muted d-small" x="10" y="122">SURFACE</text>
     <rect class="d-box" x="10" y="132" width="180" height="56" rx="10"/>
     <text class="d-text d-head" x="100" y="156" text-anchor="middle">Generate</text>
     <text class="d-muted d-small" x="100" y="174" text-anchor="middle">slabs, adlayers, μ-windows</text>
     <rect class="d-box" x="230" y="132" width="180" height="56" rx="10"/>
     <text class="d-text d-head" x="320" y="156" text-anchor="middle">Grand-canonical EMC</text>
     <text class="d-muted d-small" x="320" y="174" text-anchor="middle">GCEMC</text>
     <rect class="d-box" x="450" y="132" width="180" height="56" rx="10"/>
     <text class="d-text d-head" x="540" y="156" text-anchor="middle">Analyze</text>
     <text class="d-muted d-small" x="540" y="174" text-anchor="middle">expectations, free energies</text>
     <rect class="d-box" x="700" y="132" width="190" height="56" rx="10"/>
     <text class="d-text d-head" x="795" y="156" text-anchor="middle">Plot</text>
     <text class="d-muted d-small" x="795" y="174" text-anchor="middle">surface phase diagrams</text>
     <path class="d-line" d="M192 160 L226 160" marker-end="url(#tu-a)"/>
     <path class="d-line" d="M412 160 L446 160" marker-end="url(#tu-a)"/>
     <path class="d-line" d="M632 160 L696 160" marker-end="url(#tu-a)"/>
     <!-- bulk result feeds the surface run and its analysis -->
     <path class="d-accent-violet" d="M540 88 L540 110 L320 110 L320 128" marker-end="url(#tu-a)"/>
     <path class="d-accent-violet" d="M540 110 L540 128" marker-end="url(#tu-a)"/>
     <text class="d-text d-small" x="552" y="106">metal chemical-potential differences</text>
   </svg>
   </div>

The canonical simulations stand on their own: they need no reservoir and
feed nothing into the surface simulations.

Files of the examples
---------------------

Every example folder has the Python scripts and their command-line
counterpart side by side:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Step
     - Python
     - Command line
   * - Structure generation
     - ``generate_initial_population*.py``
     - ``catemc generate config*.yaml``
   * - Simulation
     - ``geemc_run.py``, ``gcemc_run*.py``
     - ``catemc run config*.yaml``
   * - Analysis
     - ``mbar_analysis*.py``
     - ``catemc analyze config*.yaml``
   * - All three
     -
     - ``run_cli*.sh``

.. note::

   The examples refer to fine-tuned MACE models by a path such as
   ``/mnt/nfs/emc_tutorial/.../CoMoFeNiCuNH_mace.model``. Replace it by the
   path of your own model, or by any other ASE calculator, before you run
   them. To try the workflow without a model, start with the
   :doc:`quickstart`.
