:hide-toc:

CatEMC
======

.. raw:: html

   <div class="catemc-hero">
     <h1>CatEMC</h1>
     <p>Evolutionary Monte Carlo for compositionally complex bulk materials and
     catalyst surfaces under reaction conditions. One population of replicas,
     three ensembles, and MBAR reweighting to any temperature and gas-phase
     chemical potential.</p>
     <div class="catemc-hero-buttons">
       <a class="catemc-btn primary" href="quickstart.html">Quickstart</a>
       <a class="catemc-btn" href="tutorials.html">Tutorials</a>
       <a class="catemc-btn" href="cli.html">Command line</a>
       <a class="catemc-btn" href="https://github.com/basf/CatEMC">GitHub</a>
     </div>
   </div>

**CatEMC** is a Python package for evolutionary Monte Carlo (EMC) simulations
of compositionally complex materials such as high-entropy alloys and
high-entropy oxides. A *population* of replicas at several temperatures is
evolved with Monte Carlo moves that act on one replica (swaps, mutations,
adsorbate insertion and removal) or on two replicas at once (inter-replica
swaps, crossover, replica exchange). Every candidate is relaxed, and
periodically propagated by molecular dynamics, with a machine-learning
interatomic potential, so that the sampling is both fast and off-lattice.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Canonical ensemble
      :link: canonical_emc
      :link-type: doc
      :class-card: sd-card-hover

      Fixed composition. Find the equilibrium arrangement and short-range
      order of a bulk material at several temperatures at once.

   .. grid-item-card:: Gibbs ensemble
      :link: bulk_geemc
      :link-type: doc
      :class-card: sd-card-hover

      Replicas exchange atoms. One run gives the chemical-potential
      differences between all metals as a function of temperature.

   .. grid-item-card:: Grand-canonical ensemble
      :link: surface_gcemc
      :link-type: doc
      :class-card: sd-card-hover

      A surface open to bulk and gas reservoirs. One run maps surface
      composition and adsorbate coverage over a grid of reaction conditions.

How a study is organized
------------------------

Every CatEMC study has the same three stages. Each of them is one function
or class in Python and one command on the command line.

.. raw:: html

   <div class="catemc-diagram">
   <svg viewBox="0 0 900 250" width="900" role="img"
        aria-label="CatEMC workflow: structure generation, EMC simulation, post-simulation analysis">
     <defs>
       <marker id="wf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
         <path d="M0 0 L10 5 L0 10 z" class="d-arrow"/>
       </marker>
     </defs>
     <!-- stage 1 -->
     <rect class="d-box" x="10" y="20" width="260" height="210" rx="12"/>
     <rect class="d-blue" x="10" y="20" width="260" height="6" rx="3"/>
     <text class="d-text d-head" x="30" y="56">1 · Structure generation</text>
     <text class="d-muted d-small" x="30" y="78">template structure, random decorations,</text>
     <text class="d-muted d-small" x="30" y="94">temperatures and μ-windows</text>
     <g transform="translate(40 112)">
       <circle class="d-blue" cx="10" cy="14" r="9"/><circle class="d-orange" cx="30" cy="14" r="9"/>
       <circle class="d-aqua" cx="50" cy="14" r="9"/><circle class="d-blue" cx="70" cy="14" r="9"/>
       <circle class="d-orange" cx="20" cy="32" r="9"/><circle class="d-blue" cx="40" cy="32" r="9"/>
       <circle class="d-violet" cx="60" cy="32" r="9"/><circle class="d-aqua" cx="80" cy="32" r="9"/>
       <circle class="d-aqua" cx="130" cy="14" r="9"/><circle class="d-blue" cx="150" cy="14" r="9"/>
       <circle class="d-violet" cx="170" cy="14" r="9"/><circle class="d-orange" cx="190" cy="14" r="9"/>
       <circle class="d-blue" cx="140" cy="32" r="9"/><circle class="d-aqua" cx="160" cy="32" r="9"/>
       <circle class="d-orange" cx="180" cy="32" r="9"/><circle class="d-blue" cx="200" cy="32" r="9"/>
     </g>
     <text class="d-muted d-mono" x="30" y="186">generate_population()</text>
     <text class="d-text d-mono" x="30" y="210">$ catemc generate config.yaml</text>
     <!-- stage 2 -->
     <rect class="d-box" x="320" y="20" width="260" height="210" rx="12"/>
     <rect class="d-violet" x="320" y="20" width="260" height="6" rx="3"/>
     <text class="d-text d-head" x="340" y="56">2 · EMC simulation</text>
     <text class="d-muted d-small" x="340" y="78">MC moves, relaxation, MD and replica</text>
     <text class="d-muted d-small" x="340" y="94">exchange in one of three ensembles</text>
     <g transform="translate(340 108)">
       <rect class="d-accent-blue" x="0" y="0" width="68" height="48" rx="8"/>
       <text class="d-text d-small" x="34" y="21" text-anchor="middle">canonical</text>
       <text class="d-muted d-small" x="34" y="38" text-anchor="middle">EMC</text>
       <rect class="d-accent-violet" x="76" y="0" width="68" height="48" rx="8"/>
       <text class="d-text d-small" x="110" y="21" text-anchor="middle">Gibbs</text>
       <text class="d-muted d-small" x="110" y="38" text-anchor="middle">GEEMC</text>
       <rect class="d-accent-aqua" x="152" y="0" width="68" height="48" rx="8"/>
       <text class="d-text d-small" x="186" y="21" text-anchor="middle">grand can.</text>
       <text class="d-muted d-small" x="186" y="38" text-anchor="middle">GCEMC</text>
     </g>
     <text class="d-muted d-mono" x="340" y="186">EMC / GEEMC / GCEMC .run()</text>
     <text class="d-text d-mono" x="340" y="210">$ catemc run config.yaml</text>
     <!-- stage 3 -->
     <rect class="d-box" x="630" y="20" width="260" height="210" rx="12"/>
     <rect class="d-aqua" x="630" y="20" width="260" height="6" rx="3"/>
     <text class="d-text d-head" x="650" y="56">3 · Post-simulation analysis</text>
     <text class="d-muted d-small" x="650" y="78">MBAR reweighting: Δμ, short-range order,</text>
     <text class="d-muted d-small" x="650" y="94">coverages, free-energy surfaces</text>
     <g transform="translate(655 106)">
       <path class="d-line" d="M0 52 L210 52 M0 52 L0 0"/>
       <path class="d-accent-blue" d="M4 40 C40 36 70 30 100 22 S170 10 206 6"/>
       <path class="d-accent-violet" d="M4 16 C50 20 90 30 120 36 S180 44 206 46"/>
     </g>
     <text class="d-muted d-mono" x="650" y="186">EMC / GEEMC / GCEMC Analysis</text>
     <text class="d-text d-mono" x="650" y="210">$ catemc analyze config.yaml</text>
     <!-- arrows -->
     <path class="d-line" d="M274 125 L314 125" marker-end="url(#wf-arrow)"/>
     <path class="d-line" d="M584 125 L624 125" marker-end="url(#wf-arrow)"/>
   </svg>
   </div>

A bulk Gibbs-ensemble run usually comes first, because it provides the metal
chemical potentials that a surface grand-canonical run needs as its bulk
reservoir.

Two ways to run
---------------

CatEMC can be driven from Python input files or from the ``catemc`` command
with one YAML file per system. Both do exactly the same thing: the command
only translates the YAML file into the arguments of the Python classes.

.. tab-set::

   .. tab-item:: Command line

      .. code-block:: console

         $ catemc init gibbs -o config.yaml
         $ catemc generate config.yaml
         $ catemc run config.yaml
         $ catemc analyze config.yaml

      .. code-block:: yaml

         name: Co0.25Mo0.45Fe0.1Ni0.1Cu0.1_bulk
         ensemble: gibbs
         composition: Co0.25Mo0.45Fe0.1Ni0.1Cu0.1
         population:
           template: fcc_8x8x8.traj        # any structure file that ASE reads
           temperatures: [573, 650, 723, 800, 873, 950, 1023, 1100, 1173, 1250, 1323, 1400]
           replicas_per_temperature: 4

   .. tab-item:: Python

      .. code-block:: python

         from catemc.generate import generate_population
         from catemc.geemc import GEEMC
         from catemc.analysis import GEEMCAnalysis

         input_pop = generate_population(template, pop_size=48, ...)
         emc = GEEMC(name, calculation_dir, input_pop, create_calculators, ...)
         emc.run(total_steps=5000)

         ana = GEEMCAnalysis(name, calculation_dir, pop_size=48, ...)
         ana.prepare_data()
         ana.get_dmu_estimates(target_temperatures=[573, 673, 773, 873])

Where to go next
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc
      :class-card: sd-card-hover

      Install CatEMC, its dependencies and a calculator.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc
      :class-card: sd-card-hover

      A complete Gibbs-ensemble run in about a minute, without a GPU.

   .. grid-item-card:: Method
      :link: concepts
      :link-type: doc
      :class-card: sd-card-hover

      The ideas behind EMC, the three ensembles and the reweighting.

   .. grid-item-card:: Tutorials
      :link: tutorials
      :link-type: doc
      :class-card: sd-card-hover

      Structure generation, EMC simulations and post-simulation analyses
      for a high-entropy alloy and a high-entropy oxide.

   .. grid-item-card:: Command-line interface
      :link: cli
      :link-type: doc
      :class-card: sd-card-hover

      The ``catemc`` commands and every key of the YAML configuration file.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc
      :class-card: sd-card-hover

      The classes and functions of the Python package.

.. toctree::
   :hidden:
   :caption: Getting started

   installation
   quickstart
   concepts

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials
   input_populations
   emc
   analysis
   visualization
   examples

.. toctree::
   :hidden:
   :caption: Reference

   cli
   config_reference
   accelerated_workflow
   api

.. toctree::
   :hidden:
   :caption: Project

   GitHub repository <https://github.com/basf/CatEMC>
