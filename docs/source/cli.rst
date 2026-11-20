Command line
============

The ``catemc`` command runs a complete study from one YAML file. The file
describes the system: composition, lattice, population, calculators and
sampling settings. Each command performs one stage on it.

.. raw:: html

   <div class="catemc-diagram">
   <svg viewBox="0 0 900 96" width="900" role="img"
        aria-label="The five commands in order: init writes a configuration file, check shows what it resolves to, generate builds the initial population, run performs the simulation, analyze evaluates it">
     <defs>
       <marker id="cl-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
         <path d="M0 0 L10 5 L0 10 z" class="d-arrow"/>
       </marker>
     </defs>
     <rect class="d-box" x="6" y="20" width="150" height="56" rx="10"/>
     <text class="d-text d-head d-mono" x="81" y="44" text-anchor="middle">init</text>
     <text class="d-muted d-small" x="81" y="62" text-anchor="middle">write config.yaml</text>
     <rect class="d-box" x="190" y="20" width="150" height="56" rx="10"/>
     <text class="d-text d-head d-mono" x="265" y="44" text-anchor="middle">check</text>
     <text class="d-muted d-small" x="265" y="62" text-anchor="middle">validate, no run</text>
     <rect class="d-box" x="374" y="20" width="150" height="56" rx="10"/>
     <text class="d-text d-head d-mono" x="449" y="44" text-anchor="middle">generate</text>
     <text class="d-muted d-small" x="449" y="62" text-anchor="middle">initial population</text>
     <rect class="d-box" x="558" y="20" width="150" height="56" rx="10"/>
     <text class="d-text d-head d-mono" x="633" y="44" text-anchor="middle">run</text>
     <text class="d-muted d-small" x="633" y="62" text-anchor="middle">EMC simulation</text>
     <rect class="d-box" x="742" y="20" width="150" height="56" rx="10"/>
     <text class="d-text d-head d-mono" x="817" y="44" text-anchor="middle">analyze</text>
     <text class="d-muted d-small" x="817" y="62" text-anchor="middle">MBAR reweighting</text>
     <path class="d-line" d="M158 48 L186 48" marker-end="url(#cl-a)"/>
     <path class="d-line" d="M342 48 L370 48" marker-end="url(#cl-a)"/>
     <path class="d-line" d="M526 48 L554 48" marker-end="url(#cl-a)"/>
     <path class="d-line" d="M710 48 L738 48" marker-end="url(#cl-a)"/>
   </svg>
   </div>

.. code-block:: console

   $ catemc init gibbs                 # write a commented config.yaml
   $ catemc check config.yaml          # show what the file resolves to
   $ catemc generate config.yaml       # build the initial population
   $ catemc run config.yaml            # run, or continue, the simulation
   $ catemc analyze config.yaml        # post-simulation analysis

The command line is a thin layer on the Python classes. The YAML file is
translated into the arguments of
:func:`~catemc.generate.generate_population`, :class:`~catemc.emc.EMC`,
:class:`~catemc.geemc.GEEMC`, :class:`~catemc.gcemc.GCEMC` and the analysis
classes, so a command-line run and the equivalent Python script do the
same thing and write the same files. The two can be mixed: generate a
population with a script and run it from the command line, or analyse a
command-line run in a notebook.

``python -m catemc`` is the same as ``catemc``, for environments in which
the console script is not on the ``PATH``.

Commands
--------

catemc init
^^^^^^^^^^^

.. code-block:: console

   $ catemc init [canonical|gibbs|grand_canonical] [-o FILE] [-f]

Writes a commented configuration file to start from. The default ensemble
is ``gibbs`` and the default file ``config.yaml``. ``-f`` overwrites an
existing file.

The ``canonical`` and ``gibbs`` files describe a 32-atom CuNiPdPtAu alloy
with the EMT calculator of ASE and run in about a minute, so they double
as a test of the installation (:doc:`quickstart`). The
``grand_canonical`` file describes a CuNiPd(111) surface in contact with
N and H reservoirs and expects a machine-learning potential.

catemc check
^^^^^^^^^^^^

.. code-block:: console

   $ catemc check config.yaml
   System                   : CuNiPdPtAu_bulk
   Directory                : /home/user/quickstart
   Ensemble                 : gibbs (GEEMC)
   Metals                   : {'Cu': 0.25, 'Ni': 0.25, 'Pd': 0.2, 'Pt': 0.2, 'Au': 0.1}
   Non-metals               : None
   Temperatures (K)         : [600, 700, 800, 900]
   Population size          : 8
   Template                 : Cu32 (32 sites; built from the config)
   fix_indices              : none
   mutable_metal_indices    : all
   mutable_nonmetal_indices : all
   Calculators              : {'stages': [{'calculator': 'emt'}]}
   Fitness (ASE units)      : {'md_timestep': 0.196, 'md_taut': 9.82, 'md_steps': 20, ...}
   EMC settings             : {'metal_interswap_rate': 0.8, 'intraswap_rate': 0.1, ...}
   Run settings             : {'total_steps': 100}
   Population file          : /home/user/quickstart/CuNiPdPtAu_bulk_input_pop.traj  (not generated yet)

Validates the file and prints what CatEMC derives from it, without loading
a calculator or starting a simulation. Use it after every edit. It shows

* the two sublattices the composition was split into,
* the template that will be decorated and how many atoms are fixed, mutable
  and part of the slab,
* the fitness settings converted to ASE units, exactly as the Python
  classes receive them,
* for a surface run, the accessible volume, the number of reaction
  conditions, the reference energies, the internal partition functions
  and whether the metal chemical potentials of the bulk run are available
  at all temperatures.

catemc generate
^^^^^^^^^^^^^^^

.. code-block:: console

   $ catemc generate config.yaml [-f]

Builds the template lattice and the initial population and writes
``{name}_template.traj`` and ``{name}_input_pop.traj`` to the calculation
directory (:doc:`input_populations`). The command prints the random seed
and one line per replica. An existing population is kept unless ``-f`` is
given.

Skip this command when ``population.file`` points to a population you
built yourself.

catemc run
^^^^^^^^^^

.. code-block:: console

   $ catemc run config.yaml

Starts the worker processes and runs the simulation until
``run.total_steps`` is reached (:doc:`emc`). If the databases of an earlier
run exist, the simulation continues from the last completed step. The same
command therefore starts a simulation, extends it and resumes a job that
hit its time limit:

.. code-block:: console

   $ catemc run config.yaml -s run.total_steps=4000

catemc analyze
^^^^^^^^^^^^^^

.. code-block:: console

   $ catemc analyze config.yaml [--fresh]

Runs the tasks listed in the ``analysis`` section (:doc:`analysis`). The
samples collected from the databases are cached in
``{name}_mbar_*.pkl`` and reused by the next call. ``--fresh`` collects
them again, which is needed after the simulation was extended.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Ensemble
     - Tasks
     - Result
   * - canonical, gibbs
     - ``trajectory``, ``diagnostics``, ``sro``, ``dmu``
     - replica trajectories, sampling diagnostics, Warren-Cowley
       parameters, chemical-potential differences of the metals
   * - grand_canonical
     - ``trajectory``, ``diagnostics``, ``expectations``, ``fes``
     - replica trajectories, sampling diagnostics, surface phase diagrams,
       free energies of the sampled structures

Replacing entries on the command line
-------------------------------------

``-s section.key=value`` replaces one entry of the file for this call. It
can be given several times, and the value is read as YAML, so lists and
mappings work:

.. code-block:: console

   $ catemc run config.yaml -s run.total_steps=200
   $ catemc run config.yaml -s emc.gpu_devices=[0,1] -s emc.procs_per_device=16
   $ catemc generate config.yaml -s composition=Co0.3Mo0.4Fe0.1Ni0.1Cu0.1 -s name=test -f
   $ catemc analyze config.yaml -s analysis.starting_step=1000

The file on disk is not changed. For a series of compositions or
temperatures, keep one file and loop over ``-s`` in a shell script:

.. code-block:: bash

   for x in 0.35 0.45 0.55; do
       y=$(python -c "print(round(0.7 - $x, 2))")
       name=Co${y}Mo${x}Fe0.1Ni0.1Cu0.1_bulk
       catemc generate config.yaml -s composition=Co${y}Mo${x}Fe0.1Ni0.1Cu0.1 -s name=$name
       catemc run      config.yaml -s composition=Co${y}Mo${x}Fe0.1Ni0.1Cu0.1 -s name=$name
       catemc analyze  config.yaml -s composition=Co${y}Mo${x}Fe0.1Ni0.1Cu0.1 -s name=$name
   done

Units, paths and names
----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Convention
     - Rule
   * - Units
     - temperatures in K, energies and chemical potentials in eV, lengths
       in Å, times in fs, pressures in bar. The Python classes take ASE
       units instead (``1 * units.fs``, ``1.01325 * units.bar``). The
       command line converts.
   * - Paths
     - relative paths are relative to the configuration file, not to the
       folder the command is called from. ``~`` and environment variables
       such as ``$SCRATCH`` are expanded.
   * - Output files
     - every file starts with ``name`` and is written to ``directory``.
   * - Unknown keys
     - are an error, with a suggestion for the closest valid key. A typing
       mistake can therefore not silently fall back to a default.

.. code-block:: console

   $ catemc check config.yaml
   catemc: error: Unknown key 'intraswap_rte' in emc. Did you mean 'intraswap_rate'? Allowed keys: ...

The command returns 0 on success and 2 for an invalid configuration, so a
job script that uses ``set -e`` stops at the first problem.

Using your own potential
------------------------

The ``calculators`` section lists the calculators that evaluate a
candidate, in the order in which they are used: cheap pre-relaxation
first, the production model last. ``calculator`` is one of the short names
``repulsive``, ``emt`` and ``mace``, or the import path of any ASE
calculator. ``fmax`` is the force threshold of the relaxation with this
calculator, ``args`` an optional list of positional arguments, and all
other entries are passed to the calculator as keyword arguments.

.. tab-set::

   .. tab-item:: MACE

      .. code-block:: yaml

         calculators:
           - calculator: repulsive       # catemc.repulsive.RepulsivePotential
             fmax: 0.5
           - calculator: mace            # mace.calculators.mace.MACECalculator
             model_paths: /path/to/model
             device: cuda
             default_dtype: float64
             fmax: 0.05

   .. tab-item:: Any ASE calculator

      .. code-block:: yaml

         calculators:
           - calculator: ase.calculators.lj.LennardJones
             epsilon: 0.01
             sigma: 2.5
             fmax: 0.05

      Any class that can be imported works, including the calculators of
      other machine-learning potentials. The path can also name a
      function or a class method that returns a calculator, written as
      ``package.module.Class.method`` or ``package.module:function``.

   .. tab-item:: Your own function

      When the calculator needs more than keyword arguments, write a
      function that returns one calculator or a list of them:

      .. code-block:: python

         # my_calculators.py, next to config.yaml
         def create_calculators(model, device='cuda'):
             from mace.calculators.mace import MACECalculator
             from catemc.repulsive import RepulsivePotential
             return [RepulsivePotential(),
                     MACECalculator(model_paths=model, device=device,
                                    default_dtype='float64')]

      .. code-block:: yaml

         calculators:
           factory: my_calculators.py:create_calculators
           kwargs: {model: /path/to/model}
           fmax: [0.5, 0.05]

      ``factory`` is ``file.py:function`` or ``package.module:function``.

Every worker process creates its calculators once and keeps them for the
whole run (:doc:`accelerated_workflow`). An argument that starts with
``./`` or ``../`` is taken as a path relative to the configuration file.

Converting a Python script
--------------------------

The examples contain both versions of every run, which makes the
correspondence easy to see.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Python
     - YAML
   * - ``GEEMC(name, calculation_dir=...)``
     - ``name``, ``directory``, ``ensemble: gibbs``
   * - ``target_metal_species`` and ``target_metal_composition`` of
       ``generate_population``, ``metal_element_pool`` and
       ``nonmetal_element_pool`` of the EMC class
     - ``composition``
   * - the template passed to ``generate_population``
     - ``template``
   * - ``fix_indices``, ``mutable_metal_indices``,
       ``mutable_nonmetal_indices``, ``slab_indices``
     - ``regions``
   * - ``pop_size``, the temperature loop, ``assign_mu_windows``
     - ``population``
   * - ``create_calculators()``
     - ``calculators``
   * - ``get_fitness_kwargs()``
     - ``fitness`` (``fmax_list`` comes from the ``fmax`` entries)
   * - all other keyword arguments of the EMC class
     - ``emc``, with the same names
   * - ``emc.run(total_steps=...)``
     - ``run``
   * - ``adsorbate_species``, ``other_adsorbate_species``, ``volume``,
       ``temperature_species_internal_map``, ``adsorption_sites``
     - ``adsorbates``
   * - ``reaction_conditions``, ``element_column_map``,
       ``element_eref_map``, ``element_y2x_map``, ``temperature_dmu_map``
     - ``chemical_potentials``
   * - the calls of the analysis class
     - ``analysis``

:doc:`config_reference` lists every key.

In a job script
---------------

.. code-block:: bash

   #!/bin/bash
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=32
   #SBATCH --time=24:00:00
   set -e

   export OMP_NUM_THREADS=1            # one thread per worker process

   [ -f Co0.25Mo0.45Fe0.1Ni0.1Cu0.1_bulk_input_pop.traj ] || catemc generate config.yaml
   catemc run config.yaml -s emc.scratch_dir=$TMPDIR/catemc -s emc.scratch_sync_interval=50
   catemc analyze config.yaml --fresh

Because ``catemc run`` continues an existing simulation, the same script
can be submitted again until ``run.total_steps`` is reached.
``scratch_sync_interval`` copies the live database from the node-local
disk back to the calculation directory, so that a job that is killed
loses at most that many steps (:doc:`accelerated_workflow`).
