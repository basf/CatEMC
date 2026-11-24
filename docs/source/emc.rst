EMC simulations
===============

CatEMC has one simulation class per ensemble. They share the population,
the worker processes, the relaxation and MD machinery and the restart
logic, and differ in the Monte Carlo moves and in the acceptance
probabilities. This page explains what is common. The three pages below
then treat one ensemble each, with its theory and the two example
catalysts.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Canonical ensemble
      :link: canonical_emc
      :link-type: doc
      :class-card: sd-card-hover

      :class:`~catemc.emc.EMC` · ``ensemble: canonical``

      Fixed composition. HEA and HEO bulk.

   .. grid-item-card:: Gibbs ensemble
      :link: bulk_geemc
      :link-type: doc
      :class-card: sd-card-hover

      :class:`~catemc.geemc.GEEMC` · ``ensemble: gibbs``

      Chemical-potential differences. HEA and HEO bulk.

   .. grid-item-card:: Grand-canonical ensemble
      :link: surface_gcemc
      :link-type: doc
      :class-card: sd-card-hover

      :class:`~catemc.gcemc.GCEMC` · ``ensemble: grand_canonical``

      Surfaces under reaction conditions. HEA(111) and HEO(311).

.. toctree::
   :hidden:

   canonical_emc
   bulk_geemc
   surface_gcemc

Anatomy of a simulation
-----------------------

A simulation needs four things: the initial population, the calculators,
the settings of the relaxation and MD (the *fitness* evaluation), and the
settings of the EMC itself.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      One YAML file holds all four, and the same file is used for
      structure generation and analysis:

      .. code-block:: yaml

         name: my_system
         ensemble: gibbs            # canonical, gibbs or grand_canonical
         composition: ...           # see "Structure generation"
         population: ...            # template structure, temperatures, replicas

         calculators:               # one entry per calculator
           - calculator: mace_mp    # MACE-MPA-0 foundation model
             model: medium-mpa-0
             device: cuda
             default_dtype: float64
             fmax: 0.05

         fitness:                   # relaxation and MD settings
           md_timestep: 1           # fs
           md_taut: 500             # fs
           md_steps: 400
           neighbor_cutoff: 3.5

         emc:                       # arguments of the simulation class
           metal_interswap_rate: 0.8
           intraswap_rate: 0.1
           crossover_rate: 0.1
           gpu_devices: [0]
           procs_per_device: 32

         run:
           total_steps: 5000

      .. code-block:: console

         $ catemc run config.yaml

   .. tab-item:: Python
      :sync: python

      One input file with three parts:

      .. code-block:: python

         from catemc.geemc import GEEMC
         from ase import units
         from ase.io import read


         def create_calculators():
             # 1) Called once in every worker. Import the calculator here.
             from mace.calculators import mace_mp
             return [mace_mp(model='medium-mpa-0',
                             device='cuda', default_dtype='float64')]


         def get_fitness_kwargs():
             # 2) Relaxation and MD settings, in ASE units
             return dict(fmax_list=[0.05],
                         md_timestep=1 * units.fs,
                         md_taut=500 * units.fs,
                         md_steps=400,
                         md_write_interval=10,
                         neighbor_cutoff=3.5)


         def main():
             # 3) The simulation
             input_pop = read('my_system_input_pop.traj', index=':')
             emc = GEEMC('my_system',
                         calculation_dir='.',
                         input_pop=input_pop,
                         calculator_factory=create_calculators,
                         fitness_kwargs=get_fitness_kwargs(),
                         metal_element_pool=['Co', 'Mo', 'Fe', 'Ni', 'Cu'],
                         metal_interswap_rate=0.8,
                         intraswap_rate=0.1,
                         crossover_rate=0.1,
                         gpu_devices=[0],
                         procs_per_device=32)
             emc.run(total_steps=5000)


         if __name__ == '__main__':
             main()

      .. code-block:: console

         $ python -u emc_run.py

      .. important::

         ``create_calculators`` must be defined at the top level of the
         file, and the executable code must be inside
         ``if __name__ == '__main__':``. The worker processes are started
         with the ``spawn`` method and import the input file again.

The command line is a translation layer and nothing more: the ``emc``
section is passed to the constructor of the class under the same argument
names, ``run`` to its ``run`` method, and ``fitness`` to the fitness
function. Everything written about the Python arguments on the following
pages therefore applies to the YAML keys as well.

Calculators
^^^^^^^^^^^

``create_calculators`` returns one ASE calculator or a list of calculators.
A list is applied in sequence, each one relaxing the structure to its own
force threshold in ``fmax_list``. The last calculator also runs the MD and
provides the energy that enters the acceptance probability.

This is how a cheap pre-relaxation is set up. Structures that come out of a
crossover or an insertion can contain atoms that are too close to each
other, which a machine-learning potential may handle badly. A purely
repulsive potential removes such contacts first:

.. math::
   :label: eq-repulsive

   U_\mathrm{rep}(\mathbf r) = U_\mathrm{c} + \sum_{i<j}
   \left(0.7\,\frac{r_i + r_j}{\lVert \mathbf r_i - \mathbf r_j \rVert_2}\right)^{12}

where :math:`r_i` and :math:`r_j` are covalent radii and
:math:`U_\mathrm{c}` is a constant.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: yaml

         calculators:
           - calculator: repulsive      # pre-relaxation to 0.5 eV/A
             fmax: 0.5
           - calculator: mace_mp        # final relaxation, MD and energy
             model: medium-mpa-0
             device: cuda
             default_dtype: float64
             fmax: 0.05

      ``calculator`` is one of the short names ``repulsive``, ``emt``,
      ``mace`` and ``mace_mp``, or the import path of any ASE calculator
      such as ``mace.calculators.mace_mp``. All other keys are passed to
      the calculator. ``fmax: null`` evaluates a calculator without relaxing
      with it.

      When a calculator needs more preparation than a constructor call,
      write the factory in a Python file and point to it:

      .. code-block:: yaml

         calculators:
           calculator_factory: my_calculators.py:create_calculators
           calculator_factory_kwargs: {device: cuda}
           fmax: [0.5, 0.05]

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         def create_calculators():
             from catemc.repulsive import RepulsivePotential
             from mace.calculators import mace_mp
             return [
                 RepulsivePotential(),
                 mace_mp(model='medium-mpa-0', device='cuda',
                         default_dtype='float64'),
             ]

         def get_fitness_kwargs():
             return dict(fmax_list=[0.5, 0.05], ...)

Relaxation and MD settings
^^^^^^^^^^^^^^^^^^^^^^^^^^

The fitness function relaxes a candidate and, every ``md_generation``
steps, runs MD before the relaxation. Bulk simulations use
:func:`~catemc.calculation_templates.calculate_bulk_fitness` and surface
simulations :func:`~catemc.calculation_templates.calculate_surface_fitness`.

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Setting
     - Default
     - Meaning
   * - ``fmax_list``
     -
     - force threshold of each calculator in eV/Å (``fmax`` of the
       calculator entries on the command line)
   * - ``md_timestep``
     - 1 fs
     - MD time step
   * - ``md_taut``
     - 500 fs
     - time constant of the thermostat
   * - ``md_steps``
     - 400
     - MD steps per MD run
   * - ``md_type``
     - ``berendsen``
     - thermostat: ``berendsen`` or ``langevin`` (with ``md_friction``)
   * - ``md_taup``, ``compressibility``
     - 1000 fs, 5·10\ :sup:`-5` bar\ :sup:`-1`
     - barostat settings of constant-pressure bulk simulations
   * - ``optimizer``
     - ``SQNM``
     - ``SQNM`` or any optimizer of ``ase.optimize``, e.g. ``LBFGS``
   * - ``max_relax_steps``
     - 300
     - largest number of optimizer steps. A relaxation that does not
       converge marks the candidate as failed, and it is rejected.
   * - ``optimize_cell``
     - ``False``
     - relax the cell together with the atoms (bulk, with ``pressure``)
   * - ``neighbor_cutoff``
     - 2.5 Å
     - cutoff used to put vacancies back after the atoms have moved
   * - ``update_md_positions``
     - ``True``
     - continue from the coordinates after MD
   * - ``update_relaxed_positions``
     - ``False``
     - keep the relaxed coordinates. By default the relaxation is only
       used to evaluate the energy, and MD carries the coordinates forward.

In Python the times are given in ASE units (``1 * units.fs``). In the YAML
file they are plain numbers in fs, and the defaults in the table apply.

Moves and rates
^^^^^^^^^^^^^^^

In every step, each replica is assigned one move, chosen with a
probability proportional to its *rate*. The rates are normalized, so only
their ratios matter.

.. list-table::
   :header-rows: 1
   :widths: 34 26 10 10 20

   * - Move
     - Rate argument
     - Canonical
     - Gibbs
     - Grand canonical
   * - Intra-swap: swap two unlike atoms (or an atom and a vacancy) within
       a replica
     - ``intraswap_rate``, ``slab_intraswap_rate``
     - ✓
     - ✓
     - ✓
   * - Cut-and-splice crossover between two replicas
     - ``crossover_rate``
     - ✓
     - ✓
     - ✓ (surface with adlayer)
   * - Inter-swap: swap unlike atoms between two replicas at the same
       temperature
     - ``metal_interswap_rate``, ``nonmetal_interswap_rate``,
       ``interswap_rate``
     -
     - ✓
     - ✓
   * - Mutation of a surface atom
     - ``slab_mutation_rate``
     -
     -
     - ✓
   * - Fragment insertion, removal, displacement, mutation
     - ``adsorbate_insertion_rate``, ``adsorbate_removal_rate``,
       ``adsorbate_displacement_rate``, ``adsorbate_mutation_rate``
     -
     -
     - ✓
   * - Replica exchange
     - every ``exchange_generation`` steps
     - ✓
     - ✓
     - ✓
   * - μ-walk
     - every ``mu_walk_generation`` steps
     -
     -
     - ✓

General settings
^^^^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Argument
     - Default
     - Meaning
   * - ``name``, ``calculation_dir``
     -
     - prefix and folder of all output files
   * - ``input_pop``
     -
     - the initial population
   * - ``calculator_factory``, ``calculator_factory_kwargs``
     -
     - the function that creates the calculators in each worker
   * - ``fitness_kwargs``
     -
     - relaxation and MD settings (previous section)
   * - ``fix_indices``
     - ``None``
     - atoms that are frozen
   * - ``metal_element_pool``, ``nonmetal_element_pool``
     - ``None``
     - the species of the two sublattices
   * - ``mutable_metal_indices``, ``mutable_nonmetal_indices``
     - ``None``
     - sites the moves may act on (default: all)
   * - ``pressure``
     - ``None``
     - target pressure of a bulk simulation. ``None`` keeps the volume
       fixed.
   * - ``md_generation``
     - 10
     - run MD every this many steps
   * - ``exchange_generation``
     - 1
     - attempt replica exchange every this many steps
   * - ``trim_generation``
     - 100
     - move older steps to a backup database every this many steps
   * - ``gpu_devices``, ``procs_per_device``
     - ``None``, 1
     - devices and number of worker processes per device
   * - ``scratch_dir``, ``scratch_sync_interval``
     - ``None``
     - keep the live database on a scratch disk (:doc:`accelerated_workflow`)
   * - ``traj_write_interval``
     - 1
     - write the replica trajectories every this many steps
   * - ``verbose``
     - ``True``
     - print every move and not only the summary of each step

Run, continue and restart
^^^^^^^^^^^^^^^^^^^^^^^^^

``run(total_steps=N)`` runs until step :math:`N` is reached. If the
databases of an earlier run exist, the simulation continues from the last
completed step, so the same command serves to start a simulation, to
extend it and to recover from a job that hit its time limit.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: console

         $ catemc run config.yaml                          # start
         $ catemc run config.yaml                          # continue
         $ catemc run config.yaml -s run.total_steps=8000  # extend

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         emc.run(total_steps=5000)     # start, or continue up to step 5000

During the run the folder fills with these files:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Content
   * - ``{name}.db``
     - the live ASE database with every evaluated structure, accepted or
       not
   * - ``{name}_bak0.db``, ``{name}_bak1.db``, ...
     - one backup per ``trim_generation`` steps. The analysis reads these.
   * - ``{name}.log``
     - the members of the population after every step
   * - ``tmp_{name}/``
     - working files of the workers

The standard output reports each step:

.. code-block:: text

   Generation 10: acceptance 10/32 (0.312), best_raw_score 4810.01537109, elapsed 26.9 min

The *raw score* of a structure is :math:`-U` in the bulk ensembles and
:math:`-\Phi` (:eq:`eq-phi`) in the grand-canonical ensemble. A structure
whose evaluation failed receives ``raw_score_cutoff`` and is rejected.

The published simulations used cycles of one MD run (400 steps of 1 fs)
and ten EMC steps, that is ``md_generation: 10``:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - System
     - Replicas
     - EMC steps
     - Discarded as equilibration
   * - HEA and HEO bulk (Gibbs)
     - 48
     - 1000
     - 300
   * - HEA(111)
     - 28
     - 2000
     - 500
   * - HEO(311), H\ :sub:`2` reduction
     - 28
     - 4000
     - 500

Whether a run is long enough is a question for the analysis:
:doc:`analysis` shows the diagnostics that answer it.
