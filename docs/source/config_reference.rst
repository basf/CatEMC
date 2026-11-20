Configuration file
==================

This page lists every key of the YAML file that the ``catemc`` command
reads (:doc:`cli`). ``catemc init`` writes a commented file to start from,
and ``catemc check`` shows what a file resolves to.

Units: temperatures in K, energies and chemical potentials in eV, lengths
in Å, times in fs, pressures in bar. Relative paths are relative to the
configuration file.

Layout
------

.. code-block:: yaml

   name: MySystem                  # prefix of all output files
   directory: .                    # where the files are written
   ensemble: gibbs                 # canonical, gibbs or grand_canonical
   composition: Co0.25Mo0.45Fe0.1Ni0.1Cu0.1

   template: {...}                 # the lattice that is decorated
   regions: {...}                  # which atoms are fixed or mutable
   population: {...}               # temperatures and starting structures
   calculators: [...]              # the potentials
   fitness: {...}                  # relaxation and MD settings
   emc: {...}                      # moves, rates and resources
   run: {...}                      # number of steps

   adsorbates: {...}               # grand canonical only
   chemical_potentials: {...}      # grand canonical only

   analysis: {...}                 # post-simulation tasks

.. list-table::
   :header-rows: 1
   :widths: 26 14 14 14 32

   * - Section
     - Canonical
     - Gibbs
     - Grand canonical
     - Used by
   * - ``name``, ``directory``, ``ensemble``, ``composition``
     - required
     - required
     - required
     - all commands
   * - ``template``
     - required
     - required
     - required
     - ``generate`` (then read from ``{name}_template.traj``)
   * - ``regions``
     - optional
     - optional
     - usual
     - ``generate``, ``run``, ``analyze``
   * - ``population``
     - required
     - required
     - required
     - ``generate``, ``run``
   * - ``calculators``, ``fitness``, ``emc``, ``run``
     - required
     - required
     - required
     - ``run``
   * - ``adsorbates``, ``chemical_potentials``
     -
     -
     - required
     - ``generate``, ``run``, ``analyze``
   * - ``analysis``
     - optional
     - optional
     - optional
     - ``analyze``

A key that is not listed on this page is an error. The message names the
closest valid key.

System
------

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Key
     - Default
     - Meaning
   * - ``name``
     - required
     - system name. Every output file starts with it.
   * - ``directory``
     - ``.``
     - calculation directory. It is created when it does not exist.
   * - ``ensemble``
     - required
     - ``canonical`` (:class:`~catemc.emc.EMC`), ``gibbs``
       (:class:`~catemc.geemc.GEEMC`) or ``grand_canonical``
       (:class:`~catemc.gcemc.GCEMC`). The short forms ``emc``, ``geemc``
       and ``gcemc`` are accepted.
   * - ``composition``
     - required
     - composition of the material. For a surface run it is the
       composition of the bulk the surface is in contact with.

``composition`` takes any formula. Only the ratios matter, and ``X`` is a
vacancy:

.. code-block:: yaml

   composition: Co0.25Mo0.45Fe0.1Ni0.1Cu0.1                # alloy
   composition: Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5X0.5       # oxide with oxygen vacancies
   composition: Pt3Sn                                      # intermetallic
   composition: Ti0.5Zr0.5N                                # nitride

The formula is split into a metal and a non-metal sublattice. H, C, N, O,
F, P, S, Cl, Se, Br, I and the noble gases form the non-metal sublattice,
all other elements the metal sublattice. ``X`` belongs to the non-metal
sublattice when there is one. The amounts are normalized within each
sublattice, and the order of the elements in the formula is the order of
the species in all results.

For any other division, name the two sublattices explicitly. This is the
way to describe a boride or a silicide, or vacancies on the metal
sublattice of a compound:

.. code-block:: yaml

   composition:
     metals: {Mo: 0.5, W: 0.5}        # a formula such as Mo0.5W0.5 works as well
     nonmetals: {B: 1.8, X: 0.2}

``template``
------------

The template is the undecorated lattice: every site that an atom or a
vacancy of the composition can occupy. It comes from a structure file or
from a function of ``ase.build``.

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Key
     - Default
     - Meaning
   * - ``file``
     -
     - structure file in any format ASE reads (``.traj``, ``.cif``,
       ``POSCAR``, ...)
   * - ``index``
     - ``-1``
     - image of ``file`` to use
   * - ``build``
     -
     - mapping with ``function``, the name of a function of
       ``ase.build``, and its arguments
   * - ``scale_cell``
     -
     - three cell lengths or a full 3×3 cell. The atoms are scaled with
       the cell.
   * - ``repeat``
     -
     - three repetitions, for example ``[4, 4, 4]``
   * - ``cell_height``
     -
     - height of the cell along z. The slab is centred in it.
   * - ``metal_sites``
     -
     - elements of the template that mark metal sites although they are
       not part of the composition, for example ``[Co]`` in a CoAl\
       :sub:`2`\ O\ :sub:`4` file that is decorated with five other
       metals
   * - ``randomize``
     - ``false``
     - decorate the metal sites randomly at the composition, separately
       for the fixed and the free region. For slabs whose frozen layers
       should have the bulk composition.
   * - ``relax``
     - ``false``
     - relax the template with the last calculator before the population
       is generated. ``true`` or ``{fmax: 0.01, steps: 300, optimizer:
       LBFGS}``.

Exactly one of ``file`` and ``build`` is required. The other keys are
applied in the order of the table.

.. code-block:: yaml

   # fcc bulk, 4 x 4 x 4 cubic cells
   template:
     build: {function: bulk, name: Cu, crystalstructure: fcc, a: 3.6, cubic: true}
     repeat: [4, 4, 4]

   # fcc(111) slab
   template:
     build: {function: fcc111, symbol: Cu, a: 3.776, size: [6, 6, 6], vacuum: 10.0}

   # any surface of any crystal: cut it from a bulk file
   template:
     build:
       function: surface
       lattice: {file: spinel_conventional.cif}
       indices: [3, 1, 1]
       layers: 4
       vacuum: 10.0

An argument that is itself a mapping with ``function`` or ``file`` is
built first, as ``lattice`` in the last example.

``catemc generate`` writes the final template to ``{name}_template.traj``.
``catemc run`` and ``catemc analyze`` read that file, so the atom indices
of all three stages agree.

``regions``
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Key
     - Default
     - Meaning
   * - ``fixed``
     - none
     - atoms that never move and never change (``fix_indices``)
   * - ``mutable_metals``
     - all metal sites that are not fixed
     - metal sites the moves may act on (``mutable_metal_indices``)
   * - ``mutable_nonmetals``
     - all non-metal sites that are not fixed
     - non-metal sites the moves may act on
       (``mutable_nonmetal_indices``)
   * - ``slab``
     - ``all``
     - atoms of the template that form the slab (``slab_indices``, grand
       canonical only)

Each entry is a *selection*:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Selection
     - Atoms
   * - ``all``, ``none``
     - every atom, no atom
   * - ``[0, 1, 2, 5]``
     - a list of indices
   * - ``{z_below: 15.0}``, ``{z_above: 15.0}``
     - atoms with :math:`z < 15` Å, with :math:`z \ge 15` Å
   * - ``{bottom_layers: 2}``, ``{top_layers: 1}``
     - the lowest or highest atomic layers along z (``layer_tolerance``,
       default 0.3 Å, separates the layers)
   * - ``{symbols: [O]}``
     - atoms of these elements
   * - ``{z_above: 18.8, symbols: [O]}``
     - several criteria in one mapping are combined with *and*

``population``
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Key
     - Default
     - Meaning
   * - ``temperatures``
     - required
     - list of temperatures
   * - ``replicas_per_temperature``
     - 1
     - replicas at each temperature. The Gibbs ensemble needs at least 2.
   * - ``seed``
     - random
     - random seed. The seed that was used is printed.
   * - ``file``
     -
     - use an existing population instead of
       ``{name}_input_pop.traj``. ``catemc generate`` is then not needed.
   * - ``metals``, ``nonmetals``
     - ``composition``
     - composition of the starting structures when it differs from
       ``composition``: a formula, a mapping, or ``template`` to keep the
       sublattice as it is in the template. Surface runs keep the
       non-metal lattice of the template by default.
   * - ``variants``
     -
     - groups of starting structures (below)
   * - ``adsorbate_species``
     - ``adsorbates.species``
     - fragments of the random starting adlayers. ``[]`` gives clean
       surfaces.
   * - ``adsorbate_coverage``
     - random
     - coverage of the starting adlayers in monolayers
   * - ``max_adsorbate_species``
     - no limit
     - largest number of fragment types in one starting structure
   * - ``adsorbate_adsorbate_dmin``
     - ``emc.adsorbate_adsorbate_dmin``
     - smallest distance between two fragments
   * - ``valences``
     -
     - ``{element: charge}``. Only charge-neutral decorations are
       generated.
   * - ``symmetry_constraint``, ``symmetry_dict``
     - ``false``
     - generate symmetric slabs (arguments of ACAT's
       ``SymmetricSlabOrderingGenerator``)
   * - ``round_decimals``
     - 3
     - decimals of the chemical potentials stored in the μ-windows

``variants`` divides the population into groups with different starting
structures. Every key of the table from ``metals`` on can be set per
variant. ``size`` is the number of structures of the variant. Variants
without ``size`` share what is left:

.. code-block:: yaml

   population:
     temperatures: [1023]
     replicas_per_temperature: 14
     adsorbate_species: [O, H, OH]
     variants:
       - {size: 4}                              # oxygen lattice of the template
       - {size: 5, nonmetals: {O: 20, X: 1}}    # one oxygen site in 21 vacant
       - {size: 5, nonmetals: {O: 10, X: 1}}

The structures are distributed over the temperatures in turn. For a
surface run they are sorted by their adsorbate content first, so that
every temperature receives the whole range of coverages, and each replica
receives its μ-window (:ref:`mu-windows`).

``calculators``
---------------

A list with one entry per calculator, used one after the other:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``calculator``
     - ``repulsive`` (:class:`catemc.repulsive.RepulsivePotential`),
       ``emt``, ``mace`` (``mace.calculators.mace.MACECalculator``, for a
       model file), ``mace_mp`` (``mace.calculators.mace_mp``, for a MACE
       foundation model such as ``model: medium-mpa-0``) or the import
       path of any ASE calculator
   * - ``fmax``
     - force threshold in eV/Å of the relaxation with this calculator
   * - ``args``
     - list of positional arguments of the calculator
   * - anything else
     - keyword argument of the calculator

Alternatively, a mapping that names a function of your own:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``factory``
     - ``file.py:function`` or ``package.module:function``. The function
       returns one calculator or a list of calculators.
   * - ``kwargs``
     - keyword arguments of the function
   * - ``fmax``
     - list with one force threshold per calculator

:doc:`cli` shows an example of each form.

``fitness``
-----------

Settings of the relaxation and the MD run that evaluate a candidate. The
keys are the arguments of
:func:`~catemc.calculation_templates.calculate_bulk_fitness` (canonical
and Gibbs) and
:func:`~catemc.calculation_templates.calculate_surface_fitness` (grand
canonical).

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Meaning
   * - ``md_timestep``
     - 1 fs
     - MD time step
   * - ``md_steps``
     - 400
     - MD steps per MD run
   * - ``md_type``
     - ``berendsen``
     - ``berendsen`` or ``langevin``
   * - ``md_taut``
     - 500 fs
     - time constant of the Berendsen thermostat
   * - ``md_friction``
     -
     - friction of the Langevin thermostat in fs\ :sup:`-1`
   * - ``md_taup``
     - 1000 fs
     - time constant of the barostat (bulk, with ``emc.pressure``)
   * - ``compressibility``
     - 5·10\ :sup:`-5` bar\ :sup:`-1`
     - compressibility of the barostat (bulk, with ``emc.pressure``)
   * - ``md_write_interval``
     - 10
     - interval of the MD trajectory, when it is written
   * - ``optimizer``
     - ``SQNM``
     - ``SQNM`` or an optimizer of ``ase.optimize``
   * - ``max_relax_steps``
     - 300
     - largest number of optimizer steps. A candidate that does not
       converge is rejected.
   * - ``optimize_cell``
     - ``false``
     - relax the cell together with the atoms (bulk)
   * - ``neighbor_cutoff``
     - 2.5 Å
     - cutoff used to put vacancies back after the atoms have moved
   * - ``update_md_positions``
     - ``true``
     - continue from the coordinates after MD
   * - ``update_relaxed_positions``
     - ``false``
     - continue from the relaxed coordinates
   * - ``write_relax_traj``, ``write_md_traj``
     - ``false``
     - keep the trajectories of every evaluation, for debugging
   * - ``fmax_list``
     - from ``calculators``
     - force thresholds, when they are not given with the calculators

``emc``
-------

The keyword arguments of the class selected by ``ensemble``, with the same
names and defaults as in Python. ``pressure`` is given in bar.

.. list-table:: All ensembles
   :header-rows: 1
   :widths: 30 16 54

   * - Key
     - Default
     - Meaning
   * - ``crossover_rate``
     - 0
     - rate of the cut-and-splice crossover
   * - ``md_generation``
     - 10
     - run MD every this many steps
   * - ``exchange_generation``
     - 1
     - attempt replica exchange every this many steps
   * - ``trim_generation``
     - 100
     - move older steps to a backup database every this many steps
   * - ``gpu_devices``
     -
     - list of GPU numbers. Without it the workers use the default device.
   * - ``procs_per_device``
     - 1
     - worker processes per device
   * - ``worker_startup_timeout``
     - 300
     - seconds allowed for the workers to load their calculators
   * - ``worker_result_timeout``
     - 3600
     - seconds allowed for one evaluation
   * - ``scratch_dir``
     -
     - folder on a fast local disk for the live database
   * - ``scratch_sync_interval``
     -
     - copy the live database back every this many steps
   * - ``overwrite_scratch_db``
     - ``false``
     - replace the databases in ``scratch_dir`` by those of ``directory``
       before starting
   * - ``traj_write_interval``
     - 1
     - write the population trajectory every this many steps
   * - ``raw_score_cutoff``
     - −10000
     - score given to a candidate that failed, which rejects it
   * - ``verbose``
     - ``true``
     - print every move, not only the summary of each step
   * - ``temperature_key``, ``replica_key``, ``acceptance_key``
     - ``temperature``, ``replica``, ``accepted``
     - names under which these values are stored with each structure

.. list-table:: Canonical and Gibbs
   :header-rows: 1
   :widths: 30 16 54

   * - Key
     - Default
     - Meaning
   * - ``intraswap_rate``
     - 0
     - rate of swaps within a replica
   * - ``pressure``
     -
     - pressure in bar. Without it the cell is fixed.
   * - ``metal_interswap_rate``
     - 0
     - Gibbs: rate of metal swaps between replicas
   * - ``nonmetal_interswap_rate``
     - 0
     - Gibbs: rate of non-metal swaps between replicas, for example
       oxygen and vacancies
   * - ``bias_starting_step``
     -
     - Gibbs: select the pair of an inter-swap with species weights from
       this step on (:doc:`bulk_geemc`)
   * - ``interswap_within_key``
     -
     - Gibbs: inter-swaps only between replicas with the same value of
       this entry of ``atoms.info['key_value_pairs']``
   * - ``category_key``
     -
     - canonical: an entry of ``atoms.info['key_value_pairs']`` that is
       kept with each replica

.. list-table:: Grand canonical
   :header-rows: 1
   :widths: 30 16 54

   * - Key
     - Default
     - Meaning
   * - ``interswap_rate``
     - 0
     - rate of metal swaps between replicas
   * - ``slab_intraswap_rate``
     - 0
     - rate of swaps within the slab
   * - ``slab_mutation_rate``
     - 0
     - rate of mutations of a slab atom
   * - ``adsorbate_insertion_rate``, ``adsorbate_removal_rate``
     - 0
     - rates of inserting and removing a fragment
   * - ``adsorbate_displacement_rate``
     - 0
     - rate of moving a fragment to another site
   * - ``adsorbate_mutation_rate``
     - 0
     - rate of turning a fragment into another one
   * - ``mu_walk_generation``
     - 1
     - attempt a μ-walk every this many steps
   * - ``bias_starting_step``
     -
     - apply the free-energy bias in the μ-walk from this step on, which
       makes the walk visit all reaction conditions evenly
   * - ``mbar_starting_step``
     - 100
     - first step of the samples used for the free-energy estimates of
       the bias
   * - ``mbar_sample_stride``
     - 10
     - steps between two of these samples
   * - ``adsorbate_surface_dmax``
     - 2.5 Å
     - a fragment further from the surface than this is removed
   * - ``adsorbate_adsorbate_dmin``
     - 2.0 Å
     - smallest distance between two fragments
   * - ``max_adsorbate_species``
     - no limit
     - largest number of fragment types in one replica
   * - ``max_adsorbate_coverage``
     - 1.0
     - largest coverage in monolayers
   * - ``ignore_sites``
     -
     - site types that are not used for insertion, for example ``bridge``

The remaining arguments of the classes are objects that the command line
builds from the other sections. ``fix_indices``,
``mutable_metal_indices``, ``mutable_nonmetal_indices`` and
``slab_indices`` come from ``regions``; ``metal_element_pool``,
``nonmetal_element_pool`` and ``bulk_metal_composition`` from
``composition``; ``adsorbate_species``, ``other_adsorbate_species``,
``volume`` and ``temperature_species_internal_map`` from ``adsorbates``;
``reaction_conditions`` and the element maps from
``chemical_potentials``. Setting one of them in ``emc`` replaces the
derived value.

``run``
-------

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Key
     - Default
     - Meaning
   * - ``total_steps``
     - 10000
     - step at which the simulation stops. A finished or interrupted run
       continues from its last completed step.
   * - ``starting_step``
     - last step
     - restart from this step instead
   * - ``restart_with_trim``
     - ``false``
     - together with ``starting_step``: delete the data after that step

``adsorbates``
--------------

Grand canonical only.

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Key
     - Default
     - Meaning
   * - ``species``
     - required
     - fragments that are inserted, removed, displaced and mutated
   * - ``other_species``
     - the other fragments of ``internal``
     - fragments that are not inserted but may form during MD
   * - ``internal``
     -
     - internal partition functions of the fragments (below)
   * - ``gas_height``
     -
     - height of the gas region in Å. The accessible volume
       (:eq:`eq-volume`) is the cell area times this height minus the
       covalent spheres of the metal atoms.
   * - ``volume``
     -
     - the accessible volume in Å\ :sup:`3`, instead of ``gas_height``
   * - ``sites``
     -
     - arguments of ACAT's ``SlabAdsorptionSites``. The default is
       ``{surface: autoadsorbate, composition_effect: false}``.

``internal`` gives the internal partition function
:math:`Q_k^\mathrm{int}` of every fragment (:eq:`eq-qint`) in one of three
ways:

.. code-block:: yaml

   adsorbates:
     internal:
       N: {degeneracy: 4}                          # atom: g0
       NH2: {degeneracy: 2, symmetry_number: 2}    # rigid rotor: g0 and sigma
       CO: 107.3                                   # a number is used as it is

   adsorbates:
     internal: internal_partition_functions.pkl    # {T: {fragment: Q_int}}

With ``degeneracy`` and ``symmetry_number`` the rotational part is
computed at every temperature from the moments of inertia of the ACAT
fragment. Both default to 1.

``chemical_potentials``
-----------------------

Grand canonical only.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``metals``
     - ``{name}_mbar_dmu_estimates.pkl`` of the Gibbs-ensemble run of the
       bulk. It must contain every temperature of the surface run. Leave
       it out only when the metal composition of the surface is fixed.
   * - ``elements``
     - one entry per reservoir element, in the order of the columns of
       the reaction conditions

Each element takes:

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Key
     - Default
     - Meaning
   * - ``reference``
     - 0
     - reference energy :math:`E^\mathrm{ref}` of the element. The grid
       is relative to it: :math:`\mu = E^\mathrm{ref} + \Delta\mu`
       (:eq:`eq-dmu-element`).
   * - ``range``, ``step``
     -
     - first and last value of :math:`\Delta\mu` and the spacing
   * - ``values``
     -
     - explicit list of :math:`\Delta\mu` values, instead of ``range``
       and ``step``
   * - ``y2x``
     - from ``composition``
     - ratio of this element to the metals in the bulk
       M\ :sub:`x`\ A\ :sub:`y`. It is zero for an element that is not
       part of the bulk.

.. code-block:: yaml

   chemical_potentials:
     metals: ../bulk/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_bulk_mbar_dmu_estimates.pkl
     elements:
       O: {reference: -4.3335, range: [-5.0, -3.0], step: 0.1}
       H: {reference: -3.253, range: [-1.0, 0.0], step: 0.05}

The reaction conditions of the simulation are all combinations of these
values with ``population.temperatures``.

``analysis``
------------

Settings at the top level of the section apply to every task. A task runs
when it is listed. ``true`` runs it with its defaults.

.. list-table:: Shared settings
   :header-rows: 1
   :widths: 26 16 58

   * - Key
     - Default
     - Meaning
   * - ``starting_step``
     - first step
     - first step of the samples. Set it after the equilibration.
   * - ``ending_step``
     - last step
     - last step of the samples
   * - ``sample_stride``
     - 1
     - use every this many steps
   * - ``use_exist``
     - ``true``
     - reuse the cached samples. ``catemc analyze --fresh`` sets it to
       ``false``.
   * - ``temperatures``
     -
     - grand canonical: additional temperatures to reweight to
   * - ``conditions``
     -
     - grand canonical: additional single reaction conditions (below)

.. list-table:: Tasks of the canonical and the Gibbs ensemble
   :header-rows: 1
   :widths: 22 78

   * - Task
     - Keys
   * - ``trajectory``
     - ``replicas``: list of replicas whose trajectories are written.
       ``true`` writes all of them.
   * - ``diagnostics``
     - ``replica``: restrict the diagnostics to one replica
   * - ``sro``
     - ``cutoff``: neighbour distance in Å. ``temperature``: one
       temperature instead of all. ``metal_only`` (``true``),
       ``use_energy_mask`` (``true``).
   * - ``dmu``
     - ``target_temperatures``: temperatures at which the differences
       are evaluated. ``simulation_temperatures``, ``wls_correction``
       (``true``), ``excess_only`` (``false``).

.. list-table:: Tasks of the grand-canonical ensemble
   :header-rows: 1
   :widths: 22 78

   * - Task
     - Keys
   * - ``trajectory``, ``diagnostics``
     - as above
   * - ``expectations``
     - ``step``: finer spacing of the grid per element, for example
       ``{N: 0.025, H: 0.025}``. ``range``: other limits per element.
       ``temperatures``, ``conditions``. ``compute_uncertainty``
       (``false``), ``procs`` (1).
   * - ``fes``
     - ``bin_width`` (0.05), ``variable_keys`` (``[all]``), ``step``,
       ``range``, ``temperatures``, ``conditions``, ``procs``. The free
       energies are evaluated at the listed ``conditions``, or at the
       whole grid when none is listed.
   * - both
     - which atoms are surface atoms: ``z_surf`` (height in Å) or
       ``surf_indices``. Without them AutoAdsorbate detects the surface
       (``precision``, ``touch_sphere_size``, ``surf_detect_stride``).
       Lattice vacancies: ``vacancy_species`` or ``vacancy_indices``.

These keys can be written at the top level of ``analysis`` to share them
between ``expectations`` and ``fes``.

A condition is a mapping with one value per reservoir element and the
temperature ``T``. The values are :math:`\Delta\mu` like the grid.
``absolute: true`` marks them as absolute chemical potentials, from which
the reference energies are subtracted:

.. code-block:: yaml

   analysis:
     starting_step: 500
     sample_stride: 50
     z_surf: 22.0
     conditions:
       - {N: -8.579, H: -3.960, T: 800, absolute: true}
       - {N: -0.2, H: -0.7, T: 800}
     expectations:
       step: {N: 0.025, H: 0.025}
       temperatures: [800]
     fes:
       bin_width: 0.05

The reaction conditions of a task are saved as
``{name}_mbar_{task}_conditions.npy``, one row per condition: first the
grid at the simulated temperatures, then the grid at the additional
temperatures, then the single conditions.
