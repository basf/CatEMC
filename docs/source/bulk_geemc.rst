Gibbs ensemble
==============

The Gibbs-ensemble EMC (GE-EMC) determines the chemical-potential
differences between all metals of a bulk material in a single run. These
differences define the bulk reservoir of a surface simulation
(:doc:`surface_gcemc`), and they are a result in their own right: they
measure how strongly each element is bound in the material.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Python
     - :class:`catemc.geemc.GEEMC`
   * - Command line
     - ``ensemble: gibbs``, then ``catemc run config.yaml``
   * - Examples
     - ``geemc_run.py`` and ``config.yaml`` in
       ``examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk`` and
       ``examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk``

Theory
------

The idea
^^^^^^^^

The usual way to obtain chemical potentials of an alloy is a series of
semi-grand-canonical simulations in which the chemical-potential
differences are adjusted until the composition comes out right. In a
material with five or six elements this search is expensive.

GE-EMC goes the other way. The :math:`R_t` replicas at one temperature
state :math:`t` form a *composite* system. Its total composition is fixed
at the target, and the replicas exchange atoms with each other, so each
replica sees the others as a particle reservoir at the right composition.
With the volume of each replica fixed, the target distribution of a
composite microstate :math:`\mathbf R = \{\mathbf r_r\}` is

.. math::
   :label: eq-rho-ge

   \rho_t^\mathrm{ge}(\mathbf R) \propto
   e^{-\beta_t \sum_r^{R_t} U_r(\mathbf r_r)}

Moves
^^^^^

The population is updated by four moves. Intra-swap, crossover and
exchange are those of the :doc:`canonical ensemble <canonical_emc>`
(:eq:`eq-p-intra`, :eq:`eq-p-cross`, :eq:`eq-p-exchange`). Exchange is
attempted between two replicas with different temperatures and the *same*
composition.

The new move is the **inter-swap**: two replicas :math:`a` and :math:`b`
at the same temperature are selected, then one atom in each, of different
types, and the two atoms exchange their positions. Replica :math:`a`
thereby turns an atom of type :math:`m` into type :math:`n` and replica
:math:`b` does the opposite, so the composite composition is conserved.

The acceptance probability depends on how the pair of atoms is selected.
Let :math:`\mathcal P_{ab}` be the set of eligible pairs of unlike atoms
and :math:`w_m` the selection weight of species :math:`m`. With

.. math::

   W_{ab} = \sum_{(i,j)\in\mathcal P_{ab}} w_{s_i} w_{s_j}

the pair :math:`(i, j)` is proposed with probability
:math:`w_{s_i} w_{s_j} / W_{ab}`. The reverse move has the same numerator
and the normalization of the proposed configurations, so that

.. math::
   :label: eq-p-inter

   P_\mathrm{inter} = \min\left[1, \frac{W_{ab}}{W_{a'b'}}\,
   e^{-\beta(\Delta U_a + \Delta U_b)}\right]

For uniform selection among the eligible pairs, :math:`W_{ab}` is simply
their number.

Chemical-potential differences from Widom's method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each attempted inter-swap is a Widom test: it measures the energy
:math:`\Delta U_{m\to n}` of turning one atom of type :math:`m` into type
:math:`n`. The chemical-potential difference follows from the ensemble
average of the corresponding Boltzmann factor,

.. math::
   :label: eq-widom

   \widehat{\Delta\mu}_{m\to n}^{(T),\mathrm{raw}} =
   -\frac{1}{\beta} \ln \left\langle \frac{N_m}{N_n + 1}\,
   e^{-\beta \Delta U_{m\to n}} \right\rangle

where :math:`N_m` and :math:`N_n` are the numbers of atoms of the two
types in the replica, and the sign convention is
:math:`\dmu{m}{n} = \mu_n - \mu_m`. The factor :math:`N_m/(N_n+1)` is the
ratio of the factorials in the two partition functions.

Every attempt contributes, whether it is accepted or not. CatEMC stores
:math:`\Delta U_{m\to n}` of all attempts in the database during the run,
and the analysis evaluates :eq:`eq-widom` with MBAR weights at any
temperature (:doc:`analysis`).

.. note::

   Because the volume of each replica is fixed, the chemical-potential
   differences refer to the imposed volume and the sampled structure, not
   to an unconstrained zero-pressure phase equilibrium. Use the
   constant-pressure variant below when the lattice parameter of the
   target composition is not known.

Constant pressure
^^^^^^^^^^^^^^^^^

.. admonition:: Isothermal-isobaric Gibbs ensemble
   :class: catemc-theory

   With ``pressure`` set, each replica relaxes its cell independently at
   the common pressure :math:`p`, and the composite distribution becomes

   .. math::
      :label: eq-rho-ge-npt

      \rho_t(\mathbf R, \{V_r\}) \propto \left(\prod_{r=1}^{R_t}
      V_r^{N_r}\right) e^{-\beta_t \sum_{r=1}^{R_t}
      [U_r(\mathbf r_r, V_r) + pV_r]}

   The inter-swap acceptance acquires the volume factors,

   .. math::
      :label: eq-p-inter-npt

      P_\mathrm{inter} = \min\left[1, \frac{W_{ab}}{W_{a'b'}}\,
      \frac{V_a'^{\,N_a} V_b'^{\,N_b}}{V_a^{N_a} V_b^{N_b}}\,
      e^{-\beta(\Delta U_a + \Delta U_b + p\Delta V_a + p\Delta V_b)}\right]

   and the Widom estimator becomes

   .. math::
      :label: eq-widom-npt

      \widehat{\Delta\mu}_{m\to n}^{(T),\mathrm{raw}} =
      -\frac{1}{\beta} \ln \left\langle \frac{N_m}{N_n + 1}
      \left(\frac{V'}{V}\right)^{N}
      e^{-\beta[(U' + pV') - (U + pV)]} \right\rangle

   The other moves change as in :eq:`eq-p-intra-npt` to
   :eq:`eq-p-exchange-npt`.

Example 1: HEA bulk
-------------------

The published settings: 48 replicas at twelve temperatures from 573 to
1400 K, a 512-atom fcc cell at fixed volume, and rates of 0.8 : 0.1 : 0.1
for inter-swap, intra-swap and crossover, which puts the emphasis on the
Widom attempts.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. literalinclude:: ../../examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk/config.yaml
         :language: yaml
         :start-at: name:
         :end-before: analysis:

      .. code-block:: console

         $ cd examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk
         $ catemc check config.yaml
         $ catemc generate config.yaml
         $ catemc run config.yaml

      The whole sequence, including the analysis, is in ``run_cli.sh``:

      .. literalinclude:: ../../examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk/run_cli.sh
         :language: bash

   .. tab-item:: Python
      :sync: python

      .. literalinclude:: ../../examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk/geemc_run.py
         :language: python

      .. code-block:: console

         $ python -u geemc_run.py

Example 2: HEO bulk with oxygen vacancies
-----------------------------------------

The spinel oxide has two sublattices that exchange atoms. Metals are
swapped between replicas with ``metal_interswap_rate``, and oxygen atoms
and vacancies with ``nonmetal_interswap_rate``, so the oxygen-vacancy
concentration of each replica fluctuates around one eighth. The run is at
constant pressure (1 atm) with the two-stage relaxation: repulsive
potential first, then MACE-MPA-0.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. literalinclude:: ../../examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk/config.yaml
         :language: yaml
         :start-at: # The repulsive potential
         :end-before: analysis:

      .. code-block:: console

         $ cd examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk
         $ catemc generate config.yaml
         $ catemc run config.yaml

      Pressures are given in bar and times in fs. ``catemc check
      config.yaml`` prints the values in ASE units that the classes
      receive.

   .. tab-item:: Python
      :sync: python

      .. literalinclude:: ../../examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk/geemc_run.py
         :language: python

Parameters specific to ``GEEMC``
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Argument
     - Default
     - Meaning
   * - ``metal_interswap_rate``
     - 0
     - rate of inter-swaps on the metal sublattice
   * - ``nonmetal_interswap_rate``
     - 0
     - rate of inter-swaps on the non-metal sublattice (for example O and
       vacancies)
   * - ``intraswap_rate``, ``crossover_rate``
     - 0
     - rates of the other two moves
   * - ``bias_starting_step``
     - ``None``
     - from this step on, the inter-swap selects the pair of atoms with
       the species weights :math:`w_m = 1/M` of :eq:`eq-p-inter`. ``None``
       always selects uniformly among the eligible pairs.
   * - ``interswap_within_key``
     - ``None``
     - a key of ``atoms.info['key_value_pairs']`` that divides the replicas
       into groups. Inter-swaps then only happen within a group, which
       allows several bulk compositions in one run.
   * - ``pressure``
     - ``None``
     - target pressure. ``None`` keeps every cell fixed.

The remaining arguments are described in :doc:`emc`.

Practical advice
----------------

* Use at least two replicas per temperature. With four, each replica has
  three partners for the inter-swap and a larger range of compositions is
  visited.
* Keep the inter-swap rate high. Only inter-swaps produce Widom samples.
* The analysis needs both directions :math:`m \to n` and :math:`n \to m`
  of every pair to be sampled. If an element is very dilute, enlarge the
  cell or run longer, and check the diagnostics before using the result.
* When the bulk run is meant to feed a surface run, make sure the analysis
  requests the temperatures of the surface run (``analysis.dmu.
  target_temperatures``). MBAR interpolates between the simulated
  temperatures.
* :doc:`analysis` turns the databases of this run into
  ``{name}_mbar_dmu_estimates.pkl``, the ``temperature_dmu_map`` of a
  surface run in Python (:doc:`surface_gcemc`). ``catemc analyze`` also
  prints the values as the ``chemical_potentials.metals`` block of the
  configuration file of a surface run.
