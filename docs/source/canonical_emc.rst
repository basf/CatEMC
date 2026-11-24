Canonical ensemble
==================

In the canonical ensemble every replica keeps its composition. The
simulation answers the question *how do the atoms arrange themselves at
this composition and temperature?*, and gives the short-range order, the
site preferences and low-energy structures of a bulk material at several
temperatures at once.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Python
     - :class:`catemc.emc.EMC`
   * - Command line
     - ``ensemble: canonical``, then ``catemc run config_canonical.yaml``
   * - Examples
     - ``emc_run.py`` and ``config_canonical.yaml`` in
       ``examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk`` and
       ``examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk``

Theory
------

Each replica :math:`r` at temperature state :math:`t` samples the canonical
distribution of its own composition,

.. math::
   :label: eq-rho-canonical

   \rho_t(\mathbf r) \propto e^{-\beta_t U(\mathbf r)}

Three moves act on the population.

**Intra-swap.** One replica is selected, and two atoms of different types
in it (or an atom and a vacancy) exchange their positions. The proposal is
symmetric, so detailed balance (:eq:`eq-balance-one`) gives

.. math::
   :label: eq-p-intra

   P_\mathrm{intra} = \min\left(1, e^{-\beta \Delta U}\right)

with :math:`\Delta U = U(\mathbf r') - U(\mathbf r)` the energy change
after relaxation.

**Cut-and-splice crossover.** Two replicas :math:`a` and :math:`b` are
selected and cut by the same plane. The complementary parts are joined to
two offspring in such a way that each offspring keeps the composition and
the temperature of one parent. With a symmetric proposal, the two-replica
balance (:eq:`eq-balance-two`) gives

.. math::
   :label: eq-p-cross

   P_\mathrm{cross} = \min\left(1,
   e^{-\beta_a \Delta U_a - \beta_b \Delta U_b}\right)

**Exchange.** Two replicas with different temperatures and the same
composition exchange their configurations:

.. math::
   :label: eq-p-exchange

   P_\mathrm{ex} = \min\left[1,
   e^{(\beta_b - \beta_a)(U_b - U_a)}\right]

Exchange is what makes the population more than a set of independent
simulations: arrangements found by the hot replicas, which cross barriers
easily, are passed down to the cold replicas.

.. admonition:: Constant pressure
   :class: catemc-theory

   When ``pressure`` is set, the cell of every replica is relaxed together
   with the atoms and the replicas sample the isothermal-isobaric
   distribution

   .. math::
      :label: eq-rho-npt-one

      \rho_t(\mathbf r, V) \propto V^{N} e^{-\beta_t\,[U(\mathbf r, V) + pV]}

   The energy is replaced by the enthalpy :math:`U + pV`, and a move that
   changes the volume acquires a Jacobian factor:

   .. math::
      :label: eq-p-intra-npt

      P_\mathrm{intra} = \min\left[1, \left(\frac{V'}{V}\right)^{N}
      e^{-\beta\,[(U' + pV') - (U + pV)]}\right]

   .. math::
      :label: eq-p-cross-npt

      P_\mathrm{cross} = \min\left[1,
      \frac{V_a'^{\,N_a} V_b'^{\,N_b}}{V_a^{N_a} V_b^{N_b}}\,
      e^{-\beta_a [(U_a' + pV_a') - (U_a + pV_a)]
         -\beta_b [(U_b' + pV_b') - (U_b + pV_b)]}\right]

   .. math::
      :label: eq-p-exchange-npt

      P_\mathrm{ex} = \min\left[1,
      e^{(\beta_b - \beta_a)[(U_b + pV_b) - (U_a + pV_a)]}\right]

   The volume factors cancel in the exchange, because the two replicas
   only trade places.

Example 1: HEA bulk
-------------------

The alloy keeps the nominal composition
Co\ :sub:`0.25`\ Mo\ :sub:`0.45`\ Fe\ :sub:`0.1`\ Ni\ :sub:`0.1`\ Cu\ :sub:`0.1`
in every replica. The volume is fixed at the experimental lattice
constant. Intra-swaps do most of the work, and a small crossover rate adds
large rearrangements.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. literalinclude:: ../../examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk/config_canonical.yaml
         :language: yaml
         :start-at: name:

      .. code-block:: console

         $ cd examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk
         $ catemc generate config_canonical.yaml
         $ catemc run config_canonical.yaml
         $ catemc analyze config_canonical.yaml

   .. tab-item:: Python
      :sync: python

      The population is the one from :doc:`input_populations`.

      .. literalinclude:: ../../examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk/emc_run.py
         :language: python
         :start-at: from catemc.emc

      .. code-block:: console

         $ python -u emc_run.py

Example 2: HEO bulk with oxygen vacancies
-----------------------------------------

For the spinel oxide the composition
(Co\ :sub:`0.2`\ Ni\ :sub:`0.2`\ Mg\ :sub:`0.2`\ Zn\ :sub:`0.2`\ Mn\ :sub:`0.2`)Al\ :sub:`2`\ O\ :sub:`3.5`
fixes both the metals and the number of oxygen vacancies: one eighth of the
oxygen sites is empty in every replica throughout the run. The intra-swap
now acts on both sublattices. It exchanges two unlike metals, or an oxygen
atom with a vacancy, which moves the vacancy through the lattice.

The simulation runs at constant pressure (1 atm), so the cell adapts to the
arrangement of cations and vacancies, and the repulsive potential
(:eq:`eq-repulsive`) pre-relaxes every candidate.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. literalinclude:: ../../examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk/config_canonical.yaml
         :language: yaml
         :start-at: name:

      .. code-block:: console

         $ cd examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk
         $ catemc generate config_canonical.yaml
         $ catemc run config_canonical.yaml
         $ catemc analyze config_canonical.yaml

      The vacancies come from the ``X`` in ``composition``. To study
      another vacancy concentration, change that number, for example
      ``...Al2O3.75X0.25``.

   .. tab-item:: Python
      :sync: python

      .. literalinclude:: ../../examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk/emc_run.py
         :language: python
         :start-at: from catemc.emc

      The vacancies are part of the population (``target_nonmetal_species =
      ['O', 'X']`` in :doc:`input_populations`), and
      ``nonmetal_element_pool=['O', 'X']`` tells the moves to treat them as
      a species.

Practical advice
----------------

A canonical run is analysed with :class:`~catemc.analysis.EMCAnalysis`.
The Warren-Cowley parameters show which pairs of metals attract or avoid
each other at each temperature, and the diagnostics report the acceptance
ratios and the correlation time. A canonical run has no Widom attempts, so
it gives no chemical-potential differences. These come from the
:doc:`Gibbs ensemble <bulk_geemc>`.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: yaml

         analysis:
           starting_step: 300
           ending_step: 1000
           sample_stride: 10
           trajectory: {replicas: [10]}
           diagnostics: true
           sro: {cutoff: 3.5, use_energy_mask: true}

      .. code-block:: console

         $ catemc analyze config_canonical.yaml

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from catemc.analysis import EMCAnalysis

         ana = EMCAnalysis(
                 name='Co0.25Mo0.45Fe0.1Ni0.1Cu0.1_bulk_canonical',
                 calculation_dir=dir_name,
                 pop_size=48,
                 target_metal_species=['Co', 'Mo', 'Fe', 'Ni', 'Cu'],
                 trim_generation=100)
         ana.prepare_data()
         ana.get_trajectory(replica=10)
         ana.get_diagnostics(starting_step=300, ending_step=1000)
         ana.get_sro(starting_step=300, ending_step=1000, sample_stride=10,
                     cutoff=3.5, use_energy_mask=True)

:doc:`analysis` explains the output files.

Use the canonical ensemble when the composition of the simulation cell is
the quantity you want to control, for example to compare the ordering at
two exact compositions or vacancy concentrations. Use the
:doc:`Gibbs ensemble <bulk_geemc>` when you need chemical potentials, or
when the cell is so small that a fixed composition would suppress
fluctuations that matter. The Gibbs ensemble samples the same average
composition and lets each replica fluctuate around it.
