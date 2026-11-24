Grand-canonical ensemble
========================

The grand-canonical EMC (GC-EMC) simulates a catalyst surface that is open
to two reservoirs: the bulk underneath, which exchanges metal atoms with
the surface, and the gas phase above, which exchanges non-metal atoms. One
run maps the composition of the surface and the coverage of adsorbates
over a whole grid of reaction conditions.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Python
     - :class:`catemc.gcemc.GCEMC`
   * - Command line
     - ``ensemble: grand_canonical``, then ``catemc run config.yaml``
   * - Examples
     - ``gcemc_run.py`` and ``config.yaml`` in
       ``examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/111``; ``gcemc_run_OH.py``
       and ``config_OH.yaml`` in
       ``examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/311``
   * - Needs
     - the metal chemical-potential differences of the bulk
       (:doc:`bulk_geemc`, :doc:`analysis`)

Theory
------

The grand potential of a surface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The surface is modelled as a slab in *constrained equilibrium* with the
bulk and gas reservoirs. For a slab with particle numbers :math:`\{N_i\}`
at temperature :math:`T` and pressure :math:`p`, the grand potential is

.. math::
   :label: eq-omega

   \Omega_{T,p}(\{N_i\}) = F_T(\{N_i\})
   - \sum_{m\in\mathrm{M}} N_m \mu_m^{(T)}
   - \sum_{k\in\NM} N_k \mu_k^{(T,p)}

where :math:`F_T` is the Helmholtz free energy, :math:`\mathrm M` the set
of :math:`M` metal species and :math:`\NM` the set of :math:`K` non-metal
species.

The absolute chemical potentials :math:`\mu_m` of the metals in a
compositionally complex material are very hard to obtain. The differences
between them are not: they come from the Gibbs-ensemble simulation of the
bulk. For a bulk of composition
:math:`\{\mathrm{M}\}_x\overline{\mathrm{NM}1}_{y_1}\ldots\overline{\mathrm{NM}k}_{y_k}`
with the Gibbs free
energy per metal atom

.. math::
   :label: eq-gbulk

   g_\mathrm{bulk}^{(T,p)} = \sum_{m\in\mathrm{M}} c_m^\mathrm{bulk}\mu_m^{(T)}
   + \sum_{k\in\NM} \frac{y_k}{x}\,\mu_k^{(T,p)}

and with :math:`\sum_m c_m = \sum_m c_m^\mathrm{bulk} = 1`, the grand
potential can be written entirely in terms of differences:

.. math::
   :label: eq-omega-dmu

   \begin{aligned}
   \Omega_{T,p}(\{N_i\}) = {} & F_T(\{N_i\})
   - \frac{N_\mathrm{M}}{M} \sum_{\substack{m,n\in\mathrm{M}\\ m<n}}
     \left(\Delta c_{m\to n} - \Delta c_{m\to n}^\mathrm{bulk}\right)
     \dmu{m}{n}^{(T)} \\
   & - N_\mathrm{M}\, g_\mathrm{bulk}^{(T,p)}
   - \sum_{k\in\NM} \left(N_k - \frac{y_k}{x} N_\mathrm{M}\right)\mu_k^{(T,p)}
   \end{aligned}

Here :math:`N_\mathrm{M}` is the number of metal atoms of the slab,
:math:`c_m` the mole fraction of metal :math:`m` in it,
:math:`\Delta c_{m\to n} = c_n - c_m` and
:math:`\Delta c_{m\to n}^\mathrm{bulk} = c_n^\mathrm{bulk} - c_m^\mathrm{bulk}`.
The second term measures how far the composition of the surface deviates
from the bulk, and the last term how far its non-metal content deviates
from the bulk stoichiometry :math:`y_k/x`.

The reduced grand potential
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`\Omega` contains the entropy and is only defined as an ensemble
average. The acceptance probabilities use a *reduced* grand potential that
can be evaluated for a single structure :math:`\mathbf r` at a non-metal
chemical-potential state :math:`s` and a temperature state :math:`t`:

.. math::
   :label: eq-phi

   \begin{aligned}
   \Phi_{s,t}(\mathbf r) = {} & U(\mathbf r)
   - \frac{N_\mathrm{M}}{M} \sum_{\substack{m,n\in\mathrm{M}\\ m<n}}
     \left(\Delta c_{m\to n}(\mathbf r) - \Delta c_{m\to n}^\mathrm{bulk}\right)
     \dmu{m}{n}^{(t)} \\
   & - N_\mathrm{M}\, g_\mathrm{bulk}^{(s,t)}
   - \sum_{k\in\NM} \left(N_k(\mathbf r) - \frac{y_k}{x} N_\mathrm{M}\right)\mu_k^{(s)}
   \end{aligned}

It replaces the free energy by the potential energy of the relaxed
structure. Since every slab has the same number of metal atoms, the term
:math:`N_\mathrm{M} g_\mathrm{bulk}` is the same for all structures at a
given state and cancels in every acceptance probability.

.. list-table:: Where the terms of :math:`\Phi` come from
   :header-rows: 1
   :widths: 26 34 40

   * - Quantity
     - Python argument
     - Command line
   * - :math:`U(\mathbf r)`
     - last calculator of ``calculator_factory``
     - ``calculators``
   * - :math:`c_m^\mathrm{bulk}`
     - ``bulk_metal_composition``
     - ``composition``
   * - :math:`\dmu{m}{n}^{(t)}`
     - ``temperature_dmu_map``
     - ``chemical_potentials.metals``
   * - :math:`\mu_k^{(s)}` on a grid
     - ``reaction_conditions``, ``element_column_map``,
       ``element_eref_map``
     - ``chemical_potentials.elements``
   * - :math:`y_k/x`
     - ``element_y2x_map``
     - from ``composition`` (or ``y2x`` of an element)

Fragments in the gas reservoir
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metal atoms and lattice sites are counted as explicit assignments to
sites. The mobile non-metal *fragments* :math:`k\in\mathcal F` (atoms or
molecules such as N, H, NH, NH\ :sub:`2`) are described by the weights of
an ideal gas in the accessible volume :math:`V`. The target distribution
of GC-EMC is

.. math::
   :label: eq-rho-gc

   \rho_{s,t}^\mathrm{gc}(\mathbf r) \propto \left[\prod_{k\in\mathcal F}
   \frac{1}{N_k(\mathbf r)!}
   \left(\frac{V Q_\mathrm{int}^{(k)}}{\Lambda_k^3}\right)^{N_k(\mathbf r)}
   \right] e^{-\beta_t \Phi_{s,t}(\mathbf r)}

with the thermal wavelength
:math:`\Lambda_k = h/\sqrt{2\pi m_k \kB T}` and the internal partition
function :math:`Q_\mathrm{int}^{(k)}` of the fragment.

**Accessible volume.** Because every structure is relaxed, the volume
excludes the space taken by the metal atoms,

.. math::
   :label: eq-volume

   V = V_\mathrm{tot} - N_\mathrm{M} \sum_{m\in\mathrm{M}}
   c_m^\mathrm{bulk}\, \frac{4}{3}\pi r_m^3

where :math:`r_m` is the covalent radius of metal :math:`m`. On the
command line, ``adsorbates.gas_height`` gives :math:`V_\mathrm{tot}` as the
area of the cell times this height, and the spheres are subtracted
automatically.

**Internal partition function.** The fragments are treated as rigid, so
:math:`Q_\mathrm{vib} = 1` and

.. math::
   :label: eq-qint

   Q_\mathrm{int} = Q_\mathrm{rot}\, g_0, \qquad
   Q_\mathrm{rot} =
   \begin{cases}
     1 & \text{monatomic} \\[4pt]
     \dfrac{8\pi^2 I \kB T}{\sigma h^2} & \text{linear} \\[8pt]
     \dfrac{\sqrt{\pi I_x I_y I_z}}{\sigma}
     \left(\dfrac{8\pi^2 \kB T}{h^2}\right)^{3/2} & \text{non-linear}
   \end{cases}

where :math:`I` are the moments of inertia, :math:`\sigma` is the
rotational symmetry number and :math:`g_0` the degeneracy of the electronic
ground state. On the command line you give :math:`g_0` and :math:`\sigma`
for each fragment and CatEMC evaluates :eq:`eq-qint` at every temperature.
In Python the values are passed as ``temperature_species_internal_map``
(see ``get_internal_partition_functions.py`` in the example folders).

Moves and acceptance probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nine moves update the population. With :math:`\Delta\Phi =
\Phi_{s,t}(\mathbf r') - \Phi_{s,t}(\mathbf r)`, a move of one replica that
changes the numbers of fragments by :math:`\{\Delta N_k\}` is accepted with

.. math::
   :label: eq-p-gc-general

   P_\mathrm{intra} = \min\left\{1,
   \frac{q_\mathrm{rvs}}{q_\mathrm{fwd}}
   \left[\prod_{k\in\mathcal F} \frac{N_k!}{N_k'!}
   \left(\frac{V Q_\mathrm{int}^{(k)}}{\Lambda_k^3}\right)^{\Delta N_k}\right]
   e^{-\beta\Delta\Phi}\right\}

which gives the following special cases.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Move (rate argument)
     - Acceptance probability
   * - Surface atom intra-swap (``slab_intraswap_rate``)
     - :math:`\min\left(1, e^{-\beta\Delta\Phi}\right)`
   * - Surface atom mutation :math:`m\to n`, the new species drawn from
       the bulk composition (``slab_mutation_rate``)
     - :math:`\min\left[1, \dfrac{c_m^\mathrm{bulk}(1-c_m^\mathrm{bulk})}
       {c_n^\mathrm{bulk}(1-c_n^\mathrm{bulk})}\, e^{-\beta\Delta\Phi}\right]`
   * - Fragment insertion (``adsorbate_insertion_rate``)
     - :math:`\min\left[1, \dfrac{V Q_\mathrm{int}^{(k)}}
       {(N_k+1)\Lambda_k^3}\, e^{-\beta\Delta\Phi}\right]`
   * - Fragment removal (``adsorbate_removal_rate``)
     - :math:`\min\left(1, \dfrac{N_k\Lambda_k^3}
       {V Q_\mathrm{int}^{(k)}}\, e^{-\beta\Delta\Phi}\right)`
   * - Fragment displacement to another site
       (``adsorbate_displacement_rate``)
     - :math:`\min\left(1, e^{-\beta\Delta\Phi}\right)`
   * - Fragment mutation :math:`k\to l` (``adsorbate_mutation_rate``)
     - :math:`\min\left[1, \dfrac{N_k\Lambda_k^3 Q_\mathrm{int}^{(l)}}
       {(N_l+1)\Lambda_l^3 Q_\mathrm{int}^{(k)}}\, e^{-\beta\Delta\Phi}\right]`
   * - Surface atom inter-swap between two replicas at the same
       temperature (``interswap_rate``)
     - :math:`\min\left[1, \dfrac{W_{ab}}{W_{a'b'}}\,
       e^{-\beta(\Delta\Phi_a+\Delta\Phi_b)}\right]`

For insertion, the type of fragment is selected uniformly and placed on a
free surface site. For removal and mutation, the type :math:`k` is selected
first and then one of the :math:`N_k` fragments of that type, which is the
selection the factors :math:`N_k` and :math:`N_k/(N_l+1)` correspond to.
The expressions in the table hold for equal insertion and removal rates.
When the two rates differ, their ratio enters through
:math:`q_\mathrm{rvs}/q_\mathrm{fwd}`. A change
of a lattice oxygen atom into a vacancy, or back, is a mutation on the
non-metal sublattice with the ratio
:math:`q_\mathrm{rvs} e^{-\beta\Delta\Phi}/q_\mathrm{fwd}`.

**Surface-adsorbate crossover** (``crossover_rate``) mates the slab of one
replica with the adlayer of another. Each offspring keeps the state of one
parent:

.. math::
   :label: eq-p-gc-cross

   \begin{aligned}
   P_\mathrm{cross} = \min\Bigg\{1,\ & \prod_{k\in\mathcal F}\left[
   \frac{N_k^{(a)}!\, N_k^{(b)}!}{N_k^{(a')}!\, N_k^{(b')}!}
   \left(\frac{Q_\mathrm{int}^{(k,a)}}{\beta_a^{3/2}}\right)^{\Delta N_k^{(a)}}
   \left(\frac{Q_\mathrm{int}^{(k,b)}}{\beta_b^{3/2}}\right)^{\Delta N_k^{(b)}}
   \right] \\
   & \times e^{-\beta_a\Delta\Phi_a - \beta_b\Delta\Phi_b}\Bigg\}
   \end{aligned}

**Exchange** swaps the configurations of two replicas while their
thermodynamic states stay where they are. It is attempted between
neighbouring temperatures at the same chemical potentials (*horizontal*),
between different chemical potentials at one temperature (*vertical*), and
between neighbouring temperatures with different chemical potentials
(*cross*):

.. math::
   :label: eq-p-gc-exchange

   P_\mathrm{ex} = \min\left[1, \left(\frac{\beta_b}{\beta_a}\right)^{
   \sum_{k\in\mathcal F} \lambda_k (N_k^{(b)} - N_k^{(a)})}
   e^{-\beta_a\Delta\Phi_a - \beta_b\Delta\Phi_b}\right]

with :math:`\Delta\Phi_a = \Phi_{s_a,t_a}(\mathbf r_b) -
\Phi_{s_a,t_a}(\mathbf r_a)`, :math:`\Delta\Phi_b =
\Phi_{s_b,t_b}(\mathbf r_a) - \Phi_{s_b,t_b}(\mathbf r_b)`, and
:math:`\lambda_k = 3/2` for monatomic, :math:`5/2` for linear and
:math:`3` for non-linear fragments.

The extended grand-canonical ensemble
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far every replica has one fixed reaction condition. To obtain the
surface as a *function* of the reaction conditions in a single run, the
extended grand-canonical EMC (EGC-EMC) treats the non-metal chemical
potentials as an additional variable that is sampled. It targets the joint
distribution of structures and chemical-potential states :math:`s`,

.. math::
   :label: eq-rho-joint

   \rho_t^\mathrm{joint}(\mathbf r, s) \propto C_\mathrm{gc}\,
   e^{-\beta_t\Phi_{s,t}(\mathbf r)}\, e^{\mathcal W(s)}

where :math:`C_\mathrm{gc}` is the prefactor of :eq:`eq-rho-gc` and
:math:`\mathcal W` a bias. The probability of finding a replica at state
:math:`s` is proportional to :math:`\Xi_{s,t}\, e^{\mathcal W(s)}`, with
:math:`\Xi_{s,t}` the grand partition function. Choosing the bias equal to
the dimensionless free energy, :math:`\mathcal W(s) = f_{s,t} =
-\ln\Xi_{s,t}`, makes all points of the grid equally likely.

The additional move is the **μ-walk**: a replica moves to another point of
its μ-window (:doc:`input_populations`) without changing its structure.
With the bias set to the free energy :math:`\hat f` estimated by MBAR
during the run,

.. math::
   :label: eq-p-muwalk

   P_{s\to s'} = \min\left[1, e^{-\beta_t[\Phi_{s',t}(\mathbf r) -
   \Phi_{s,t}(\mathbf r)] + \hat f_{s',t} - \hat f_{s,t}}\right]

A single trial is often rejected once a replica sits at a point with a much
higher weight than the rest of its window. Evaluating :math:`\Phi` at other
chemical potentials costs nothing, so CatEMC uses a multiple-try Metropolis
(MTM) scheme. All :math:`J-1` other points of the window are candidates,
one of them, :math:`s'`, is selected with a probability proportional to its
weight, and the move is accepted with

.. math::
   :label: eq-p-mtm

   P_{s\to s'}^\mathrm{MTM} = \min\left[1,
   \frac{\sum_{j\ne s} e^{-\beta_t\Phi_{j,t}(\mathbf r) + \hat f_{j,t}}}
        {\sum_{j\ne s'} e^{-\beta_t\Phi_{j,t}(\mathbf r) + \hat f_{j,t}}}\right]

.. list-table:: Arguments that control the μ-walk
   :header-rows: 1
   :widths: 30 12 58

   * - Argument
     - Default
     - Meaning
   * - ``mu_walk_generation``
     - 1
     - attempt the μ-walk of every replica every this many steps
   * - ``mbar_starting_step``
     - 100
     - first step of the samples from which MBAR estimates :math:`\hat f`
   * - ``mbar_sample_stride``
     - 10
     - spacing in steps between these samples
   * - ``bias_starting_step``
     - ``None``
     - apply the bias :math:`\hat f` from this step on. Without it the
       grid is not sampled uniformly.

In the published simulations the bias is switched on after 300 steps and
estimated from the samples of steps 100 to 300.

From reaction conditions to chemical potentials
-----------------------------------------------

The grid of non-metal chemical potentials should cover the conditions the
catalyst sees during the reaction. They follow from the chemical
potentials of the gases.

Gas molecules
^^^^^^^^^^^^^

The chemical potential of a gas :math:`i` at partial pressure :math:`p_i`
and temperature :math:`T` is

.. math::
   :label: eq-mu-gas

   \mu_i^{(T,p_i)} = E_i^\mathrm{tot} + E_i^\mathrm{zpe}
   + \Delta\mu_i^{(T,p^\circ)} + \kB T \ln\frac{p_i}{p^\circ}

with the total energy and zero-point energy from DFT and the ideal-gas
contribution :math:`\Delta\mu_i^{(T,p^\circ)}` at the standard pressure
:math:`p^\circ`. The partial pressures change as the reaction proceeds,
which is described by the progress
:math:`\xi = (p_i - p_{i,\mathrm{init}})/(p_{i,\mathrm{eq}} - p_{i,\mathrm{init}})`
between the initial mixture (:math:`\xi = 0`) and equilibrium
(:math:`\xi = 1`).

Elemental chemical potentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simulation needs chemical potentials of *elements*. They are obtained
by assuming which gas pins each element. The grid then uses relative
values,

.. math::
   :label: eq-dmu-element

   \Delta\mu_k = \mu_k - \mu_k^\mathrm{ref}, \qquad
   \mu_\mathrm{N}^\mathrm{ref} = \tfrac12\left(E_{\mathrm{N_2}}^\mathrm{tot}
   + E_{\mathrm{N_2}}^\mathrm{zpe}\right)

and likewise for H and O. The reference energies must come from the same
level of theory as the potential. The examples use PBE energies, the
functional of the data that MACE-MPA-0 is trained on.

.. tab-set::

   .. tab-item:: Ammonia decomposition

      2 NH\ :sub:`3` ⇌ N\ :sub:`2` + 3 H\ :sub:`2`, with an initial NH\
      :sub:`3` pressure of 0.05 atm at 800 K.

      .. list-table::
         :header-rows: 1
         :widths: 34 22 22 22

         * - Chemical potential (eV)
           - NH\ :sub:`3`
           - N\ :sub:`2`
           - H\ :sub:`2`
         * - early stage, :math:`\xi = 0.1`
           - −20.459
           - −18.492
           - −7.920
         * - equilibrium, :math:`\xi = 1`
           - −20.928
           - −18.333
           - −7.762

      Hydrogen is pinned by the H\ :sub:`2` that is produced, and nitrogen
      by NH\ :sub:`3` or N\ :sub:`2`, whichever gives the higher value:

      .. math::

         \mu_\mathrm{H} = \tfrac12\mu_{\mathrm{H_2}}, \qquad
         \mu_\mathrm{N} = \max\left(\mu_{\mathrm{NH_3}} - 3\mu_\mathrm{H},\;
         \tfrac12\mu_{\mathrm{N_2}}\right)

      This gives :math:`\mu_\mathrm{N} = -8.579` eV and
      :math:`\mu_\mathrm{H} = -3.960` eV at the early stage. At
      equilibrium the two ways of pinning nitrogen agree to within
      0.12 eV, and the example uses the N\ :sub:`2` value,
      :math:`\mu_\mathrm{N} = -9.167` eV with
      :math:`\mu_\mathrm{H} = -3.881` eV. With the references
      :math:`\mu_\mathrm{N}^\mathrm{ref} = -8.2425` eV and
      :math:`\mu_\mathrm{H}^\mathrm{ref} = -3.253` eV, both conditions lie
      inside the grid :math:`\Delta\mu_\mathrm{N}, \Delta\mu_\mathrm{H}
      \in [-1, 0]` eV of the example.

   .. tab-item:: H\ :sub:`2` reduction

      M\ :sub:`x`\ O\ :sub:`y` + y H\ :sub:`2` ⇌ x M + y H\ :sub:`2`\ O,
      at 1023 K with :math:`p_{\mathrm{H_2}} = 1` atm and
      :math:`p_{\mathrm{H_2O}} = 10^{-5}` atm, which gives
      :math:`\mu_{\mathrm{H_2}} = -7.961` eV and
      :math:`\mu_{\mathrm{H_2O}} = -16.782` eV.

      Hydrogen is pinned by H\ :sub:`2` and oxygen by the water that is
      formed:

      .. math::

         \mu_\mathrm{H} = \tfrac12\mu_{\mathrm{H_2}} = -3.981\ \mathrm{eV},
         \qquad
         \mu_\mathrm{O} = \mu_{\mathrm{H_2O}} - \mu_{\mathrm{H_2}}
         = -8.821\ \mathrm{eV}

      With :math:`\mu_\mathrm{O}^\mathrm{ref} = -4.3335` eV and
      :math:`\mu_\mathrm{H}^\mathrm{ref} = -3.253` eV this is
      :math:`\Delta\mu_\mathrm{O} = -4.49` eV and
      :math:`\Delta\mu_\mathrm{H} = -0.73` eV, inside the grid
      :math:`\Delta\mu_\mathrm{O} \in [-5, -3]` eV,
      :math:`\Delta\mu_\mathrm{H} \in [-1, 0]` eV of the example.

When it is not obvious which gas pins which element,
:func:`catemc.generate.get_ls_elemental_mu` returns the elemental chemical
potentials that reproduce the chemical potentials of all gases best in the
least-squares sense, :math:`\boldsymbol\mu^\mathrm{LS} = \arg\min_{\boldsymbol\mu}
\lVert \mathbf N\boldsymbol\mu - \mathbf g\rVert_2^2`, where
:math:`\mathbf N` counts the atoms of each element in each gas:

.. code-block:: python

   >>> from catemc.generate import get_ls_elemental_mu
   >>> get_ls_elemental_mu({'H2O': -16.782, 'H2': -7.961})
   {'H': -3.9805, 'O': -8.821}
   >>> get_ls_elemental_mu({'NH3': -20.459, 'H2': -7.920})
   {'H': -3.96, 'N': -8.579}

Example 1: HEA(111) under ammonia-decomposition conditions
----------------------------------------------------------

The slab, the population and the μ-windows come from
:doc:`input_populations`. The simulation adds the adsorbates, the
reservoirs and the moves.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. literalinclude:: ../../examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/111/config.yaml
         :language: yaml
         :start-at: calculators:
         :end-before: analysis:

      .. code-block:: console

         $ cd examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/111
         $ catemc check config.yaml
         $ catemc generate config.yaml
         $ catemc run config.yaml

      ``catemc check`` prints the derived quantities, among them the
      accessible volume, the grid, the internal partition functions and
      whether the bulk reservoir file was found.

   .. tab-item:: Python
      :sync: python

      .. literalinclude:: ../../examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/111/gcemc_run.py
         :language: python

      The internal partition functions are written beforehand by
      ``get_internal_partition_functions.py``:

      .. literalinclude:: ../../examples/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/111/get_internal_partition_functions.py
         :language: python

Some points about this input:

* ``adsorbates.species`` (``adsorbate_species``) are the fragments that
  the moves insert, remove and mutate. ``other_species``
  (``other_adsorbate_species``) can form on the surface during relaxation
  and MD, for example NH\ :sub:`3` and H\ :sub:`2`, and are recognized
  and counted but not inserted.
* A fragment that moves further than ``adsorbate_surface_dmax`` from the
  surface has desorbed and is taken out of the cell.
* ``ignore_sites: bridge`` restricts insertions to top and hollow sites,
  which speeds up the run. Fragments can still relax into bridge
  positions.
* The metal reservoir must contain the temperatures of the population
  (573, 673, 773 and 873 K here). The bulk example requests them in its
  ``analysis.dmu.target_temperatures``.

Example 2: HEO(311) under H\ :sub:`2`-reduction conditions
----------------------------------------------------------

On the oxide, lattice oxygen takes part in the exchange with the gas
phase: a surface oxygen atom can turn into a vacancy and back. The
stoichiometry :math:`y/x = 3.5/3` of the bulk enters :eq:`eq-phi` so that a
slab with the bulk oxygen content is the reference.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Command line
      :sync: cli

      .. literalinclude:: ../../examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/311/config_OH.yaml
         :language: yaml
         :start-at: # Pre-relaxation
         :end-before: analysis:

      .. code-block:: console

         $ cd examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/311
         $ catemc generate config_OH.yaml
         $ catemc run config_OH.yaml

      The same folder contains ``config_COH.yaml`` for a simulation with
      C, O and H reservoirs (CO\ :sub:`2` hydrogenation), which shows a
      three-dimensional grid of chemical potentials.

   .. tab-item:: Python
      :sync: python

      .. literalinclude:: ../../examples/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/311/gcemc_run_OH.py
         :language: python

Some points about this input:

* Two calculators are applied in sequence: the repulsive potential to
  0.5 eV/Å and MACE-MPA-0 to 0.05 eV/Å. The pre-relaxation avoids
  unphysical structures when an
  oxygen atom is inserted into a vacancy that is surrounded closely by
  metal atoms.
* Oxygen vacancies are pseudo-atoms ``X``. They are removed for the
  energy evaluation and put back at the centroid of their neighbouring
  metal atoms afterwards, which keeps the indices of the surface atoms
  fixed.
* ``O`` as an adsorbate (degeneracy 3) and lattice oxygen are distinct:
  the former is a mobile fragment, the latter a lattice site.

Parameters specific to ``GCEMC``
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Argument
     - Meaning
   * - ``slab_indices``
     - atoms of the catalyst (all others are adsorbates)
   * - ``bulk_metal_composition``
     - :math:`c_m^\mathrm{bulk}`
   * - ``adsorbate_species``, ``other_adsorbate_species``
     - fragments that are inserted, and fragments that may form
   * - ``volume``
     - accessible volume :math:`V` (:eq:`eq-volume`)
   * - ``reaction_conditions``
     - array with one row per condition: the chemical potentials of the
       elements, then the temperature in the last column
   * - ``element_column_map``
     - column of each element in ``reaction_conditions``
   * - ``element_eref_map``
     - reference energy of each element when the grid holds relative
       chemical potentials
   * - ``element_y2x_map``
     - :math:`y_k/x` of each element in the bulk
   * - ``temperature_dmu_map``
     - ``{T: {'A2B': mu_B - mu_A}}`` from the bulk analysis
   * - ``temperature_species_internal_map``
     - ``{T: {fragment: Q_int}}``
   * - ``adsorption_sites``, ``ignore_sites``
     - ACAT adsorption sites and the site types that are skipped
   * - ``adsorbate_surface_dmax``
     - largest distance from the surface at which a fragment still counts
       as adsorbed (2.5 Å)
   * - ``adsorbate_adsorbate_dmin``
     - smallest distance between two adsorbates (2 Å)
   * - ``max_adsorbate_species``, ``max_adsorbate_coverage``
     - largest number of fragment types and largest coverage in one cell

The rates and the arguments of the μ-walk are listed in the theory
section above, and the remaining arguments in :doc:`emc`.

Practical advice
----------------

* Run the bulk first. Without ``temperature_dmu_map`` the metal
  composition of the surface cannot change meaningfully.
* Equal insertion and removal rates are a good default. Different rates
  are allowed and are corrected for in the acceptance probability.
* Start the bias only after the population has equilibrated, and let MBAR
  collect samples for a few hundred steps before that.
* Watch the acceptance of the μ-walk and of the exchange in the
  diagnostics (:doc:`analysis`). If replicas stay in their starting
  windows, use more replicas or a coarser grid.
* The reservoirs define a constrained thermodynamic reference. Outside
  gas-phase equilibrium the result is not a prediction of the kinetic
  steady state.
