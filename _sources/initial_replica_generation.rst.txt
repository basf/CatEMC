Initial replica generation
==========================

This section covers creation of initial replicas for both **bulk** (GE-EMC)
and **surface** (GC-EMC) simulations. It shows how to produce on-lattice
replicas, assign metadata (replica id, temperature, emc_step), and — for
surfaces — how to assign μ-windows (reaction-condition grid mapping).

.. contents::
   :local:
   :depth: 2

Why careful replica generation matters
-------------------------------------

Starting replicas define the configuration space the EMC engines will explore.
Good practice:

* Validate the relaxed template structure visually before randomization.
* Keep population size a multiple of the number of temperatures/replicas.
* For surfaces, ensure adsorption sites are detected correctly.

Bulk replica generation (GE-EMC)
-------------------------------

Generates ``pop_size`` on-lattice bulk replicas consistent with the target
composition using :func:`catemc.generate.generate_population`.

**Inputs**

* A relaxed template structure (CIF / ASE trajectory).
* Target metal/nonmetal species and target compositions.

**Output**

* ``*_input_pop.traj`` containing ``pop_size`` ``ase.Atoms`` objects with
  ``atoms.info`` metadata.

**Example (bulk)**

.. code-block:: python

   from catemc.generate import generate_population
   from ase.io import read, write
   from ase.visualize import view

   comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
   template = read('Al2CoO4.cif')
   # expand cell to avoid image-image interactions
   cell = template.cell.copy()
   for i in range(3):
       cell[i][i] = 8.06
   template.set_cell(cell, scale_atoms=True)
   template *= (2,2,2)

   # split metals and oxygens so we preserve ordering on the lattice
   metal_ids, oxygen_ids = [], []
   for i, atom in enumerate(template):
       if atom.symbol == 'O':
           oxygen_ids.append(i)
       else:
           metal_ids.append(i)
   metal_atoms, oxygen_atoms = template[metal_ids], template[oxygen_ids]
   # restore original ordering
   sorted_ids = [0] * len(template)
   for k, v in enumerate(metal_ids + oxygen_ids):
       sorted_ids[v] = k
   bulk = (metal_atoms + oxygen_atoms)[sorted_ids]

   input_pop = generate_population(
       template=bulk,
       pop_size=24,
       target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
       target_metal_composition={'Co': 0.2, 'Ni': 0.2, 'Mg': 0.2, 'Zn': 0.2, 'Mn': 0.2, 'Al': 2},
       target_nonmetal_species=['O', 'X'],
       target_nonmetal_composition={'O': 3.5, 'X': 0.5},
   )

Assigning replica metadata (bulk)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After generating the images you should assign a replica id and temperature
per image so the GEEMC engine and analysis tools can group images by replica.

.. code-block:: python

   rep = 0
   for T in [623, 723, 823, 923, 1023, 1123]:
       for _ in range(4):  # 6 temperatures * 4 = 24 images
           img = input_pop[rep]
           img.info = {'key_value_pairs': {}, 'data': {}}
           img.info['key_value_pairs']['temperature'] = T
           img.info['key_value_pairs']['replica'] = rep
           img.info['key_value_pairs']['emc_step'] = 0
           rep += 1

   write('Co_bulk_input_pop.traj', input_pop)
   view(input_pop)

Surface replica generation (GC-EMC) + μ-window mapping
-----------------------------------------------------

Surface replica generation includes adsorbates; we generate on-lattice surface
candidates and then map μ-grid points (reaction conditions) to replicas using
:func:`catemc.generate.assign_mu_windows`.

**Inputs**

* Relaxed slab template.
* Adsorbate species and adsorption sites (recommended: ``SlabAdsorptionSites``).

**Output**

* ``*_input_pop.traj`` with per-image ``atoms.info['data']['mu']`` and ``mu_window``.

**Example (surface)**

.. code-block:: python

   from catemc.generate import generate_population, assign_mu_windows
   from acat.adsorption_sites import SlabAdsorptionSites
   from ase.io import read, write
   from ase.visualize import view
   import numpy as np

   template = read('template_Co_COH_311.traj')
   fix_ids = [a.index for a in template if a.position[2] < 18.8]
   mut_ids = [i for i in range(len(template)) if i not in fix_ids]
   sas = SlabAdsorptionSites(template, surface='autoadsorbate', composition_effect=False, ignore_sites='bridge')

   onlat_images = generate_population(
       template=template.copy(),
       pop_size=14,
       allowed_indices=mut_ids,
       target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
       target_metal_composition={'Co':0.2,'Ni':0.2,'Mg':0.2,'Zn':0.2,'Mn':0.2,'Al':2},
       adsorbate_species=['O','H','OH','CO','CO2'],
       adsorbate_adsorbate_dmin=2.,
       max_adsorbate_species=1,
       adsorption_sites=sas,
   )

   # assemble input population and metadata (temperatures)
   input_images, rep = [], 0
   for T in [623, 723, 823]:
       for _ in range(14):
           atoms = onlat_images[rep]
           atoms.info = {'key_value_pairs': {}, 'data': {}}
           atoms.info['key_value_pairs']['temperature'] = T
           atoms.info['key_value_pairs']['replica'] = rep
           atoms.info['key_value_pairs']['emc_step'] = 0
           input_images.append(atoms)
           rep += 1

   # Build reaction conditions grid (example C/O/H mesh)
   E_C = -9.944    # Etot per atom graphite PBE
   E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
   E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE
   dmu_C = np.arange(0., 2.001, 0.1)
   dmu_O = np.arange(-4., -2.999, 0.1)
   dmu_H = np.arange(-1., 0.001, 0.1)
   grid = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H)).T.reshape(-1, 3)
   element_column_map = {'C': 0, 'O': 1, 'H': 2}
   assigned_input_pop = assign_mu_windows(input_images, reaction_conditions=grid, element_column_map=element_column_map, round_decimals=3)

   write('Co_surface_input_pop.traj', assigned_input_pop)
   view(assigned_input_pop)

Practical checklist
-------------------

* Visually inspect several generated replicas (ASE `view`) before production runs.
* Keep a consistent metadata schema: use ``.info['key_value_pairs']`` for temperature/replica/emc_step and ``.info['data']`` for algorithmic data (mu, mu_window).
* Save the template and the final `*_input_pop.traj` into your calculation directory for reproducibility.

