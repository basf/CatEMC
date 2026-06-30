Concepts
========

CatEMC uses a two-stage thermodynamic workflow for complex catalysts.

Bulk GE-EMC
-----------

GE-EMC samples multiple bulk replicas at different temperatures. Inter-replica
composition exchange allows each replica to drift locally while the composite
population preserves the target composition at each temperature. Widom mutation
records are accumulated during the run and later processed with MBAR to estimate
metal chemical-potential differences.

Surface GC-EMC
--------------

Surface GC-EMC treats the slab as an open system. Surface metal atoms, lattice
non-metals, vacancies and adsorbates can be mutated, inserted, removed or moved.
The acceptance probability includes the potential-energy change, metal
chemical-potential differences from bulk GE-EMC, elemental chemical potentials,
internal partition functions and proposal prefactors.

μ-window sampling
-----------------

Each surface replica carries a current chemical-potential point and a μ-window.
The μ-walk move changes the reaction condition sampled by that replica while
preserving detailed balance over the assigned chemical-potential grid. This makes
one production run useful for a broad reaction-condition phase diagram.

MBAR postprocessing
-------------------

Bulk and surface runs write databases that contain the sampled structures and
thermodynamic metadata. MBAR reweights these samples across temperature and
chemical-potential states to estimate chemical-potential differences, structural
expectations, free-energy surfaces and representative configurations.
