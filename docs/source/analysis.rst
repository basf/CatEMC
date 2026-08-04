Analysis
========

GE-EMC and GC-EMC write ASE databases during the run. Analysis classes parse the
database, prepare MBAR-ready samples, and write pickle files for downstream
plotting.

GE-EMC outputs
--------------

``GEEMCAnalysis.prepare_data()`` creates MBAR data files from database backups.
``GEEMCAnalysis.get_dmu_estimates()`` estimates directed metal substitution
chemical-potential differences and can apply weighted least-squares correction
to enforce antisymmetry and transitivity.

Main outputs:

* ``*_mbar_data.pkl``
* ``*_mbar_widom_data.pkl``
* ``*_mbar_dmu_estimates.pkl``

GC-EMC outputs
--------------

``GCEMCAnalysis.prepare_data()`` parses sampled structures and energies.
``GCEMCAnalysis.get_structural_expectations()`` reweights the samples across the
reaction-condition grid.

Main outputs:

* ``*_mbar_data.pkl``
* ``*_mbar_expectations.pkl``
* ``*_mbar.traj``
* ``*_sampled_indices.npy``

Choosing analysis windows
-------------------------

Use ``starting_step`` after the biased early stage and after raw scores,
coverages and μ-walk behavior have stabilized. Increase ``sample_stride`` when
replica time series are strongly correlated. Keep ``ending_step`` inside the
available database range after any trimming or restart operation.
