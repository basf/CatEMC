Quick start
===========


This quick-start shows the minimal steps to get CatEMC running locally for development and documentation.


Requirements
~~~~~~~~~~~~


* Python 3.9+ recommended
* Virtual environment (venv or conda)
* Key Python packages used by the code: ``autoadsorbate``, ``acat``, ``ase``, ``pymbar``


Environment and install
~~~~~~~~~~~~~~~~~~~~~~~


1. Create and activate a virtual environment:


.. code-block:: bash


python3 -m venv .venv
source .venv/bin/activate


2. Install CatEMC in editable mode and the runtime deps you need:


.. code-block:: bash


pip install -e .
pip install autoadsorbate acat ase pymbar


3. Verify import:


.. code-block:: bash


python -c "import catemc; print('catemc', getattr(catemc, '__version__', 'unknown'))"


Now you can follow the tutorials below to generate populations, run GE-EMC/GC-EMC and analyze results.
