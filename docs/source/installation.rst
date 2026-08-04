Installation
============

CatEMC is intended to be installed in the same Python environment that contains
ASE and the calculator package used for production runs.

Editable install
----------------

.. code-block:: bash

   git clone https://github.com/basf/CatEMC.git
   cd CatEMC
   python -m pip install -e .

Runtime packages
----------------

Install the packages required by your workflow. A typical environment contains:

.. code-block:: bash

   python -m pip install ase numpy scipy pymbar

CatEMC also uses ACAT and AutoAdsorbate for surface site handling and structure
generation. Install the MLIP calculator package used by your calculation script,
for example MACE or metatomic/PET-MAD.

Build the documentation locally
-------------------------------

The documentation has its own lightweight requirements:

.. code-block:: bash

   cd /path/to/CatEMC
   python -m pip install -r docs/requirements.txt
   make -C docs clean html

Open the generated page:

.. code-block:: text

   docs/build/html/index.html

Publish on ReadTheDocs
----------------------

The repository includes ``.readthedocs.yaml`` at the top level. To publish:

1. push the repository to GitHub;
2. import the repository on ReadTheDocs;
3. keep the default configuration file path ``.readthedocs.yaml``;
4. trigger a build.

This route does not require enabling GitHub Pages in the repository settings.
