Installation
============

Requirements
------------

CatEMC is a pure Python package. It depends on the following packages,
which are installed automatically with it:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Package
     - Used for
   * - `ASE <https://wiki.fysik.dtu.dk/ase/>`_
     - structures, calculators, optimizers, molecular dynamics and databases
   * - `ACAT <https://asm-dtu.gitlab.io/acat/>`_ (2.0.3 or newer)
     - random orderings, adsorption sites and the genetic operators that the
       EMC moves are built on
   * - `AutoAdsorbate <https://github.com/basf/autoadsorbate>`_
     - surface binding sites and placement of adsorbate fragments
   * - `pymbar <https://pymbar.readthedocs.io>`_
     - MBAR free energies and reweighting
   * - `PyYAML <https://pyyaml.org>`_
     - the configuration file of the command-line interface

Install from source
-------------------

.. code-block:: console

   $ git clone https://github.com/basf/CatEMC.git
   $ cd CatEMC
   $ pip install -e .

The editable install also creates the ``catemc`` command. Check it with

.. code-block:: console

   $ catemc --version
   catemc 1.0.0

Calculator
----------

CatEMC works with any ASE calculator. The tutorials use the MACE-MPA-0
foundation model of `MACE <https://github.com/ACEsuit/mace>`_, which the
examples import directly:

.. code-block:: python

   from mace.calculators import mace_mp

   calculator = mace_mp(model='medium-mpa-0', dispersion=False,
                        device='cuda', default_dtype='float64')

``mace_mp`` downloads the model at its first call. Call it once before the
first run with many worker processes.

The :doc:`quickstart` uses the EMT calculator that ships with ASE, so you can
try CatEMC without a machine-learning potential.

.. tip::

   When many worker processes share one GPU, limit the CPU threads of each
   worker so that they do not compete with each other:

   .. code-block:: console

      $ export OMP_NUM_THREADS=1
      $ export MKL_NUM_THREADS=1
      $ export OPENBLAS_NUM_THREADS=1
      $ export NUMEXPR_NUM_THREADS=1

Run the tests
-------------

The tests of the configuration layer need no calculator and take a second:

.. code-block:: console

   $ pip install pytest
   $ pytest tests

Build this documentation
------------------------

.. code-block:: console

   $ pip install -r docs/requirements.txt
   $ sphinx-build -b html docs/source docs/build/html
