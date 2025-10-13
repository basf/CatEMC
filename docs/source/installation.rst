Installation
============


Install directly from this repository (editable mode is recommended for development):


.. code-block:: bash


git clone https://github.com/basf/CatEMC.git
cd CatEMC
pip install -e .


Notes
~~~~~


* If you will run calculations that use heavy ML models (MACE), ensure those runtime dependencies are installed on the target machine where you will actually run EMC jobs (they are mocked during doc builds by default).
* If using ReadTheDocs, put the documentation-specific requirements (e.g. ``sphinx-wagtail-theme``) in ``docs/requirements.txt``.
