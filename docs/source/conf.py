import sys
import os

# -- Path setup --------------------------------------------------------------
# Add the repo root so Sphinx can import `catemc` (installed in editable mode)
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
project = 'CatEMC'
author = 'Shuang Han'
release = '1.0.0'
version = release


# -- General configuration ---------------------------------------------------
extensions = [
'sphinx.ext.autodoc',
'sphinx.ext.autosummary',
'sphinx.ext.napoleon',
'sphinx.ext.viewcode',
'sphinx.ext.todo',
'sphinx_autodoc_typehints',
'sphinxcontrib.programoutput',
'sphinx_wagtail_theme',
]


autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_admonition_for_examples = True


# Mock imports to avoid heavy deps failing the docs build (CI-friendly)
# Adjust list if you know other heavy modules that Sphinx cannot import on the build host
autodoc_mock_imports = [
'mace',
'acat',
'autoadsorbate',
]


# Source and templates
source_suffix = {'.rst': 'restructuredtext'}
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- HTML options ------------------------------------------------------------
html_theme = 'sphinx_wagtail_theme'
html_theme_options = {
'navigation_depth': 3,
'show_related': True,
}
html_static_path = ['_static']
html_css_files = ['custom.css']


# show todos
todo_include_todos = True


# autodoc options
autodoc_typehints = 'both'
autodoc_member_order = 'bysource'


# last updated
html_last_updated_fmt = '%b %d, %Y'
