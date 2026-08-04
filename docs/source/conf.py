from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = 'CatEMC'
author = 'Shuang Han'
release = '1.0.0'
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

autodoc_mock_imports = [
    'acat',
    'autoadsorbate',
    'mace',
    'metatomic_ase',
    'pymbar',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_title = 'CatEMC'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'navigation_with_keys': True,
    'sidebar_hide_name': False,
}
