"""Sphinx configuration of the CatEMC documentation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = 'CatEMC'
author = 'Shuang Han'
copyright = '2025, Shuang Han'
release = '1.0.0'
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx_design',
]

# -- API reference ----------------------------------------------------------
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
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}
intersphinx_timeout = 10

# -- Sources ----------------------------------------------------------------
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
numfig = True
math_numfig = True
math_number_all = False
math_eqref_format = 'Eq. {number}'
pygments_style = 'tango'
pygments_dark_style = 'monokai'

# The prompt of console blocks is not copied by the copy button.
copybutton_prompt_text = r'\$ '
copybutton_prompt_is_regexp = True

mathjax3_config = {
    'tex': {
        'macros': {
            'kB': r'k_{\mathrm{B}}',
            'NM': r'\overline{\mathrm{NM}}',
            'dmu': [r'\Delta\mu_{#1\to #2}', 2],
        },
    },
}

# -- HTML -------------------------------------------------------------------
html_theme = 'furo'
html_title = 'CatEMC'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/catemc_logo.svg'
html_favicon = '_static/catemc_icon.svg'
html_theme_options = {
    'navigation_with_keys': True,
    'sidebar_hide_name': True,
    'source_repository': 'https://github.com/basf/CatEMC',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
    'light_css_variables': {
        'color-brand-primary': '#1f5fbf',
        'color-brand-content': '#1f5fbf',
        'color-brand-visited': '#5b4bc4',
    },
    'dark_css_variables': {
        'color-brand-primary': '#7fb2ff',
        'color-brand-content': '#7fb2ff',
        'color-brand-visited': '#b3a6ff',
    },
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/basf/CatEMC',
            'html': (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                '3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01'
                '-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13'
                '-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 '
                '2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31'
                '-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 '
                '1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 '
                '1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 '
                '3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55'
                '.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'),
            'class': '',
        },
    ],
}
