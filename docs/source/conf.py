# Configuration file for the Sphinx documentation builder.

import os
import sys
# For Python 3.11+, tomllib is standard. For 3.10, we might need tomli as a docs dependency.
# For simplicity with target >=3.10, let's assume tomllib or tomli is available.
try:
    import tomllib # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib # Python < 3.11, requires `pip install tomli`
    except ImportError:
        tomllib = None # Fallback if toml parser is not available

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'ScatterBrain'
copyright = '2023, [Your Name/Organization Name]. Licensed under CC-BY-NC-SA 4.0'
author = '[Your Name/Organization Name]'

# Get version from pyproject.toml
release = '0.0.1.dev0' # Default fallback
if tomllib:
    try:
        with open("../../pyproject.toml", "rb") as f:
            pyproject_data = tomllib.load(f)
        release = pyproject_data.get("project", {}).get("version", release)
    except FileNotFoundError:
        pass # Keep default if pyproject.toml not found (e.g. during some CI build steps)
else: # Fallback if no TOML parser
    try:
        from scatterbrain import __version__ as release_from_pkg
        release = release_from_pkg
    except ImportError:
        pass # Keep default fallback

version = '.'.join(release.split('.')[:2])


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',     # << Now correctly listed as a built-in extension
    'myst_parser',
    'sphinx_autodoc_typehints',
    'sphinx_rtd_theme',
]

# ... (rest of conf.py remains largely the same as before, but ensure it's compatible)
# Specifically, the intersphinx_mapping and napoleon settings are still valid.

# Napoleon settings (still valid and useful)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True # Changed to True for better formatting of examples
napoleon_use_admonition_for_notes = True   # Changed to True
napoleon_use_admonition_for_references = True # Changed to True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# MyST Parser options (ensure these are still current with myst_parser versions)
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3