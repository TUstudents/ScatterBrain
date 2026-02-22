# Configuration file for the Sphinx documentation builder.

import os
import sys

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
project = "ScatterBrain"
copyright = "2026, Johannes Poms. Licensed under CC-BY-NC-SA 4.0"
author = "Johannes Poms"

release = "0.0.1.dev0"  # default fallback
if tomllib is not None:
    try:
        with open(os.path.join(os.path.dirname(__file__), "../../pyproject.toml"), "rb") as f:
            pyproject_data = tomllib.load(f)
        release = pyproject_data.get("project", {}).get("version", release)
    except FileNotFoundError:
        pass
else:
    try:
        from scatterbrain import __version__ as release  # type: ignore[assignment]
    except ImportError:
        pass

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- MyST settings -----------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path: list = []

# -- autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
