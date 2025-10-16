# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add src to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "BPDneo-CXR"
copyright = "2025, Philipp Flotho"
author = "Philipp Flotho"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # Markdown support
    "sphinx.ext.autodoc",  # Auto API docs
    "sphinx.ext.autosummary",  # Summary tables
    "sphinx.ext.napoleon",  # NumPy/Google docstrings
    "sphinx.ext.intersphinx",  # Cross-project links
    "sphinx.ext.viewcode",  # Source code links
    "sphinx_copybutton",  # Copy button for code
    "sphinx_design",  # UI components
    "sphinxcontrib.bibtex",  # Citations
    "sphinx_autodoc_typehints",  # Type hint support
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/FlowRegSuite/bpdneo-cxr-ui",
    "logo": {
        "text": "BPDneo-CXR",
    },
    "navbar_end": ["navbar-icon-links"],
    "show_nav_level": 2,
    "navigation_depth": 3,
}

# -- MyST Parser configuration -----------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # ::: fences
    "deflist",  # Definition lists
    "tasklist",  # Task lists
    "attrs_inline",  # Inline attributes
]

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Autodoc settings --------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autodoc_typehints = "description"
autosummary_generate = True

# -- Intersphinx mappings ----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- BibTeX configuration ----------------------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
