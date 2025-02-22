# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime as dt
import os
import sys
import tomllib

sys.path.insert(0, os.path.abspath("../../"))


def get_version_from_toml() -> str:
    if os.path.isfile("../../pyproject.toml"):
        # read pyproject.toml
        with open("../../pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        # get version from pyproject.toml
        version = pyproject.get("project", {}).get("version", {})
    else:
        version = None
    return version if version else "dev"


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyRTE-RRTMGP"
copyright = f"{dt.datetime.now().year}, Atmospheric and Environmental Research"
author = "Atmospheric and Environmental Research"
version = get_version_from_toml()
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

autodoc_mock_imports = [
    "pyrte_rrtmgp.pyrte_rrtmgp",
    "numpy",
    "xarray",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "display_version": True,
}
