"""Configuration file for the Sphinx documentation builder."""

import datetime as dt
import os
import tomllib

# import sys
# sys.path.insert(0, os.path.abspath("../../"))


def get_version_from_toml() -> str:
    """Get the version from the pyproject.toml file."""
    if os.path.isfile("../../pyproject.toml"):
        # read pyproject.toml
        with open("../../pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        # get version from pyproject.toml
        version = pyproject.get("project", {}).get("version", {})
    else:
        version = None
    return version if version else "dev"


def get_python_requires_from_toml() -> str:
    """Get the version from the pyproject.toml file."""
    if os.path.isfile("../../pyproject.toml"):
        # read pyproject.toml
        with open("../../pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        # get version from pyproject.toml
        python_requires = pyproject.get("project", {}).get("requires-python", "")
    else:
        python_requires = None
    return python_requires


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyRTE-RRTMGP"
copyright = f"{dt.datetime.now().year}, Atmospheric and Environmental Research"
author = "Atmospheric and Environmental Research"
version = get_version_from_toml()
release = version
python_requires = get_python_requires_from_toml()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "substitution",
]

myst_substitutions = {
    "python_requires": python_requires,
    "version": version,
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]  # not needed for now

html_theme_options = {
    "display_version": True,
}
