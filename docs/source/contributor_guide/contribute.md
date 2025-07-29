# Contributing to pyRTE-RRTMGP

Thanks for considering making a contribution to pyRTE-RRTMGP!

This document contains information about the many ways you can support pyRTE-RRTMGP, including how to report bugs, propose new features, set up a local development environment, contribute to the documentation, run and update tests, and make a new release.

## How to Report a Bug

Please file a bug report on the [GitHub repository](https://github.com/earth-system-radiation/pyRTE-RRTMGP/issues/new/choose).
If possible, your issue should include a [minimal, complete, and verifiable example](https://stackoverflow.com/help/mcve) of the bug.

## How to Contribute a Patch That Fixes a Bug

Please fork this repository, branch from `main`, make your changes, and open a
GitHub [pull request](https://github.com/earth-system-radiation/pyRTE-RRTMTP/pulls)
against the `main` branch.

## How to Propose a New Feature

Please file a feature request on the [GitHub page](https://github.com/earth-system-radiation/pyRTE-RRTMGP/issues/new/choose).

## How to Contribute New Features

Please fork this repository, branch from `main`, make your changes, and open a
GitHub [pull request](https://github.com/earth-system-radiation/pyRTE-RRTMTP/pulls)
against the `main` branch.

Pull Requests for new features should include tests and documentation.

## How to Contribute to the Documentation

The documentation uses [Sphinx](https://www.sphinx-doc.org/en/master/) with [MystMD](https://myst-parser.readthedocs.io/en/latest/) for Markdown support. The source for the documentation is in the `docs` directory.

To build the documentation locally, first install the required documentation dependencies with ``mamba``:

```bash
mamba env create -f docs/environment-docs.yml
```
which will install pyRTE's dependencies and the package itself in editable mode, which means that any changes you make to the package will be reflected in the documentation build right after you make them, without needing to reinstall the package.

Enter the newly created environment:

```bash
mamba activate pyrte_rrtmgp_docs
```

and build the documentation:

```bash
cd docs
make html
```

The built documentation will be located in `docs/build/html`. You can use a web browser to open the `index.html` file in this directory to view the documentation.

The documentation is automatically built and deployed to [Read the Docs](https://pyrte-rrtmgp.readthedocs.io/) whenever a new commit is pushed to the `main` branch. The configuration for the Read the Docs build is in the `.readthedocs.yml` file.

## How to Make a New Release (for project maintainers)

For maintainers:

First, check for common issues that might arise between RTE-RRTMGP (Fortran) and pyRTE-RRTMGP (Python), use the `check_binds.py` script. This script will compare the functions in the Fortran code with the functions in the Python bindings. The script will output any missing functions or functions that have been removed.

```bash
python check_binds.py --c_headers /path/to/rrtmgp_kernels.h /path/to/rte_kernels.h --pybind /path/to/pybind_interface.cpp
```

For more details, see [](./fortran-compatibility.md).

After checking the compatibility, follow these steps to make a new release:

1. Update the version number (using https://semver.org/) in `pyproject.toml` and `conda.recipe/meta.yaml`. Also update `CITATION.cff` as necessary (for [Zenodo integration](https://zenodo.org/records/1117789)). e.g. ``version = "1.1.0"`` in pyproject.toml and ``version: 1.1.0`` in meta.yaml and CITATION.cff.
2. Create a new tag with the version number off the `main` branch on GitHub, adding a ``v`` before the version number (e.g. `v1.1.0`).
3. Create a new release, using the "Draft a new release" button in [https://github.com/earth-system-radiation/pyRTE-RRTMGP/releases](https://github.com/earth-system-radiation/pyRTE-RRTMGP/releases) adding a ``v`` before the release (e.g. `v1.1.0`).
4. Review and update the the release notes as necessary, publish the release, and set it as the latest release.

A PR to update the conda forge recipe should be created automatically by [regro-cf-autotick-bot](https://conda-forge.org/docs/maintainer/updating_pkgs/#pushing-to-regro-cf-autotick-bot-branch). It can take several hours for the bot to detect the update and create the PR!

The feedstock for the conda package is located at [https://github.com/conda-forge/pyRTE_RRTMGP-feedstock](https://github.com/conda-forge/pyRTE_RRTMGP-feedstock). Once the PR on the feedstock repo passes all tests, one of the pyRTE-RRTMGP maintainers can merge the PR and the new version of the package will be available on conda-forge.

The documentation on [https://pyrte-rrtmgp.readthedocs.io/](https://pyrte-rrtmgp.readthedocs.io/) will update automatically. To make changes to the build process and other aspects of the readthedocs configuration, see the `.readthedocs.yml` file.
