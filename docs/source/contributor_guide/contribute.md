# Contributing to pyRTE-RRTMGP

Thanks for considering making a contribution to pyRTE-RRTMGP!

This document contains information about the many ways you can support pyRTE-RRTMGP, including how to report bugs, propose new features, set up a local development environment, contribute to the documentation, run and update tests, and make a new release.

## How to Report a Bug

Please file a bug report on the [GitHub repository](https://github.com/earth-system-radiation/pyRTE-RRTMGP/issues/new/choose).
If possible, your issue should include a [minimal, complete, and verifiable example](https://stackoverflow.com/help/mcve) of the bug.

## How to Propose a New Feature

Please file a feature request on the [GitHub page](https://github.com/earth-system-radiation/pyRTE-RRTMGP/issues/new/choose).

## How to Contribute to the Documentation

The documentation uses [Sphinx](https://www.sphinx-doc.org/en/master/) with [MystMD](https://myst-parser.readthedocs.io/en/latest/) for Markdown support. The source for the documentation is in the `docs` directory.

To build the documentation locally, first install the required documentation dependencies (optimally in a dedicated virtual environment):

```bash
pip install -r docs/requirements-doc.txt
```

Then, build the documentation:

```bash
cd docs
make html
```

The built documentation will be located in `docs/build/html`.

## How to Set up a Local Development Environment

<!-- TBD: DEV install instead of user install -->

Please follow the instructions for [installing pyRTE-RRTMTP with pip or conda in the documentation](https://pyrte-rrtmgp.readthedocs.io/en/latest/user_guide/installation.html).

### How to Set up Pre-Commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to maintain consistent code formatting (using [flake8](https://flake8.pycqa.org/en/latest/), [isort](https://pycqa.github.io/isort/), and [black](https://black.readthedocs.io/en/stable/)) and run static type checking with [mypy](https://github.com/python/mypy) before each commit. To set up pre-commit hooks, first install the required dependencies:

```bash
pip install pre-commit
pre-commit install
```

After installing pre-commit, you can also run the checks manually with `pre-commit run --all-files`.

Please ensure that new contributions pass the formatting and mypy checks before submitting pull requests.

### How to Run and Update Tests

pyRTE-RRTMTP uses [pytest](https://docs.pytest.org/en/stable/) for testing. To run the tests, first install the required testing dependencies (optimally in a dedicated virtual environment):

```bash
pip install -r tests/requirements-test.txt
```

Then, run the tests:

```bash
pytest tests
```

## How to Contribute a Patch That Fixes a Bug

Please fork this repository, branch from `main`, make your changes, and open a
GitHub [pull request](https://github.com/earth-system-radiation/pyRTE-RRTMTP/pulls)
against the `main` branch.

## How to Contribute New Features

Please fork this repository, branch from `main`, make your changes, and open a
GitHub [pull request](https://github.com/earth-system-radiation/pyRTE-RRTMTP/pulls)
against the `main` branch.

Pull Requests for new features should include tests and documentation.

## How to Make a New Release

For maintainers:

First, check for common issues that might arise between RTE-RRTMGP (Fortran) and pyRTE-RRTMGP (Python), use the `check_binds.py` script. This script will compare the functions in the Fortran code with the functions in the Python bindings. The script will output any missing functions or functions that have been removed.

```bash
python check_binds.py --c_headers /path/to/rrtmgp_kernels.h /path/to/rte_kernels.h --pybind /path/to/pybind_interface.cpp
```

For more details, see [](./fortran-compatibility.md).

After checking the compatibility, follow these steps to make a new release:

1. Update the version number in `pyproject.toml` and `conda.recipe/meta.yaml`. Also update `CITATION.cff` as necessary (for [Zenodo integration](https://zenodo.org/records/1117789)).
2. Create a new release off the `main` branch on GitHub, using the "Draft a new release" button in [https://github.com/earth-system-radiation/pyRTE-RRTMGP/releases](https://github.com/earth-system-radiation/pyRTE-RRTMGP/releases).
3. Create a new tag with the version number, and use the "Generate release notes" button to create the release notes.
4. Review and update the the release notes as necessary, publish the release, and set it as the latest release.

A PR to update the conda forge recipe should be created automatically by [regro-cf-autotick-bot](https://conda-forge.org/docs/maintainer/updating_pkgs/#pushing-to-regro-cf-autotick-bot-branch).

The documentation on [https://pyrte-rrtmgp.readthedocs.io/](https://pyrte-rrtmgp.readthedocs.io/) will update automatically. To make changes to the build process and other aspects of the readthedocs configuration, see the `.readthedocs.yml` file.
