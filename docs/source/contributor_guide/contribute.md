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

(local-install)=
## How to Set up a Local Development Environment

To build and test the package locally, you need to install two sets of dependencies: the system dependencies and the package itself (using [pip in "editable" mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)).

Before installing the package, you should **create a virtual environment** to avoid conflicts with other packages. You can create a virtual environment in a folder of your choice using the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then, follow the instructions below for your respective platform to **install the system dependencies**:

* Debian/Ubuntu: On Debian/Ubuntu systems, you can use a tool like `apt` to install the dependencies: ``sudo apt install build-essential gfortran cmake git``
* Other Linux distributions: Install the dependencies using the package manager of your distribution.
* Mac OS: On MacOS systems you can use a tool like `brew` to install the dependencies: ``brew install git gcc cmake``

Next, **download the source code**. Clone the repository to your local machine using

```bash
git clone https://github.com/earth-system-radiation/pyRTE-RRTMGP.git
```

After cloning the repository, **enter the repository directory**:

```bash
cd pyRTE-RRTMGP
```

Then, **install the package** in "editable" mode:

```bash
pip install -e .
```

### How to Set up Pre-Commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to maintain consistent code formatting (using [flake8](https://flake8.pycqa.org/en/latest/), [isort](https://pycqa.github.io/isort/), and [black](https://black.readthedocs.io/en/stable/)) and run static type checking with [mypy](https://github.com/python/mypy) before each commit. 

To set up pre-commit hooks, first install the required dependencies:

```bash
pip install pre-commit
```

Then, make sure you are in the root directory of your local clone of the repository and
install the pre-commit hooks:

```bash
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

## How to Locally Build and Test the Conda Package

Before creating a new release and updating the conda package, you should test the package locally to ensure that it builds correctly. To build the conda package locally, follow these steps:

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone https://github.com/earth-system-radiation/pyRTE-RRTMGP.git
    ```

    After cloning the repository, enter the repository directory:

    ```bash
    cd pyRTE-RRTMGP
    ```

2. **Make sure you have conda installed**. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).
    To make sure your conda setup is working, run the command below:

    ```bash
    conda --version
    ```

    If this runs without errors, you are good to go.

3. **Install the conda build requirements** (if you haven't already):

    ```bash
    conda install conda-build conda-verify
    ```

4. **Build the conda package locally**:
    ```bash
    conda build conda.recipe
    ```

5. **Install the package** in your current conda environment:
    ```bash
    conda install -c ${CONDA_PREFIX}/conda-bld/ pyrte_rrtmgp
    ```

    ```{note}
    This will install the package in your current conda environment. If you want to install the package in a different environment, activate your environment before running the `conda install` command above.
    ```

The recipe for the conda package is located in the `conda.recipe` directory. The recipe contains the metadata for the package, including the dependencies and the build instructions.

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
