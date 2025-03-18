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

Building and testing the package locally requires [conda](https://docs.conda.io) to be installed on your system. You can install conda by following the instructions [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

Building pyRTE-RRTMGP locally **currently works with Linux (Ubuntu) and macOS (Intel/ARM) only**.

Follow the instructions below to set up a local development environment:

1. Once `conda` is available on your system, you should **create a new conda environment** to avoid conflicts with other packages. You should specify a Python version that works with the version of pyRTE-RRTMGP you want to work on (currently Python *{{ python_requires }}*). For example:

    ```bash
    conda create -n pyrte_rrtmgp_dev python=3.12
    ```

2. **Clone the GitHub repository**:

    ```bash
    git clone https://github.com/earth-system-radiation/pyRTE-RRTMGP.git
    ```

3. After cloning the repository, **enter the repository directory**:

    ```bash
    cd pyRTE-RRTMGP
    ```

4. Make sure you **have a C++ compiler available on your system**.

    On Debian/Ubuntu systems, you can use a tool like `apt` to install the dependencies:

    ```bash
    sudo apt install build-essential
    ```

    On macOS systems, you can use a tool like `brew` to install a compiler:

    ```bash
    brew install gcc
    ```

    You can also use conda to install a compiler, for example:

    ```bash
    conda install -c conda-forge gcc_linux-64
    ```

4. **Install the main RTE-RRTMGP (Fortran) package** into the conda environment:

    ```bash
    conda install -c conda-forge rte_rrtmgp
    ```

    See the [RTE-RRTMGP GitHub repository](https://github.com/earth-system-radiation/rte-rrtmgp) for more information about the Fortran package.

5. **Install the Ninja build system** into the conda environment:

    ```bash
    conda install -c conda-forge ninja
    ```

5. Finally, **install the package** in ["editable" mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs):

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

Before creating a new release and updating the conda package, you should test the package locally to ensure that it builds correctly.

To build the conda package locally, first set up a local development environment as described in the {ref}`local-install` section above.

1. Make sure your local development environment is active (e.g. `conda activate pyrte_rrtmgp_dev`). Before you can build the conda package locally, you need to **install the conda build requirements** (if they aren't already on your system):

    ```bash
    conda install conda-build conda-verify
    ```

4. **Build the conda package locally**:

    ```bash
    conda build conda.recipe
    ```

5. **Install the locally built package** in your current conda environment:

    ```bash
    conda install -c ${CONDA_PREFIX}/conda-bld/ pyrte_rrtmgp
    ```

    ```{note}
    Make sure to remove any other versions of the package that might be installed in your environment (e.g. if you used `pip install -e .` before)!
    ```

The recipe for the conda package is located in the `conda.recipe` directory in the GitHub repository. This recipe contains the metadata for the package, including the dependencies and the build instructions.

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

1. Update the version number (using https://semver.org/) in `pyproject.toml` and `conda.recipe/meta.yaml`. Also update `CITATION.cff` as necessary (for [Zenodo integration](https://zenodo.org/records/1117789)). e.g. ``version = "1.1.0"`` in pyproject.toml and ``version: 1.1.0`` in meta.yaml and CITATION.cff.
2. Create a new tag with the version number off the `main` branch on GitHub, adding a ``v`` before the version number (e.g. `v1.1.0`).
3. Create a new release, using the "Draft a new release" button in [https://github.com/earth-system-radiation/pyRTE-RRTMGP/releases](https://github.com/earth-system-radiation/pyRTE-RRTMGP/releases) adding a ``v`` before the release (e.g. `v1.1.0`).
4. Review and update the the release notes as necessary, publish the release, and set it as the latest release.

A PR to update the conda forge recipe should be created automatically by [regro-cf-autotick-bot](https://conda-forge.org/docs/maintainer/updating_pkgs/#pushing-to-regro-cf-autotick-bot-branch). It can take several hours for the bot to detect the update and create the PR!

The feedstock for the conda package is located at https://github.com/conda-forge/pyRTE_RRTMGP-feedstock

Once the PR on the feedstock repo passes all tests, one of the pyRTE-RRTMGP maintainers can merge the PR and the new version of the package will be available on conda-forge.

The documentation on [https://pyrte-rrtmgp.readthedocs.io/](https://pyrte-rrtmgp.readthedocs.io/) will update automatically. To make changes to the build process and other aspects of the readthedocs configuration, see the `.readthedocs.yml` file.
