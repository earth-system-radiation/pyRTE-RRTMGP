[![CI passing](https://github.com/earth-system-radiation/pyRTE-RRTMGP/actions/workflows/conda.yml/badge.svg)](https://github.com/earth-system-radiation/pyRTE-RRTMGP/actions/workflows/conda.yml)
[![Mypy](https://github.com/earth-system-radiation/pyRTE-RRTMGP/actions/workflows/mypy.yml/badge.svg)](https://github.com/earth-system-radiation/pyRTE-RRTMGP/actions/workflows/mypy.yml)
[![Documentation Status](https://readthedocs.org/projects/pyrte-rrtmgp/badge/?version=latest)](https://pyrte-rrtmgp.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10982460.svg)](https://doi.org/10.5281/zenodo.10982460)

# pyRTE-RRTMGP

This is the repository for the pyRTE-RRTMGP project.

This project provides a **Python interface to the [RTE+RRTMGP](https://earth-system-radiation.github.io/rte-rrtmgp/)
Fortran software package**. RTE+RRTMGP package is a set of libraries for for computing radiative fluxes in
planetary atmospheres.

## Documentation

Technical documentation for pyRTE-RRTMGP including installation instructions is available on [Read the Docs](https://pyrte-rrtmgp.readthedocs.io/en/latest/).

User documentation including explanations, tutorials, and examples is available on [Github](https://earth-system-radiation.github.io/pyRTE-RRTMGP)

## Project Status

The project is currently under active development. See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute.

> **Note**:
> The Python API is not entirely stable and is subject to change.

## Installing pyRTE-RRTMGP

### Installing with Mamba (recommended)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyrte_rrtmgp.svg)](https://anaconda.org/conda-forge/pyrte_rrtmgp)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pyrte_rrtmgp.svg)](https://anaconda.org/conda-forge/pyrte_rrtmgp)

<!-- start-installation-section -->

pyRTE-RRTMGP is available as a [conda package for Linux (x86_64) and MacOS](https://anaconda.org/conda-forge/pyrte_rrtmgp). You can install it from the `conda-forge` channel or, better, use `mamba`:

```bash
mamba install pyrte_rrtmgp
```

This will install the package in your current conda environment. If you want to install the package in a different environment, activate your environment before running the `conda install` command above.

After installing the package, you can import it in your Python code:

```python
import pyrte_rrtmgp
```

To verify your installation, you can run a set of tests with the following command:

```bash
pyrte_rrtmgp run_tests
```

### Instlling with `pip`

pyRTE-RRTMGP can also be installed for development through `pip`. The `scikit-build-core` build backend is used by this package to compile and include files from RTE-RRTMGP. Before installation, please make sure your C, CXX, and Fortran compilers are accessible to the environment that will be used for installation. To perform the installation using `pip` run from the top level of the repository:

```bash
pip install -v .
```

To test the installation you can run:

```bash
pyrte_rrtmgp run_tests
```

<!-- end-installation-section -->

For platforms not supported by the `conda` package and alternatives to installing with conda, see the [installation instructions in the documentation](https://pyrte-rrtmgp.readthedocs.io/en/latest/how_to/installation.html) and the [Contributor Guide](https://pyrte-rrtmgp.readthedocs.io/en/latest/how-to/installation-local-dev.html).
