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

Documentation for pyRTE-RRTMGP is available on [Read the Docs](https://pyrte-rrtmgp.readthedocs.io/en/latest/).

## Project Status

The project is currently under active development. We hope to have a stable first usable release by Apr 2025. 

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to this effort

> **Note**:
> The code in this repository is a work in progress. The Python API is not yet stable and is subject to change.

## Installing pyRTE-RRTMGP


### Installing with Conda (recommended)

<!-- start-installation-section -->

pyRTE-RRTMGP is available as a [conda package for Linux (x86_64)](https://anaconda.org/conda-forge/pyrte_rrtmgp). You can install it from the `conda-forge` channel:

```bash
conda install -c conda-forge pyrte_rrtmgp
```

This will install the package in your current conda environment. If you want to install the package in a different environment, activate your environment before running the `conda install` command above.

After installing the package, you can import it in your Python code:

```python
import pyrte_rrtmgp
```

For platforms other than Linux for x64 processors, you can build the package from source using the instructions in the [documentation](https://pyrte-rrtmgp.readthedocs.io/en/latest/user_guide/installation.html).

<!-- end-installation-section -->


### Building Locally

See the documentation for [instructions](https://pyrte-rrtmgp.readthedocs.io/en/latest/user_guide/installation.html).


## Running Tests

After installing the package, you can run the tests by executing the following command:

```bash
pyrte_rrtmgp run_tests
```
