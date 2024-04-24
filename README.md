[![CI passing](https://github.com/earth-system-radiation/pyRTE-RRTMGP/actions/workflows/conda.yml/badge.svg)](https://github.com/earth-system-radiation/pyRTE-RRTMGP/actions/workflows/conda.yml)
[![Documentation Status](https://readthedocs.org/projects/pyrte-rrtmgp/badge/?version=latest)](https://pyrte-rrtmgp.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10982460.svg)](https://doi.org/10.5281/zenodo.10982460)

# pyRTE-RRTMGP

This is the repository for the pyRTE-RRTMGP project.

This project provides a **Python interface to the [RTE+RRTMGP](https://earth-system-radiation.github.io/rte-rrtmgp/)
Fortran software package**.

The RTE+RRTMGP package is a set of libraries for for computing radiative fluxes in
planetary atmospheres. RTE+RRTMGP is described in a
[paper](https://doi.org/10.1029/2019MS001621) in
[Journal of Advances in Modeling Earth Systems](http://james.agu.org/).

## Documentation

Documentation for the pyRTE-RRTMGP package is available on [Read the Docs](https://pyrte-rrtmgp.readthedocs.io/en/latest/).

## Project Status

This project is currently in an **early stage of development**.

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to this project!

## Functions Currently Available

The goal of this project is to provide a Python interface to the most important
Fortran functions in the RTE+RRTMGP package.

Currently, the following functions are available in the `pyrte_rrtmgp` package:

### RTE Functions (WIP)

<!-- start-rte-functions-section -->

| Function name                           | Covered |
|-----------------------------------------|:-------:|
| **SHORTWAVE SOLVERS**                   |         |
| `rte_sw_solver_noscat`                  |   ✅   |
| `rte_sw_solver_2stream`                 |   ✅   |
| **LONGWAVE SOLVERS**                    |         |
| `rte_lw_solver_noscat`                  |   ✅   |
| `rte_lw_solver_2stream`                 |   ✅   |
| **OPTICAL PROPS - INCREMENT**           |         |
| `rte_increment_1scalar_by_1scalar`      |   ✅   |
| `rte_increment_1scalar_by_2stream`      |   ✅   |
| `rte_increment_1scalar_by_nstream`      |   ✅   |
| `rte_increment_2stream_by_1scalar`      |   ✅   |
| `rte_increment_2stream_by_2stream`      |   ✅   |
| `rte_increment_2stream_by_nstream`      |   ✅   |
| `rte_increment_nstream_by_1scalar`      |   ✅   |
| `rte_increment_nstream_by_2stream`      |   ✅   |
| `rte_increment_nstream_by_nstream`      |   ✅   |
| **OPTICAL PROPS - INCREMENT BYBND**     |         |
| `rte_inc_1scalar_by_1scalar_bybnd`      |   ✅   |
| `rte_inc_1scalar_by_2stream_bybnd`      |   ✅   |
| `rte_inc_1scalar_by_nstream_bybnd`      |   ✅   |
| `rte_inc_2stream_by_1scalar_bybnd`      |   ✅   |
| `rte_inc_2stream_by_2stream_bybnd`      |   ✅   |
| `rte_inc_2stream_by_nstream_bybnd`      |   ✅   |
| `rte_inc_nstream_by_1scalar_bybnd`      |   ✅   |
| `rte_inc_nstream_by_2stream_bybnd`      |   ✅   |
| `rte_inc_nstream_by_nstream_bybnd`      |   ✅   |
| **OPTICAL PROPS - DELTA SCALING**       |         |
| `rte_delta_scale_2str_k`                |   ✅   |
| `rte_delta_scale_2str_f_k`              |   ✅   |
| **OPTICAL PROPS - SUBSET**              |         |
| `rte_extract_subset_dim1_3d`            |   ✅   |
| `rte_extract_subset_dim2_4d`            |   ✅   |
| `rte_extract_subset_absorption_tau`     |   ✅   |
| **Fluxes - Reduction**                  |         |
| `rte_sum_broadband`                     |   ✅   |
| `rte_net_broadband_full`                |   ✅   |
| `rte_net_broadband_precalc`             |   ✅   |
| `rte_sum_byband`                        |   🔲   |
| `rte_net_byband_full`                   |   🔲   |
| **Array Utilities**                     |         |
| `zero_array_1D`                         |   ✅   |
| `zero_array_2D`                         |   ✅   |
| `zero_array_3D`                         |   ✅   |
| `zero_array_4D`                         |   ✅   |

<!-- end-rte-functions-section -->

### RRTMGP Functions

<!-- start-rrtmgp-functions-section -->

| Function name                           | Covered |
|-----------------------------------------|:-------:|
| `rrtmgp_interpolation`                  |   ✅   |
| `rrtmgp_compute_tau_absorption`         |   ✅   |
| `rrtmgp_compute_tau_rayleigh`           |   ✅   |
| `rrtmgp_compute_Planck_source`          |   ✅   |

<!-- end-rrtmgp-functions-section -->

## Setup Instructions

> **Note**:
> The code in this repository is a work in progress. The Python API is not yet stable and is subject to change.

<!-- start-setup-section -->

### Prerequisites

As pyRTE-RRTMGP is built in install time it will require a compatible Fortran compiler, a C++ compiler and CMake to be installed on your system. The package is compatible with posix systems and is tested on Linux and macOS using the GNU Fortran compiler (gfortran) and the GNU C++ compiler (g++). The package should also work with the Intel Fortran compiler (ifort) but it was not tested with it.

The package source code is hosted [on GitHub](https://github.com/earth-system-radiation/pyRTE-RRTMGP). The easiest way to install pyRTE-RRTMGP is to use `git`. You can install git from [here](https://git-scm.com/downloads).

### Installation with conda (recommended)

1. **Clone the repository**:

    ```bash
    git clone git@github.com:earth-system-radiation/pyRTE-RRTMGP.git
    ```

    or

    ```bash
    git clone https://github.com/earth-system-radiation/pyRTE-RRTMGP.git
    ```

    After cloning the repository, enter the repository directory:

    ```bash
    cd pyRTE-RRTMGP
    ```

2. **Update the submodules**:

    ```bash
    git submodule update --init --recursive
    ```

3. **Make sure you have conda installed**. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).
    To make sure your conda setup is working, run the command below:

    ```bash
    conda --version
    ```

    If this runs without errors, you are good to go.

4. **Install the conda build requirements** (if you haven't already):

    ```bash
    conda install conda-build conda-verify
    ```

5. **Build the conda package locally**:

    ```bash
    conda build conda.recipe
    ```

6. **Install the package** in your current conda environment:

    ```bash
    conda install -c ${CONDA_PREFIX}/conda-bld/ pyrte_rrtmgp
    ```

    Note: This will install the package in your current conda environment. If you want to install the package in a different environment, activate your environment before running the `conda install` command above.

### Installation with pip

You also have the option to build and install the package with pip. This should work with macOS and Linux systems.

#### Mac OS

1. **Install the requirements** On MacOS systems you can use `brew` to install the dependencies as following `brew install git gcc cmake`, but you can also install de requirements using other package managers, such as conda.

2. **Install the package** Then you can install the package directly from the git repository `pip install git+https://github.com/earth-system-radiation/pyRTE-RRTMGP`

#### Debian/Ubuntu

1. **Install the requirements** On Debian/Ubuntu systems you can use `apt` to install the dependencies as following `sudo apt install build-essential gfortran cmake git`, but you can also install de requirements using other package managers, such as conda.

2. **Install the package** Then you can install the package directly from the git repository `pip install git+https://github.com/earth-system-radiation/pyRTE-RRTMGP`

Other linux distributions should also support the installation of the package, you just need to install the dependencies using the package manager of your distribution.

For development purposes, you can install the package in editable mode: ``pip install -e .``.

Once built, the module will be located in a folder called `pyrte_rrtmgp`

<!-- end-setup-section -->

## Pytest Setup Instructions

* Go to the 'tests' folder: `cd tests/`
* Install the test prerequisites (if you haven't already) by running `pip3 install -r requirements-test.txt`
* Run `pytest tests`
