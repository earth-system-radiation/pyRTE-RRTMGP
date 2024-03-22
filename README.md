# pyRTE-RRTMGP

This is the repository for the pyRTE-RRTMGP project.

This project provides a **Python interface to the [RTE+RRTMGP](https://earth-system-radiation.github.io/rte-rrtmgp/)
Fortran software package**.

The RTE+RRTMGP package is a set of libraries for for computing radiative fluxes in
planetary atmospheres. RTE+RRTMGP is described in a
[paper](https://doi.org/10.1029/2019MS001621) in
[Journal of Advances in Modeling Earth Systems](http://james.agu.org/).

## Project Status

This project is currently in an early stage of development.

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to this project!

## Functions Currently Available

The goal of this project is to provide a Python interface to the most important
Fortran functions in the RTE+RRTMGP package.

Currently, the following functions are available in the `pyrte` package:

### RTE Functions (WIP)

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
| `rte_inc_1scalar_by_nstream_bybnd`      |   🔲   |
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

### RRTMGP Functions

| Function name                           | Covered |
|-----------------------------------------|:-------:|
| `rrtmgp_interpolation`                  |   ✅   |
| `rrtmgp_compute_tau_absorption`         |   ✅   |
| `rrtmgp_compute_tau_rayleigh`           |   ✅   |
| `rrtmgp_compute_Planck_source`          |   ✅   |

## Setup Instructions

The current code in this repo are early experiments with the goal of exploring details fo the Fortran works and testing different options for creating an interface with C.
Later on, we will use this knowledge to bind it with Python

### Prerequisites

pyRTE-RRTMGP is currently only tested on x86_64 architecture with Linux and macOS.

To build and install the package, you need the conda package manager. If you don't have conda installed, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).

The package source code is hosted [on GitHub](https://github.com/earth-system-radiation/pyRTE-RRTMGP) and uses git submodules to include the [RTE+RRTMGP Fortran software](https://earth-system-radiation.github.io/rte-rrtmgp/). The easiest way to install pyRTE-RRTMGP is to use `git`. You can install git from [here](https://git-scm.com/downloads).

### Installation with conda (recommended)

1. Clone the repository:

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

2. Update the submodules:

    ```bash
    git submodule update --init --recursive
    ```

3. Make sure you have conda installed. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).
    To make sure your conda setup is working, run the command below:

    ```bash
    conda --version
    ```

    If this runs without errors, you are good to go.

4. Install the conda build requirements (if you haven't already):

    ```bash
    conda install conda-build conda-verify
    ```

5. Build the conda package locally:

    ```bash
    conda build conda.recipe
    ```

6. Install the package in your current conda environment:

    ```bash
    conda install -c ${CONDA_PREFIX}/conda-bld/ pyrte
    ```

    Note: This will install the package in your current conda environment. If you want to install the package in a different environment, activate your environment before running the `conda install` command above.

### Installation with pip

You also have the option to build and install the package with pip. This might work on additional, untested architectures (such as macOS on M1). However, this is not recommended as it requires you to have a working Fortran compiler and other prerequisites installed on your system.

* To install with pip, you first need to clone the repo (``git clone git@github.com:earth-system-radiation/pyRTE-RRTMGP.git``) and update the submodules (``git submodule update --init --recursive``) as described in the conda installation instructions above.

    ``` bash
    git submodule update --init --recursive
    ```

* Install dependencies in your operating system.

    With Ubuntu, for example, use:

    ``` bash
    sudo apt install -y \
        libnetcdff-dev \
        gfortran-10 \
        python3-dev \
        cmake
    ```

    On other systems, you might be able to install the necessary dependencies with a package manager like `brew`.

* Compile the Fortran code and build and install the Python package in your current environment with:

    ``` bash
    pip install .
    ```

    For development purposes, you can install the package in editable mode: ``pip install -e .``.

Once built, the module will be located in a folder called `pyrte`

## Pytest Setup Instructions

* Go to the 'tests' folder: `cd tests/`
* Install the test prerequisites (if you haven't already) by running `pip3 install -r requirements-test.txt`
* Run `pytest`
