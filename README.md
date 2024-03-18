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
| `rte_increment_1scalar_by_nstream`      |   🔲   |
| `rte_increment_2stream_by_1scalar`      |   🔲   |
| `rte_increment_2stream_by_2stream`      |   🔲   |
| `rte_increment_2stream_by_nstream`      |   🔲   |
| `rte_increment_nstream_by_1scalar`      |   🔲   |
| `rte_increment_nstream_by_2stream`      |   🔲   |
| `rte_increment_nstream_by_nstream`      |   🔲   |
| **OPTICAL PROPS - INCREMENT BYBND**     |         |
| `rte_inc_1scalar_by_1scalar_bybnd`      |   🔲   |
| `rte_inc_1scalar_by_2stream_bybnd`      |   🔲   |
| `rte_inc_1scalar_by_nstream_bybnd`      |   🔲   |
| `rte_inc_2stream_by_1scalar_bybnd`      |   🔲   |
| `rte_inc_2stream_by_2stream_bybnd`      |   🔲   |
| `rte_inc_2stream_by_nstream_bybnd`      |   🔲   |
| `rte_inc_nstream_by_1scalar_bybnd`      |   🔲   |
| `rte_inc_nstream_by_2stream_bybnd`      |   🔲   |
| `rte_inc_nstream_by_nstream_bybnd`      |   🔲   |
| **OPTICAL PROPS - DELTA SCALING**       |         |
| `rte_delta_scale_2str_k`                |   🔲   |
| `rte_delta_scale_2str_f_k`              |   🔲   |
| **OPTICAL PROPS - SUBSET**              |         |
| `rte_extract_subset_dim1_3d`            |   🔲   |
| `rte_extract_subset_dim2_4d`            |   🔲   |
| `rte_extract_subset_absorption_tau`     |   🔲   |
| **Fluxes - Reduction**                  |         |
| `rte_sum_broadband`                     |   🔲   |
| `rte_net_broadband_full`                |   🔲   |
| `rte_net_broadband_precalc`             |   🔲   |
| `rte_sum_byband`                        |   🔲   |
| `rte_net_byband_full`                   |   🔲   |
| **Array Utilities**                     |         |
| `zero_array_1D`                         |   ✅   |
| `zero_array_2D`                         |   ✅   |
| `zero_array_3D`                         |   ✅   |
| `zero_array_4D`                         |   ✅   |

### RRTMGP Functions

RRTMGP functions are not yet available in the `pyrte` package.
Covering those functions is a future goal of this project.

## Setup Instructions

The current code in this repo are early experiments with the goal of exploring details fo the Fortran works and testing different options for creating an interface with C.
Later on, we will use this knowledge to bind it with Python

### Usage

* Make sure you have all the sources:

``` bash
git submodule update --init --recursive
```

* Install dependencies:

``` bash
sudo apt install -y \
    libnetcdff-dev \
    gfortran-10 \
    python3-dev \
    cmake
```

* The building of the sources and installation of the packages are handle by `CMakeLists.txt` and automatically called by pip.
* It builds both `rte` and `rrtmgp` and install the python package.


``` bash
pip install -e .
```

Once built, the module will be located in a folder called `pyrte`

## Pytest Setup Instructions

* Run `pip3 install pytest`

### Pytest Usage
* Run `cd test/`
* Create/update env (if you haven't already) by running `pip3 install -r requirements-test.txt`
* Run `pytest`
