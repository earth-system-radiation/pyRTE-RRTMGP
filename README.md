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

Currently, the following functions are available in the `pyRTE_RRTMGP` package:

### RTE Functions (WIP)

| Function name                           | Covered |
|-----------------------------------------|:-------:|
| **SHORTWAVE SOLVERS**                   |         |
| `rte_sw_solver_noscat`                  |   🔲   |
| `rte_sw_solver_2stream`                 |   🔲   |
| **LONGWAVE SOLVERS**                    |         |
| `rte_lw_solver_noscat`                  |   🔲   |
| `rte_lw_solver_2stream`                 |   🔲   |
| **OPTICAL PROPS - INCREMENT**           |         |
| `rte_increment_1scalar_by_1scalar`      |   🔲   |
| `rte_increment_1scalar_by_2stream`      |   🔲   |
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
| `zero_array_1D`                         |   🔲   |
| `zero_array_2D`                         |   🔲   |
| `zero_array_3D`                         |   🔲   |
| `zero_array_4D`                         |   🔲   |

### RRTMGP Functions

RRTMGP functions are not yet available in the `pyRTE_RRTMGP` package.
Covering those functions is a future goal of this project.

# Instructions

The goal of the project is to create Python bindings for various Fortran libraries for solving radiance transmit equations.

This repo explores how we can make that happen, starting with understanding how Fortran works and how we can interface it with C.  
Later on we will use this knowledge to bind it with python

## Structure:

* `test.F90` - A Fortran program. This file contains a fortran main function and can be compiles as a standalone program. It also defines 2 more functions - `add` and `hello_world`. The `add` function is an example function that takes in 3 double parameter, first 2 are input parameters for the summations, and the third is an output parameter holding in the result of the summation. The `hello_world` method is even simpler, it takes no parameters, nor it returns ant, just showcases how function invocation works. The `add` and `hello_world` subroutines use BIND(C) to use the C language function call convention, so that they can be called from C functions (or python later on)
* `fortran_interface.h` - A simple C Header file that provides C forward declarations for the fortran methods defined in `test.F90`.
* `main.cpp` - A simple C program that invokes the fortran functions
* `fcompile.sh` - A script to compile the code. It will use test.F90 to build a shared library (`libtest.so`). All build artifacts would be located in a `./build` folder

## 
sudo apt install python-dev
brew install python