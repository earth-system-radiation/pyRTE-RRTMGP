# Basic Usage

This section provides a brief overview of how to use `pyrte_rrtmgp` with Python.

## Python Package Structure

The `pyrte_rrtmgp` package contains the following submodules:

- `pyrte_rrtmgp.pyrte_rrtmgp`: The main module that provides access to a subset of RTE-RRTMGP's Fortran functions in Python. The functions available in this module mirror the Fortran functions (see below). You can think of this as the low-level implementation that allows you to access the respective Fortran functions directly in Python.
- `pyrte_rrtmgp.rte`: A high-level module that provides a more user-friendly Python interface for select RTE functions. This module is still under development and will be expanded in future releases.
- `pyrte_rrtmgp.rrtmgp`: A high-level module that provides a more user-friendly Python interface for select RRTMGP functions. This module is still under development and will be expanded in future releases.
- `pyrte_rrtmgp.utils`: A module that provides utility functions for working with RTE-RRTMGP data. This module is still under development and will be expanded in future releases.

```{seealso}
The folder `examples` in the repository contains a Jupyter notebook example that demonstrates how to use the package.

The functions in `pyrte_rrtmgp.rte`, `pyrte_rrtmgp.rrtmgp`, and `pyrte_rrtmgp.utils` contain docstrings that are available in IDEs with features such as IntelliSense (in VSCode).
```

## Importing the Package

To use any of the RTE-RRTMGP functions in Python, you must first import `pyrte_rrtmgp`.

For example ``import pyrte_rrtmgp.pyrte_rrtmgp`` or ``import pyrte_rrtmgp.rrtmgp as rrtmgp``.

This gives you access to [RTE-RRTMGP](https://github.com/earth-system-radiation/pyRTE-RRTMGP)'s Fortran functions directly in Python.

For example:

```python
import pyrte_rrtmgp.pyrte_rrtmgp as py

args = list_of_arguments

py.rte_lw_solver_noscat(*args)
```

## Available RTE and RRTMGP Functions

The following RTE functions from [RTE-RRTMGP](https://github.com/earth-system-radiation/pyRTE-RRTMGP) are currently available in the `pyrte_rrtmgp.pyrte_rrtmgp` module:

```{include} ../../../README.md
:start-after: <!-- start-rte-functions-section -->
:end-before: <!-- end-rte-functions-section -->
```

The following RRTMGP functions from [RTE-RRTMGP](https://github.com/earth-system-radiation/pyRTE-RRTMGP) are currently available in the `pyrte_rrtmgp.pyrte_rrtmgp` module:

```{include} ../../../README.md
:start-after: <!-- start-rrtmgp-functions-section -->
:end-before: <!-- end-rrtmgp-functions-section -->
```

```{seealso}
See the [RTE-RRTMGP repository on GitHub](https://github.com/earth-system-radiation/pyRTE-RRTMGP) and the [RTE-RRTMGP documentation](https://earth-system-radiation.github.io/rte-rrtmgp/) for more information about these functions.
```
