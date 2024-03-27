# Basic Usage

This section provides a brief overview of how to use `pyrte_rrtmgp` with Python.

## Importing the Package

To use any of the RTE-RRTMGP functions in Python, you must first import the `pyrte_rrtmgp` package:

```python
import pyrte_rrtmgp.pyrte_rrtmgp
```

This gives you access to [RTE-RRTMGP](https://github.com/earth-system-radiation/pyRTE-RRTMGP)'s Fortran functions directly in Python.

For example:

```python
import pyrte_rrtmgp.pyrte_rrtmgp as py

args = list_of_arguments

py.rte_lw_solver_noscat(*args)
```

## RTE Functions

The following RTE functions from [RTE-RRTMGP](https://github.com/earth-system-radiation/pyRTE-RRTMGP) are currently available in the `pyrte_rrtmgp` package:

```{include} ../../../README.md
:start-after: <!-- start-rte-functions-section -->
:end-before: <!-- end-rte-functions-section -->
```

## RRTMGP Functions

The following RRTMGP functions from [RTE-RRTMGP](https://github.com/earth-system-radiation/pyRTE-RRTMGP) are currently available in the `pyrte_rrtmgp` package:

```{include} ../../../README.md
:start-after: <!-- start-rrtmgp-functions-section -->
:end-before: <!-- end-rrtmgp-functions-section -->
```

```{seealso}
See the [RTE-RRTMGP repository on GitHub](https://github.com/earth-system-radiation/pyRTE-RRTMGP) and the [RTE-RRTMGP documentation](https://earth-system-radiation.github.io/rte-rrtmgp/) for more information about these functions.
```
