# Basic Usage

This section provides a brief overview of how to use `pyrte_rrtmgp` with Python.

## Python Package Structure

The `pyrte_rrtmgp` package contains the following submodules:

- `pyrte_rrtmgp.pyrte_rrtmgp`: The main module that provides access to a subset of RTE-RRTMGP's Fortran functions in Python. The functions available in this module mirror the Fortran functions (see below). You can think of this as the low-level implementation that allows you to access the respective Fortran functions directly in Python. See [](low_level_interface) for more information.
- `pyrte_rrtmgp.rte`: A high-level module that provides a more user-friendly Python interface for select RTE functions. This module is still under development and will be expanded in future releases. See [](module_ref) for details.
- `pyrte_rrtmgp.rrtmgp`: A high-level module that provides a more user-friendly Python interface for select RRTMGP functions. This module is still under development and will be expanded in future releases. See [](module_ref) for details.
- `pyrte_rrtmgp.utils`: A module that provides utility functions for working with RTE-RRTMGP data. This module is still under development and will be expanded in future releases. See [](module_ref) for details.

```{seealso}
The folder `examples` in the repository contains a Jupyter notebook example that demonstrates how to use the package.

The functions in `pyrte_rrtmgp.rte`, `pyrte_rrtmgp.rrtmgp`, and `pyrte_rrtmgp.utils` contain docstrings that are available in IDEs with features such as IntelliSense (in VSCode).
```

## Importing the Package

To use any of the RTE-RRTMGP functions in Python, you first need to import `pyrte_rrtmgp` and the respective submodule you want to use.
For example: ``import pyrte_rrtmgp.pyrte_rrtmgp`` or ``import pyrte_rrtmgp.rrtmgp as rrtmgp``.

The example below uses the `pyrte_rrtmgp.pyrte_rrtmgp` submodule, which gives you low-level access to [RTE-RRTMGP](https://github.com/earth-system-radiation/pyRTE-RRTMGP)'s Fortran functions in Python:

```python
import pyrte_rrtmgp.pyrte_rrtmgp as py

args = list_of_arguments

py.rte_lw_solver_noscat(*args)
```
