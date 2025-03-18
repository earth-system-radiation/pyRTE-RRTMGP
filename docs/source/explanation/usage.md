# Basic Usage

This section provides a brief overview of how to use `pyrte_rrtmgp` with Python.

## RRTMGP and RTE

pyRTE-RRTMGP is a Python package that provides a Python interface to the [RTE-RRTMGP software](https://github.com/earth-system-radiation/rte-rrtmgp) (Fortran).

You can use this Python package to define a radiative transfer problem, compute gas and cloud optics, and solve the radiative transfer equations, using [RTE-RRTMGP](https://github.com/earth-system-radiation/rte-rrtmgp).


Specifically, this Python package contains tools for:

* RRTMGP (RRTM for General circulation model applications—Parallel)
* RTE (Radiative Transfer for Energetics)

See the paper [Balancing Accuracy, Efficiency, and Flexibility in Radiation Calculations for Dynamical Models](https://doi.org/10.1029/2019MS001621) for more details about the Fortran code.

## Defining and Solving Radiative Transfer Problems

A typical workflow with pyRTE-RRTMGP consists of the following steps:

### 1. Loading gas optics data using the [load_gas_optics](pyrte_rrtmgp.rrtmgp_gas_optics.load_gas_optics) function

Gas data usually contains atmospheric profiles with various combinations of temperature, pressure, and gas concentrations. See [](pyrte_rrtmgp.rrtmgp_gas_optics.load_gas_optics) for more details.

### 2. Loading cloud optics data using the [load_cloud_optics](pyrte_rrtmgp.rrtmgp_cloud_optics.load_cloud_optics) function

[TBD] [what specifications do we expect the input data to conform to?] See [](pyrte_rrtmgp.rrtmgp_cloud_optics.load_cloud_optics) for more details.

### 3. Define an atmosphere with the [load_rrtmgp_file](pyrte_rrtmgp.utils.load_rrtmgp_file) function

[TBD] See [](pyrte_rrtmgp.utils.load_rrtmgp_file) for more details.

### 4. Computing gas optics using the [compute_gas_optics](pyrte_rrtmgp.rrtmgp_gas_optics.GasOpticsAccessor) accessor

This function can handle two different types of problems: `absorption` (longwave) and `two-stream` (shortwave). It usually also uses a gas mapping dictionary to map gas names to their corresponding indices in the atmosphere data. See [](pyrte_rrtmgp.rrtmgp_gas_optics.GasOpticsAccessor) for more details.

### 5. Computing cloud optics using the [compute_cloud_optics](pyrte_rrtmgp.rrtmgp_cloud_optics.CloudOpticsAccessor) accessor

This function can handle two different types of problems: `absorption` (longwave) and `two-stream` (shortwave).

### 6. Adding cloud optics to the gas optics using the [add_to](pyrte_rrtmgp.rrtmgp_cloud_optics.CombineOpticalPropsAccessor) accessor

This function combines the gas and cloud optics to create a single set of optical properties for the atmosphere. See [](pyrte_rrtmgp.rrtmgp_cloud_optics.CombineOpticalPropsAccessor) for more details.

### 7. Solving the radiative transfer equations using the [rte_solve](pyrte_rrtmgp.rte_solver.rte_solve) function

This function uses the optical properties of the atmosphere to solve the radiative transfer equations and compute the radiative fluxes. See [](pyrte_rrtmgp.rte_solver.rte_solve) for more details.


<!-- Look at long wave example

RTE Radio transfer problem (how light is transmitted through the atmosphere)

RRTMGP What are the charactersitcs of the athmospehre, then we compute clouds and gasses (aerosols not included here, but available in fortran) - describes the athmosphere

two types of problem:
shortwave: light of the sun
longwave: radiation from the planet

sometimes separately, sometimes both together

You can merge gasses and clouds
then you compute the RTE with the solver() method

First we import gas optics
We can then load the atmosphere
Then with both, we compute the RRTMGP (compute_gas_optics: takes a problem_type absoprtion or two-stream)

Adds tau to the atmosphere [how atomsphere reacts with light, reflectance]


For all_sky example: gasses and clouds

``load_cloud_optics``
``compute_cloud_optics``

then add clouds to clear sky with ``add_to()`` [clouds_optical_props is added to clear_sky_optical_props] changes the clear_sky_optical_props


``load_cloud_optics``
``compute_cloud_optics``

``load_gas_optics``
``compute_gas_optics``

``add_to()``

``rte_solve()``


The `pyrte_rrtmgp` package contains multiple submodules. The most important ones are:

- `pyrte_rrtmgp.rte_solver`: The main module that provides functions to solve radiative transfer equations (RTE) using RRTMGP. See {ref}`pyrte_rrtmgp_solver` for details.
- `pyrte_rrtmgp.rrtmgp_gas_optics`: A module that provides functions to calculate gas optics for RRTMGP. See {ref}`pyrte_rrtmgp_gas_optics` for details.
- `pyrte_rrtmgp.rrtmgp_cloud_optics`: A module that provides functions to calculate cloud optics for RRTMGP. See {ref}`pyrte_rrtmgp_cloud_optics` for details. -->


```{seealso}
See {ref}`tutorials` for examples of how to use the package.
See {ref}`python_module_ref` for a reference of all available submodules.
```
