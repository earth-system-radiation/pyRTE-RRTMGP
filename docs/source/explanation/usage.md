# Basic Usage

This section provides a brief overview of how to use the pyRTE-RRTMGP package.

## RRTMGP and RTE

pyRTE-RRTMGP is a Python package that provides a Python interface to the [RTE-RRTMGP software](https://github.com/earth-system-radiation/rte-rrtmgp) (Fortran).

You can use this Python package to define a radiative transfer problem, compute gas and cloud optics, and solve the radiative transfer equations, using [RTE-RRTMGP](https://github.com/earth-system-radiation/rte-rrtmgp).


Specifically, this Python package contains tools for:

* RRTMGP (RRTM for General circulation model applications—Parallel)
* RTE (Radiative Transfer for Energetics)

See the paper [Balancing Accuracy, Efficiency, and Flexibility in Radiation Calculations for Dynamical Models](https://doi.org/10.1029/2019MS001621) for more details about the Fortran code.

Additionally, pyRTE-RRTMGP is able to use [Dask](https://docs.dask.org/en/stable/) for parallel computing, which can be useful for solving problems with large datasets on multi-core machines or clusters.

## Defining and Solving Radiative Transfer Problems

A typical workflow with pyRTE-RRTMGP consists of the following steps:

### 1. Loading gas optics data using the {func}`~pyrte_rrtmgp.rrtmgp_gas_optics.load_gas_optics` function

The {func}`~pyrte_rrtmgp.rrtmgp_gas_optics.load_gas_optics` function retrieves essential gas optics data for radiative transfer calculations. This data includes absorption coefficients and other optical properties for various atmospheric gases at different temperatures, pressures, and concentrations. See {func}`pyrte_rrtmgp.rrtmgp_gas_optics.load_gas_optics` for more details.

The package includes four default gas optics files, which can be accessed via the {data}`~pyrte_rrtmgp.data_types.GasOpticsFiles` enum:

*   **Longwave:**
    *   {data}`~pyrte_rrtmgp.data_types.GasOpticsFiles.LW_G128`: Longwave gas optics file with 128 g-points.
    *   {data}`~pyrte_rrtmgp.data_types.GasOpticsFiles.LW_G256`: Longwave gas optics file with 256 g-points.
*   **Shortwave:**
    *   {data}`~pyrte_rrtmgp.data_types.GasOpticsFiles.SW_G112`: Shortwave gas optics file with 112 g-points.
    *   {data}`~pyrte_rrtmgp.data_types.GasOpticsFiles.SW_G224`: Shortwave gas optics file with 224 g-points.

These files can be loaded using the {func}`~pyrte_rrtmgp.rrtmgp_gas_optics.load_gas_optics` function in conjunction with the {data}`~pyrte_rrtmgp.data_types.GasOpticsFiles` enum.

For example:

```python
gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
    gas_optics_file=GasOpticsFiles.LW_G128
)
```

### 2. Loading cloud optics data using the {func}`~pyrte_rrtmgp.rrtmgp_cloud_optics.load_cloud_optics` function

The {func}`~pyrte_rrtmgp.rrtmgp_cloud_optics.load_cloud_optics` function retrieves essential cloud optics data for radiative transfer calculations. See {func}`pyrte_rrtmgp.rrtmgp_cloud_optics.load_cloud_optics` for more details.

The package includes two default cloud optics files, which can be accessed via the {data}`~pyrte_rrtmgp.data_types.CloudOpticsFiles` enum:

*   **Longwave:**
    *   {data}`~pyrte_rrtmgp.data_types.CloudOpticsFiles.LW_BND`: Longwave cloud optics file with band points.
    *   {data}`~pyrte_rrtmgp.data_types.CloudOpticsFiles.LW_G128`: Longwave cloud optics file with 128 g-points.
    *   {data}`~pyrte_rrtmgp.data_types.CloudOpticsFiles.LW_G256`: Longwave cloud optics file with 256 g-points.
*   **Shortwave:**
    *   {data}`~pyrte_rrtmgp.data_types.CloudOpticsFiles.SW_BND`: Shortwave cloud optics file with band points.
    *   {data}`~pyrte_rrtmgp.data_types.CloudOpticsFiles.SW_G112`: Shortwave cloud optics file with 112 g-points.
    *   {data}`~pyrte_rrtmgp.data_types.CloudOpticsFiles.SW_G224`: Shortwave cloud optics file with 224 g-points.

These files can be loaded using the {func}`~pyrte_rrtmgp.rrtmgp_cloud_optics.load_cloud_optics` function in conjunction with the {data}`~pyrte_rrtmgp.data_types.CloudOpticsFiles` enum.

For example:

```python
cloud_optics_lw = rrtmgp_cloud_optics.load_cloud_optics(
    cloud_optics_file=CloudOpticsFiles.LW_G128
)
```

### 3. Define the gas optics atmosphere

The atmosphere file defines the gas concentrations for each layer of the atmosphere in mole fractions. The supported gases are:

* `h2o`: Water vapor
* `co2`: Carbon dioxide
* `o3`: Ozone
* `n2o`: Nitrous oxide
* `co`: Carbon monoxide
* `ch4`: Methane
* `o2`: Oxygen
* `n2`: Nitrogen
* `ccl4`: Carbon tetrachloride
* `cfc11`: Chlorofluorocarbon-11
* `cfc12`: Chlorofluorocarbon-12
* `cfc22`: Chlorofluorocarbon-22
* `hfc143a`: Hydrofluorocarbon-143a
* `hfc125`: Hydrofluorocarbon-125
* `hfc23`: Hydrofluorocarbon-23
* `hfc32`: Hydrofluorocarbon-32
* `hfc134a`: Hydrofluorocarbon-134a
* `cf4`: Carbon tetrafluoride
* `no2`: Nitrogen dioxide

You can use any of the alternative names in your dataset, and use a mapping dictionary to map them to the correct gas. See {data}`~pyrte_rrtmgp.config.DEFAULT_GAS_MAPPING` for the default mapping.

If any of the gases are not present in the dataset, it will be set to zero. `h2o` is required, and if it is not present, the package will raise an error.

The atmospheric input dataset also must contain the following core variables for computing gast optics:

*   `temp_layer`: Mean temperature within each atmospheric layer (K)
*   `temp_level`: Temperature at the boundaries between layers (K)
*   `pres_layer`: Mean pressure within each atmospheric layer (Pa)
*   `pres_level`: Pressure at the layer boundaries (Pa)

For longwave radiation calculations, an additional variable is required:

*   `surface_temperature`: Skin temperature of the surface (K), used to compute surface emissivity

These variables must include `layer` and `level` dimensions, where the `level` dimension is exactly one element larger than the `layer` dimension. Any additional dimensions in the dataset (e.g., spatial or temporal dimensions) will be automatically processed, with calculations performed for each combination of these dimensions.

For instance, consider a dataset with these dimensions:

*   `lat`: Latitude coordinates
*   `lon`: Longitude coordinates
*   `layer`: Atmospheric layers
*   `level`: Layer boundaries
*   `time`: Timesteps

In this case, the radiative transfer calculations will be executed independently for each unique combination of latitude, longitude, and time, across all atmospheric levels.

To get started quickly, you can load a preconfigured sample atmosphere dataset using the {func}`~pyrte_rrtmgp.examples.load_example_file` function.

### 4. Computing gas optics using the {class}`compute_gas_optics<pyrte_rrtmgp.rrtmgp_gas_optics.GasOpticsAccessor>` accessor

This function can handle two different types of problems:
* {data}`~pyrte_rrtmgp.data_types.OpticsProblemTypes.ABSORPTION` (longwave)
* {data}`~pyrte_rrtmgp.data_types.OpticsProblemTypes.TWO_STREAM` (shortwave)

Use a mapping dictionary to map gas names to their corresponding indices in the atmosphere data. For example:

```python
gas_mapping = {
    "h2o": "water_vapor",
    "co2": "carbon_dioxide_GM",
    "o3": "ozone",
    "n2o": "nitrous_oxide_GM",
    "co": "carbon_monoxide_GM",
    "ch4": "methane_GM",
    "o2": "oxygen_GM",
    "n2": "nitrogen_GM",
}
```

With the atmosphere data and the gas mapping dictionary, you can compute the gas optics using the {class}`compute_gas_optics<pyrte_rrtmgp.rrtmgp_gas_optics.GasOpticsAccessor>` accessor.

```python
gas_optics = rrtmgp_gas_optics.compute_gas_optics(
    atmosphere,
    problem_type=OpticsProblemTypes.ABSORPTION,
    gas_name_map=gas_mapping,
)
```

See {class}`pyrte_rrtmgp.rrtmgp_gas_optics.GasOpticsAccessor` for more details.

### 5. Define the cloud optics atmosphere

The cloud optics atmosphere requires the following variables in the input dataset for computing cloud optical properties:

*   `lwp`: Liquid water path (g/m²)
*   `iwp`: Ice water path (g/m²)
*   `rel`: Effective radius of liquid cloud particles (microns)
*   `rei`: Effective radius of ice cloud particles (microns)

They are defined in the `layer` dimension. Similar to the gas optics atmosphere, any additional dimensions will be processed independently.

### 6. Computing cloud optics using the {class}`compute_cloud_optics<pyrte_rrtmgp.rrtmgp_cloud_optics.CloudOpticsAccessor>` accessor

You can compute the cloud optics using the {class}`compute_cloud_optics<pyrte_rrtmgp.rrtmgp_cloud_optics.CloudOpticsAccessor>` accessor.

This function can handle two different types of problems:
* {data}`~pyrte_rrtmgp.data_types.OpticsProblemTypes.ABSORPTION` (longwave)
* {data}`~pyrte_rrtmgp.data_types.OpticsProblemTypes.TWO_STREAM` (shortwave)

```python
cloud_optics = rrtmgp_cloud_optics.compute_cloud_optics(
    atmosphere,
    problem_type=problem_type=OpticsProblemTypes.ABSORPTION,
)
```

See {class}`pyrte_rrtmgp.rrtmgp_cloud_optics.CloudOpticsAccessor` for more details.

### 7. Combining Cloud Optics with Gas Optics using the {class}`add_to<pyrte_rrtmgp.rrtmgp_cloud_optics.CombineOpticalPropsAccessor>` Accessor

To integrate cloud optical properties with gas optical properties, you can use the {class}`add_to<pyrte_rrtmgp.rrtmgp_cloud_optics.CombineOpticalPropsAccessor>` accessor. This allows you to either add the cloud optical properties independently or combine them with the gas optical properties.

For example, to combine the optical properties:

```python
clouds_optical_props.add_to(clear_sky_optical_props, delta_scale=True)
```

The clouds will be included in place over the clear sky optical properties.

### 8. Solving the radiative transfer equations using the {func}`~pyrte_rrtmgp.rte_solver.rte_solve` function

This function uses the optical properties of the atmosphere to solve the radiative transfer equations and compute the radiative fluxes. See {func}`pyrte_rrtmgp.rte_solver.rte_solve` for more details.

```{seealso}
See {ref}`tutorials` for examples of how to use the package.
See {ref}`python_module_ref` for a reference of all available submodules.
```
