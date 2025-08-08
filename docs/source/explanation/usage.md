# Basic Usage

This section provides a brief overview of how to use the pyRTE-RRTMGP package. The [example scripts and notebooks](https://github.com/earth-system-radiation/pyRTE-RRTMGP/tree/main/examples) packaged with pyRTE-RRTMGP may also be useful (and are guaranteed to work with the current code).

The examples will likely supersede this documention over time.

## RRTMGP and RTE

pyRTE-RRTMGP is a Python package that provides a Python interface to the [RTE-RRTMGP software](https://github.com/earth-system-radiation/rte-rrtmgp) (Fortran). You can use this Python package to define a radiative transfer problem, compute gas and cloud optics, and solve the radiative transfer equations, using the same underlying code. See the paper [Balancing Accuracy, Efficiency, and Flexibility in Radiation Calculations for Dynamical Models](https://doi.org/10.1029/2019MS001621) for more details about the Fortran code.

pyRTE-RRTMGP uses [Dask](https://docs.dask.org/en/stable/) for parallel computing, which can be useful for solving problems with large datasets on multi-core machines or clusters.

## Defining and Solving Radiative Transfer Problems

A typical workflow with pyRTE-RRTMGP consists of the following steps:

### 1. Loading gas optics data by initializing the {class}`~pyrte_rrtmgp.rrtmgp.GasOptics` class

Initializing the {class}`~pyrte_rrtmgp.rrtmgp.GasOptics` class retrieves essential gas optics data for radiative transfer calculations. This data includes absorption coefficients and other optical properties for various atmospheric gases at different temperatures, pressures, and concentrations. See {class}`~pyrte_rrtmgp.rrtmgp.GasOptics` for more details.

The package includes four default gas optics files, which can be accessed via the {data}`~pyrte_rrtmgp.rrtmgp_data_files.GasOpticsFiles` enum:

*   **Longwave:**
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.GasOpticsFiles.LW_G128`: Longwave gas optics file with 128 g-points.
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.GasOpticsFiles.LW_G256`: Longwave gas optics file with 256 g-points.
*   **Shortwave:**
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.GasOpticsFiles.SW_G112`: Shortwave gas optics file with 112 g-points.
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.GasOpticsFiles.SW_G224`: Shortwave gas optics file with 224 g-points.

The data is loaded by passing one of the values of the enum to the `GasOptics` constructor. For example:

```python
gas_optics_lw = rrtmgp.GasOptics(
    gas_optics_file=GasOpticsFiles.LW_G256
)
```

File names can also be supplied directly.

### 2. Loading cloud optics data by initializing the {class}`~pyrte_rrtmgp.rrtmgp.CloudOptics` class

{class}`~pyrte_rrtmgp.rrtmgp.CloudOptics` is the cloud equivalent of {class}`~pyrte_rrtmgp.rrtmgp.GasOptics`.
pyRTE-RRTMGP includes cloud optics files which can be accessed via the {data}`~pyrte_rrtmgp.rrtmgp_data_files.CloudOpticsFiles` enum:

*   **Longwave:**
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.CloudOpticsFiles.LW_BND`: Longwave cloud optics file with band points.
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.CloudOpticsFiles.LW_G128`: Longwave cloud optics file with 128 g-points.
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.CloudOpticsFiles.LW_G256`: Longwave cloud optics file with 256 g-points.
*   **Shortwave:**
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.CloudOpticsFiles.SW_BND`: Shortwave cloud optics file with band points.
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.CloudOpticsFiles.SW_G112`: Shortwave cloud optics file with 112 g-points.
    *   {data}`~pyrte_rrtmgp.rrtmgp_data_files.CloudOpticsFiles.SW_G224`: Shortwave cloud optics file with 224 g-points.

Loading the data is the same as for the gas optics:
```python
cloud_optics_lw = rrtmgp.CloudOptics(
    cloud_optics_file=CloudOpticsFiles.LW_G256
)
```

### 3. Define the gas optics atmosphere

The atmosphere file defines the gas concentrations for each layer of the atmosphere in mole fractions. Gases
are specified in terms of their (lowercase) chemical formula, with `cfcX` used for chlorofluorocarbon-X and
`hfcX` for hydrofluorocarbon-X. Supported gases, i.e. those that will influence the calculation of optical properties,
are available using the `available_gases` property of the gas optics, i.e.
```python
gas_optics = rrtmgp.GasOptics(
    gas_optics_file=GasOpticsFiles.LW_G256
)
gas_optics.available_gases
```

Concentrations of some gases are required, those these can vary between the shortwave and longwave gas optics.
```python
gas_optics = rrtmgp.GasOptics(
    gas_optics_file=GasOpticsFiles.LW_G256
)

gas_optics.required_gases
```

You can use any of the alternative names in your dataset, and use a mapping dictionary to map them to the correct gas. See {data}`~pyrte_rrtmgp.config.DEFAULT_GAS_MAPPING` for the default mapping.

If any of the gases are not present in the dataset, it will be set to zero. RRTMGP requires some gases to be specified; if are not present the package will raise an error.

The atmospheric input dataset also must contain the following core variables for computing gast optics:

*   `temp_layer`: Mean temperature within each atmospheric layer (K)
*   `temp_level`: Temperature at the boundaries between layers (K)
*   `pres_layer`: Mean pressure within each atmospheric layer (Pa)
*   `pres_level`: Pressure at the layer boundaries (Pa)

These variables must include `layer` and `level` dimensions, where the `level` dimension is one element larger than the `layer` dimension.
Any additional dimensions in the dataset (e.g., spatial or temporal dimensions) will be automatically processed, with calculations performed for each combination of these dimensions.

For longwave radiation calculations, an additional variable is required:

*   `surface_temperature`: Skin temperature of the surface (K), used to compute the surface source of radiation

For shortwave radiation calculation the atmosphere may also contain an optional variable

* `total_solar_irradiance`: Total solar irradiance at the top of the atmosphere (W/m2)

Neither `surface_temperature` nor `total_solar_irradiance` have a vertical dependence i.e. they should have no `layer` or `level` coordinate.

For instance, consider a dataset with these dimensions:

*   `lat`: Latitude coordinates
*   `lon`: Longitude coordinates
*   `layer`: Atmospheric layers
*   `level`: Layer boundaries
*   `time`: Timesteps

In this case, the radiative transfer calculations will be executed independently for each unique combination of latitude, longitude, and time, across all atmospheric levels.


### 4. Computing gas optics using `GasOptics.compute()`

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
gas_optics = gas_optics_lw.compute(
    atmosphere,
    gas_name_map=gas_mapping,
)
```

The gas optical properties also include the radiation source information.

### 5. Define the cloud optics

The cloud optics atmosphere requires the following variables in the input dataset for computing cloud optical properties:

*   `lwp`: Liquid water path (g/m²)
*   `iwp`: Ice water path (g/m²)
*   `rel`: Effective radius of liquid cloud particles (microns)
*   `rei`: Effective radius of ice cloud particles (microns)

They are defined in the `layer` dimension. Similar to the gas optics atmosphere, any additional dimensions will be processed independently.

### 6. Computing cloud optics using `CloudOptics.compute()`

You can compute the cloud optics using function `compute()` from {class}`pyrte_rrtmgp.rrtmgp.CloudOptics`.

This function can handle two different types of problems:
* {data}`~pyrte_rrtmgp.rte.OpticsTypes.ABSORPTION` (longwave)
* {data}`~pyrte_rrtmgp.rte.OpticsTypes.TWO_STREAM` (shortwave)

```python
cloud_optics = cloud_optics_lw.compute(
    atmosphere,
    problem_type=problem_type=OpticsTypes.ABSORPTION,
)
```

See {class}`pyrte_rrtmgp.rrtmgp_cloud_optics.CloudOpticsAccessor` for more details.

### 7. Combining Cloud Optics with Gas Optics using {class}`add_to()<pyrte_rrtmgp.rte>`.

To integrate cloud optical properties with gas optical properties use the {class}`add_to<pyrte_rrtmgp.rte>` accessor. This adds the cloud optical properties to the pre-computed gas optical properties, including expanding from bands to g-points if needed.

For example, to combine the optical properties:

```python
clouds_optical_props.rte.add_to(
    clear_sky_optical_props,
    delta_scale=True)
```
The clouds will be included in place over the clear sky optical properties.

### 8. Solving the radiative transfer equations using {class}`solve()<~pyrte_rrtmgp.rte>`

This function uses the optical properties of the atmosphere to solve the radiative transfer equations and compute the radiative fluxes.
```python
fluxes = clouds_optical_props.rte.add_to(
    clear_sky_optical_props,
    delta_scale=True).\
    rte.solve(add_to_input = False)
```

See {class}`solve<~pyrte_rrtmgp.rte>` for more details.

```{seealso}
See {ref}`tutorials` for examples of how to use the package.
See {ref}`python_module_ref` for a reference of all available submodules.
```
