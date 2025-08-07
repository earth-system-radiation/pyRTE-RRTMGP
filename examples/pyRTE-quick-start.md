---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: pyRTE
    language: python
    name: pyrte
---

# Quick start: Using pyRTE


## Overview

PyRTE-RRTMGP provides a flexible and efficient framework for computing radiative fluxes in planetary atmospheres. This example shows an end-to-end problem with both clear skies and clouds.

To use RTE and RRTMGP you'll need to:

1. Load data for cloud and gas optics 

Each calculation requires 

2. Computing gas and cloud optical properties and combining them to produce an all-sky problem
3. Solving the radiative transfer equation to obtain upward and downward fluxes

The package leverages `xarray` to represent data. Input data sets to the cloud and gas optics functions need to have specific datasets and specific dimensions. 

This example demonstrates the workflow for both longwave and shortwave radiative transfer calculations.

See the [documentation](https://pyrte-rrtmgp.readthedocs.io/en/latest/) for more information.


# Initialization




```python
# Plotting - off by default 
do_plots = False
```

## Import dependencies

```python
%matplotlib inline

from dask.diagnostics import ProgressBar
import xarray as xr

if do_plots: import matplotlib.pyplot as plt
```

## Import pyRTE entitites 

(The organization is a work in progress) 

```python
from pyrte_rrtmgp.rrtmgp_data_files import (
    CloudOpticsFiles,
    GasOpticsFiles,
)
from pyrte_rrtmgp.examples import (
    compute_RCE_clouds,
    compute_RCE_profiles,
    ALLSKY_EXAMPLES,
    load_example_file,
)
from pyrte_rrtmgp import rte
from pyrte_rrtmgp.rrtmgp import GasOptics, CloudOptics
```

## Initialize gas and cloud optics 

```python
cloud_optics_lw = CloudOptics(
    cloud_optics_file=CloudOpticsFiles.LW_BND
)
gas_optics_lw = GasOptics(
    gas_optics_file=GasOpticsFiles.LW_G256
)

cloud_optics_sw = CloudOptics(
    cloud_optics_file=CloudOpticsFiles.SW_BND
)
gas_optics_sw = GasOptics(
    gas_optics_file=GasOpticsFiles.SW_G224
)
```

The optics classes are `xarray Datasets` but the underlying data isn't meant to be accessed directly.

```python
cloud_optics_lw, gas_optics_lw
```

# Create an idealized problem 

## Temperature, humidity, composition

The routine `compute_RCE_profiles()` packaged with `pyRTE_RRTMGP` computes temperature, pressure, and humidity profiles following a moist adibat. The concentrations of other gases are also needed.

```python
def make_profiles(ncol=24, nlay=72):
    # Create atmospheric profiles and gas concentrations
    atmosphere = compute_RCE_profiles(300, ncol, nlay)

    # Add other gas values
    gas_values = {
        "co2": 348e-6,
        "ch4": 1650e-9,
        "n2o": 306e-9,
        "n2": 0.7808,
        "o2": 0.2095,
        "co": 0.0,
    }

    for gas_name, value in gas_values.items():
        atmosphere[gas_name] = value

    return atmosphere


atmosphere = make_profiles()
```

The dataset produced by `make_profiles` variable contains the minimum amount of information needed to compute clear-sky optical properties: 
- vertical dimensions `layer` and `level` with one more `level` than `layer`
- values of pressure and temperature on both vertical coordinates
- a surface temperature (for longwave problems)
- concentrations of seven gases defined on layers

```python
atmosphere
```

## Clouds 

`compute_RCE_clouds()` adds clouds (liquid and ice water path, liquid radius and ice diameter) to 2/3 of the columns 

```python
#
# Temporary workaround - compute_RCE_clouds() needs to know the particle size;
#   that's set as the mid-point of the valid range from cloud_optics
#
cloud_props = compute_RCE_clouds(
    cloud_optics_lw, atmosphere["pres_layer"], atmosphere["temp_layer"]
)

atmosphere = atmosphere.merge(cloud_props)
atmosphere
```

# Longwave fluxes

Taking one step at a time... First we will compute the optical properties of the gases (clear skies). 

## Clear-sky fluxes

```python
optical_props = gas_optics_lw.compute(
    atmosphere, 
    add_to_input=False,
)
```

The optical property for absorption/emission problems is optical depth `tau` which has a spectral dimension (`gpt`). The gas optics calculation also provides the radiation source functions, which for a longwave problem is the Planck function at the layer, level, and surface temperatures. 

```python
optical_props
```

Apply the boundary conditions:

```python
optical_props["surface_emissivity"] = 0.98
```

Dataset `optical_props` now contains a complete description of a longwave radiative transfer problem: 
- optical properties (`tau` for problem without scattering)
- radiation sources (`layer_source`, `level_source`, and `surface_source` for problems with internal emission)
- boundary conditions (`surface_emissivity` for problems with internal emission)
These variables are `spectrally-resolved` i.e. they have a `gpt` dimension

So we can compute fluxes (spectrally-integrated by default) 

```python
clr_fluxes = optical_props.rte.solve(add_to_input=False)
clr_fluxes
```

What do the fluxes look like? They're the same in every column so plot just the first.

```python
if do_plots:
    plt.plot(clr_fluxes.lw_flux_up.isel(column=0),   clr_fluxes.level, label="Flux up")
    plt.plot(clr_fluxes.lw_flux_down.isel(column=0), clr_fluxes.level, label="Flux down")
    plt.legend(frameon=False)
```

## Cloudy-sky fluxes

Next compute the optical properties of the clouds. Because we're treating absorption the only output is `tau`. The source functions aren't returned because the temperature isn't known. 

```python
clouds_optical_props = cloud_optics_lw.compute(
    atmosphere, 
    problem_type=rte.OpticsTypes.ABSORPTION,
)
# The optical properties of the clouds alone
clouds_optical_props
```

Next compute the combined optical properties of the clouds and gases with `add_to()`, which updates the optical properties in its argument.

```python
# add_to() changes the value of `optical_props`
clouds_optical_props.rte.add_to(optical_props)
optical_props
```

```python
fluxes = optical_props.rte.solve(add_to_input=False)
```

```python
if do_plots:
    plt.plot(fluxes.lw_flux_up.isel  (column=0), fluxes.level, label="LW all-sky flux up")
    plt.plot(fluxes.lw_flux_down.isel(column=0), fluxes.level, label="LW all-sky down")
    plt.plot(fluxes.lw_flux_up.isel  (column=2), fluxes.level, label="LW clear flux up")
    plt.plot(fluxes.lw_flux_down.isel(column=2), fluxes.level, label="LW clear flux down")
    plt.legend(frameon=False)
```

# Shortwave fluxes

Shortwave problems require different sets of optical properties, source, and boundary conditions. The first two 
are provided by the shortwave gas optics, as you'll see. 
Python allows intermediate steps to be combined. For example we can compute the all-sky optical properties directly by combining
the `gas_optics()`, `cloud_optics()`, and `add_to()` steps:

```python
# add_to() also returns updated optical properties 
optical_props = cloud_optics_sw.compute(atmosphere).\
    rte.add_to(
        gas_optics_sw.compute(
            atmosphere, 
            add_to_input=False,
        ), 
    delta_scale=True,
)
```

(Delta scaling increases accuracy for clouds, which have strong forward scattering (large `g`)).


Dataset `optical_props` now contains a nearly-complete description of a shortwave radiative transfer problem: 
- optical properties (`tau`, `ssa`, and `g` define a two-stream problem)
- radiation source (`toa_source`, `level_source`, and `surface_source` for the incoming sunlight as a collimated beam)
These variables are `spectrally-resolved` i.e. they have a `gpt` dimension

```python
optical_props
```

To compute fluxes the boundary condition needs to specified by supplying `surface_albedo` or the respective albedos for direct and diffuse radiation (`surface_albedo_direct`, `surface_albedo_diffuse`) as well as the cosine of the incident solar zenith angle `mu0`. 


```python
optical_props["surface_albedo"] = 0.06
optical_props["mu0"] = 0.86
fluxes = optical_props.rte.solve(add_to_input=False)
```

The returned fluxes include the total downwelling (`sw_flux_down`) and the direct beam component of that flux (`sw_flux_dir`) 

```python
if do_plots:
    plt.plot(fluxes.sw_flux_up.isel  (column=0), fluxes.level, label="SW all-sky flux up")
    plt.plot(fluxes.sw_flux_down.isel(column=0), fluxes.level, label="SW all-sky down")
    plt.plot(fluxes.sw_flux_up.isel  (column=2), fluxes.level, label="SW clear flux up")
    plt.plot(fluxes.sw_flux_down.isel(column=2), fluxes.level, label="SW clear flux down")
    plt.legend(frameon=False)
```

# Variants 

## Combining steps
The computation of optical properties and fluxes can be combined:

```python
fluxes = xr.merge(
            [cloud_optics_sw.compute(atmosphere).
            rte.add_to(
                gas_optics_sw.compute(
                    atmosphere, 
                    add_to_input=False,
                ), 
                delta_scale=True,
            ), 
            xr.Dataset(data_vars = {"surface_albedo":0.06, "mu0":0.86})],
        ).rte.solve(add_to_input = False)
```


## Parallelization with dask
Calculations can be divided and performed in parallel using `dask` using dask arrays. The only restriction is that 
vertical dimensions can't be chuncked over  

```python
atmosphere = make_profiles(ncol=24*16)
cloud_props = compute_RCE_clouds(
    cloud_optics_lw, atmosphere["pres_layer"], atmosphere["temp_layer"]
)

atmosphere = atmosphere.merge(cloud_props).chunk({"column":16})
atmosphere
```

```python
with ProgressBar():
    fluxes = xr.merge(
            [cloud_optics_sw.compute(atmosphere).rte.add_to(
                gas_optics_sw.compute(
                    atmosphere, 
                    add_to_input=False,
                ), 
                delta_scale=True,
            ), 
            xr.Dataset(data_vars = {"surface_albedo":0.06, "mu0":0.86})],
        ).rte.solve( 
             add_to_input = False,
    )
```

```python

```
