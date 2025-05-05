# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: pyRTE
#     language: python
#     name: pyrte
# ---

# %% [markdown]
# # Using pyRTE

# %% [markdown]
# ## Overview
#
# PyRTE-RRTMGP provides a flexible and efficient framework for computing radiative fluxes in planetary atmospheres. This example shows an end-to-end problem with both clear skies and clouds.
#
# To use RTE and RRTMGP you'll need to:
#
# 1. Load data for cloud and gas optics 
#
# Each calculation requires 
#
# 2. Computing gas and cloud optical properties and combining them to produce an all-sky problem
# 3. Solving the radiative transfer equation to obtain upward and downward fluxes
#
# The package leverages `xarray` to represent data. Input data sets to the cloud and gas optics functions need to have specific datasets and specific dimensions. 
#
# This example demonstrates the workflow for both longwave and shortwave radiative transfer calculations.
#
# See the [documentation](https://pyrte-rrtmgp.readthedocs.io/en/latest/) for more information.

# %% [markdown]
# # Initialization

# %% [markdown]
# ## Import dependencies

# %%
# %matplotlib inline

import xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

# %% [markdown]
# ## Import pyRTE entitites 
#
# (The organization is a work in progress) 

# %%
from pyrte_rrtmgp import rrtmgp_cloud_optics, rrtmgp_gas_optics
from pyrte_rrtmgp.data_types import (
    AllSkyExampleFiles,
    CloudOpticsFiles,
    GasOpticsFiles,
    OpticsProblemTypes,
)
from pyrte_rrtmgp.rte_solver import rte_solve
from pyrte_rrtmgp.utils import (
    compute_clouds,
    compute_profiles,
    load_rrtmgp_file,
)

# %% [markdown]
# ## Initialize gas and cloud optics 

# %%
cloud_optics_lw = rrtmgp_cloud_optics.load_cloud_optics(
    cloud_optics_file=CloudOpticsFiles.LW_BND
)
gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
    gas_optics_file=GasOpticsFiles.LW_G256
)

cloud_optics_sw = rrtmgp_cloud_optics.load_cloud_optics(
    cloud_optics_file=CloudOpticsFiles.SW_BND
)
gas_optics_sw = rrtmgp_gas_optics.load_gas_optics(
    gas_optics_file=GasOpticsFiles.SW_G224
)

# %% [markdown]
# The optics classes are `xarray Datasets` but the underlying data isn't meant to be accessed directly.

# %%
cloud_optics_lw, gas_optics_lw


# %% [markdown]
# # Create an idealized problem 
#
# ## Temperature, humidity, composition
#
# The routine `compute_profiles()` packaged with `pyRTE_RRTMGP` computes temperature, pressure, and humidity profiles following a moist adibat. The concentrations of other gases are also needed.

# %%
def make_profiles(ncol=24, nlay=72):
    # Create atmospheric profiles and gas concentrations
    atmosphere = compute_profiles(300, ncol, nlay)

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

# %% [markdown]
# The dataset produced by `make_profiles` variable contains the minimum amount of information needed to compute clear-sky optical properties: 
# - vertical dimensions `layer` and `level` with one more `level` than `layer`
# - values of pressure and temperature on both vertical coordinates
# - a surface temperature (for longwave problems)
# - concentrations of seven gases defined on layers

# %%
atmosphere

# %% [markdown]
# ## Clouds 
#
# `compute_clouds()` adds clouds (liquid and ice water path, liquid radius and ice diameter) to 2/3 of the columns 

# %%
#
# Temporary workaround - compute_clouds() needs to know the particle size;
#   that's set as the mid-point of the valid range from cloud_optics
#
cloud_props = compute_clouds(
    cloud_optics_lw, atmosphere["pres_layer"], atmosphere["temp_layer"]
)

atmosphere = atmosphere.merge(cloud_props)
atmosphere

# %% [markdown]
# # Longwave fluxes
#
# Taking one step at a time... First we will compute the optical properties of the gases (clear skies). 
#
# ## Clear-sky fluxes

# %%
optical_props = gas_optics_lw.compute_gas_optics(
    atmosphere, 
    problem_type=OpticsProblemTypes.ABSORPTION, 
    add_to_input=False,
)

# %% [markdown]
# The optical property for absorption/emission problems is optical depth `tau` which has a spectral dimension (`gpt`). The gas optics calculation also provides the radiation source functions, which for a longwave problem is the Planck function at the layer, level, and surface temperatures. 

# %%
optical_props

# %% [markdown]
# Apply the boundary conditions:

# %%
optical_props["surface_emissivity"] = 0.98

# %% [markdown]
# Dataset `optical_props` now contains a complete description of a longwave radiative transfer problem: 
# - optical properties (`tau` for problem without scattering)
# - radiation sources (`layer_source`, `level_source`, and `surface_source` for problems with internal emission)
# - boundary conditions (`surface_emissivity` for problems with internal emission)
# These variables are `spectrally-resolved` i.e. they have a `gpt` dimension
#
# So we can compute fluxes (spectrally-integrated by default) 

# %%
clr_fluxes = rte_solve(optical_props, add_to_input=False)
clr_fluxes

# %% [markdown]
# What do the fluxes look like? They're the same in every column so plot just the first.

# %%
plt.plot(clr_fluxes.lw_flux_up.isel(site=0),   clr_fluxes.level, label="Flux up")
plt.plot(clr_fluxes.lw_flux_down.isel(site=0), clr_fluxes.level, label="Flux down")
plt.legend(frameon=False)

# %% [markdown]
# ## Cloudy-sky fluxes
#
# Next compute the optical properties of the clouds. Because we're treating absorption the only output is `tau`. The source functions aren't returned because the temperature isn't known. 

# %%
clouds_optical_props = cloud_optics_lw.compute_cloud_optics(
    atmosphere, problem_type=OpticsProblemTypes.ABSORPTION
)
# The optical properties of the clouds alone
clouds_optical_props

# %% [markdown]
# Next compute the combined optical properties of the clouds and gases with `add_to()`, which updates the optical properties in its argument.

# %%
# add_to() changes the value of `optical_props`
clouds_optical_props.add_to(optical_props)
optical_props

# %%
fluxes = rte_solve(optical_props, add_to_input=False)

# %%
plt.plot(fluxes.lw_flux_up.isel  (site=0), fluxes.level, label="LW all-sky flux up")
plt.plot(fluxes.lw_flux_down.isel(site=0), fluxes.level, label="LW all-sky down")
plt.plot(fluxes.lw_flux_up.isel  (site=2), fluxes.level, label="LW clear flux up")
plt.plot(fluxes.lw_flux_down.isel(site=2), fluxes.level, label="LW clear flux down")
plt.legend(frameon=False)

# %% [markdown]
# # Shortwave fluxes
#
# Shortwave problems require different sets of optical properties, source, and boundary conditions. The first two 
# are provided by the shortwave gas optics, as you'll see. 
# Python allows intermediate steps to be combined. For example we can compute the all-sky optical properties directly by combining
# the `gas_optics()`, `cloud_optics()`, and `add_to()` steps:

# %%
# add_to() also returns updated optical properties 
optical_props = cloud_optics_sw.compute_cloud_optics(atmosphere).add_to(
    gas_optics_sw.compute_gas_optics(
        atmosphere, 
        problem_type=OpticsProblemTypes.TWO_STREAM, 
        add_to_input=False,
    ), 
    delta_scale=True,
)

# %% [markdown]
# (Delta scaling increases accuracy for clouds, which have strong forward scattering (large `g`)).

# %% [markdown]
# Dataset `optical_props` now contains a nearly-complete description of a shortwave radiative transfer problem: 
# - optical properties (`tau`, `ssa`, and `g` define a two-stream problem)
# - radiation source (`toa_source`, `level_source`, and `surface_source` for the incoming sunlight as a collimated beam)
# These variables are `spectrally-resolved` i.e. they have a `gpt` dimension

# %%
optical_props

# %% [markdown]
# To compute fluxes the boundary condition needs to specified by supplying `surface_albedo` or the respective albedos for direct and diffuse radiation (`surface_albedo_direct`, `surface_albedo_diffuse`) as well as the cosine of the incident solar zenith angle `mu0`. 
#

# %%
optical_props["surface_albedo"] = 0.06
optical_props["mu0"] = 0.86
fluxes = rte_solve(optical_props, add_to_input=False)

# %% [markdown]
# The returned fluxes include the total downwelling (`sw_flux_down`) and the direct beam component of that flux (`sw_flux_dir`) 

# %%
plt.plot(fluxes.sw_flux_up.isel  (site=0), fluxes.level, label="SW all-sky flux up")
plt.plot(fluxes.sw_flux_down.isel(site=0), fluxes.level, label="SW all-sky down")
plt.plot(fluxes.sw_flux_up.isel  (site=2), fluxes.level, label="SW clear flux up")
plt.plot(fluxes.sw_flux_down.isel(site=2), fluxes.level, label="SW clear flux down")
plt.legend(frameon=False)

# %% [markdown]
# ## Variants 
#
# ### Combining steps
# The computation of optical properties and fluxes can be combined:

# %%
fluxes = rte_solve(
    xr.merge(
        [cloud_optics_sw.compute_cloud_optics(atmosphere).add_to(
            gas_optics_sw.compute_gas_optics(
                atmosphere, 
                problem_type=OpticsProblemTypes.TWO_STREAM, 
                add_to_input=False,
            ), 
            delta_scale=True,
        ), 
        xr.Dataset(data_vars = {"surface_albedo":0.06, "mu0":0.86})],
    ), 
    add_to_input = False,
)

# ### Parallelization with dask
# Calculations can be divided and performed in parallel using `dask` using dask arrays. The only restriction is that 
# vertical dimensions can't be chuncked over  

# %%
atmosphere = make_profiles(ncol=24*16)
cloud_props = compute_clouds(
    cloud_optics_lw, atmosphere["pres_layer"], atmosphere["temp_layer"]
)

atmosphere = atmosphere.merge(cloud_props).chunk({"site":16})
atmosphere

# %%
with ProgressBar():
    fluxes = rte_solve(
        xr.merge(
            [cloud_optics_sw.compute_cloud_optics(atmosphere).add_to(
                gas_optics_sw.compute_gas_optics(
                    atmosphere, 
                    problem_type=OpticsProblemTypes.TWO_STREAM, 
                    add_to_input=False,
                ), 
                delta_scale=True,
            ), 
            xr.Dataset(data_vars = {"surface_albedo":0.06, "mu0":0.86})],
        ), 
        add_to_input = False,
    )


# %%
