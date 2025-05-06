# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # All-sky Radiative Transfer Example with PyRTE-RRTMGP
#
# This notebook demonstrates how to use the PyRTE-RRTMGP package to solve an idealized problem including clouds and clear skies. PyRTE-RRTMGP is a Python implementation of the Radiative Transfer for Energetics (RTE).
#
# ## Overview
#
# PyRTE-RRTMGP provides a flexible and efficient framework for computing radiative fluxes in planetary atmospheres. This example shows an end-to-end problem with both clear skies and clouds. 
#
# 1. Loading data for cloud and gas optics 
# 2. Computing gas and cloud optical properties and combining them to produce an all-sky problem
# 3. Solving the radiative transfer equation to obtain upward and downward fluxes
# 4. Validating results against reference solutions generated with the original RTE fortran code
#
# The package leverages `xarray` to represent data.
#
# ## Key Components
#
# - **Gas and Cloud Optics**: Handles spectral properties of atmospheric gases and clouds and combines them to make a complete problem
# - **RTE Solver**: Computes radiative fluxes based on atmospheric properties
#
# This example demonstrates the workflow for both longwave and shortwave radiative transfer calculations.
#
# See the [documentation](https://pyrte-rrtmgp.readthedocs.io/en/latest/) for more information.

# %% [markdown]
# ## Preliminaries

# %% [markdown]
# ### Installing dependencies

# %%
import numpy as np

# %% [markdown]
# ### Importing pyRTE components

# %%
from pyrte_rrtmgp import rrtmgp_cloud_optics, rrtmgp_gas_optics
from pyrte_rrtmgp.data_types import (
    CloudOpticsFiles,
    GasOpticsFiles,
    OpticsProblemTypes,
)
from pyrte_rrtmgp.rte_solver import rte_solve
from pyrte_rrtmgp.examples import (
    compute_clouds,
    compute_profiles,
    load_example_file,
    ALLSKY_EXAMPLES,
)


# %% [markdown]
# ### Setting up the problem
#
# The routine `compute_profiles()` packaged with `pyRTE_RRTMGP` computes temperature, pressure, and humidity profiles following a moist adibat. The concentrations of other gases are also needed. Clouds are distributed in 2/3 of the columns 

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
atmosphere

# %% [markdown]
# ## Longwave calculations
#
# In this example datasets are saved to intermediate variables at each step

# %% [markdown]
# ### Initialize the cloud and gas optics data 

# %%
cloud_optics_lw = rrtmgp_cloud_optics.load_cloud_optics(
    cloud_optics_file=CloudOpticsFiles.LW_BND
)
gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
    gas_optics_file=GasOpticsFiles.LW_G256
)

cloud_optics_lw, gas_optics_lw

# %% [markdown]
# ### Atmospheric profiles - clear sky, then clouds

# %%
atmosphere = make_profiles()
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
# ### Clear-sky (gases) optical properties; surface boundary conditions 

# %%
optical_props = gas_optics_lw.compute_gas_optics(
    atmosphere, problem_type=OpticsProblemTypes.ABSORPTION, add_to_input=False
)
optical_props["surface_emissivity"] = 0.98
optical_props

# %% [markdown]
# ### Calculate cloud properties; create all-sky optical properties
#
# First compute the optical properties of the clouds

# %%
clouds_optical_props = cloud_optics_lw.compute_cloud_optics(
    atmosphere, problem_type=OpticsProblemTypes.ABSORPTION
)
# The optical properties of the clouds alone
clouds_optical_props

# %% [markdown]
# Then add the optical properties of the clouds to the clear sky to get the total

# %%
# add_to() changes the value of `optical_props`
clouds_optical_props.add_to(optical_props)
optical_props

# %% [markdown]
# ### Compute broadband fluxes

# %%
fluxes = rte_solve(optical_props, add_to_input=False)
fluxes

# %% [markdown]
# ### Compare to reference results

# %%
# Load reference data and verify results
ref_data = load_example_file(ALLSKY_EXAMPLES.REF_LW_NO_AEROSOL)
assert np.isclose(
    fluxes["lw_flux_up"],
    ref_data["lw_flux_up"].T,
    atol=1e-7,
).all()

assert np.isclose(
    fluxes["lw_flux_down"],
    ref_data["lw_flux_dn"].T,
    atol=1e-7,
).all()

print("All-sky longwave calculations validated successfully")

# %% [markdown]
# # Shortwave calculations
#
# In this example steps are combined where possible

# %% [markdown]
# ## Initialize optics data 

# %%
cloud_optics_sw = rrtmgp_cloud_optics.load_cloud_optics(
    cloud_optics_file=CloudOpticsFiles.SW_BND
)
gas_optics_sw = rrtmgp_gas_optics.load_gas_optics(
    gas_optics_file=GasOpticsFiles.SW_G224
)

cloud_optics_sw, gas_optics_sw

# %% [markdown]
# ### Atmospheric profiles - clear sky, then clouds

# %%
atmosphere = make_profiles()
#
# Temporary workaround - compute_clouds() needs to know the particle size;
#    that's set as the mid-point of the valid range from cloud_optics
#
atmosphere = atmosphere.merge(
    compute_clouds(
        cloud_optics_sw, atmosphere["pres_layer"], atmosphere["temp_layer"]
    )
)
atmosphere

# %% [markdown]
# ### Compute gas and cloud optics and combine in one step

# %%
# compute_cloud_optics() returns two-stream properties by default?
optical_props = gas_optics_sw.compute_gas_optics(
    atmosphere, problem_type=OpticsProblemTypes.TWO_STREAM, add_to_input=False
)
# add_to() changes the values in optical_props
cloud_optics_sw.compute_cloud_optics(atmosphere).add_to(
    optical_props, delta_scale=True
)
#
# Add SW-specific surface and angle properties
#
optical_props["surface_albedo_direct"] = 0.06
optical_props["surface_albedo_diffuse"] = 0.06
# Could also specific a single "surface_albedo"
optical_props["mu0"] = 0.86
optical_props

# %% [markdown]
# ### Compute fluxes

# %%
fluxes = rte_solve(optical_props, add_to_input=False)
fluxes

# %% [markdown]
# ### Compare to reference results

# %%
ref_data = load_example_file(ALLSKY_EXAMPLES.REF_SW_NO_AEROSOL)
assert np.isclose(
    fluxes["sw_flux_up"],
    ref_data["sw_flux_up"].T,
    atol=1e-7,
).all()
assert np.isclose(
    fluxes["sw_flux_down"],
    ref_data["sw_flux_dn"].T,
    atol=1e-7,
).all()
assert np.isclose(
    fluxes["sw_flux_dir"],
    ref_data["sw_flux_dir"].T,
    atol=1e-7,
).all()

print("All-sky shortwave calculations validated successfully")
