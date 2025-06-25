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
# # Shortwave Radiative Transfer Example with PyRTE-RRTMGP
#
# This notebook demonstrates how to use the PyRTE-RRTMGP package to solve a shortwave radiative transfer problem. PyRTE-RRTMGP is a Python implementation of the Radiative Transfer for Energetics (RTE).
#
# ## Overview
#
# PyRTE-RRTMGP provides a flexible and efficient framework for computing radiative fluxes in planetary atmospheres. This example specifically focuses on:
#
# 1. Loading gas optics data for shortwave radiation
# 2. Processing atmospheric data from the RFMIP (Radiative Forcing Model Intercomparison Project)
# 3. Computing gas optics properties
# 4. Solving the radiative transfer equation to obtain upward and downward fluxes
# 5. Validating results against reference solutions generated with the original RTE fortran code
#
# The package leverages xarray and dask for efficient data handling and parallel computation, making it suitable for large-scale atmospheric modeling applications.
#
# ## Key Components
#
# - **Gas Optics**: Handles spectral properties of atmospheric gases
# - **RTE Solver**: Computes radiative fluxes based on atmospheric properties
# - **Data Handling**: Uses xarray for labeled, multi-dimensional data structures
#
# This example demonstrates the workflow for shortwave radiative transfer calculations, which are essential for understanding Earth's energy budget and climate modeling.
#
# See the [documentation](https://pyrte-rrtmgp.readthedocs.io/en/latest/) for more information.
#

# %% [markdown]
# ## Setup and Configuration
#
# First, we import the necessary libraries and modules. PyRTE-RRTMGP relies on:
# - **numpy** and **xarray** for data handling
# - Various modules from the `pyrte_rrtmgp` package used to load the data
#
# The key components we'll use are:
# - `rrtmgp_gas_optics`: Handles spectral properties of gases
# - `rte_solver`: Solves the radiative transfer equation

# %%
import numpy as np
import xarray as xr

from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.data_types import (
    GasOpticsFiles,
    OpticsProblemTypes,
)
from pyrte_rrtmgp.rte_solver import rte_solve
from pyrte_rrtmgp.examples import RFMIP_FILES, load_example_file

# %% [markdown]
# ## Loading Gas Optics Data
#
# We get the default data files from the package that are available in the [rrtmgp-data](https://github.com/earth-system-radiation/rrtmgp-data) repository.
#
# We're using the shortwave gas optics file with 224 g-points (`SW_G224`).
#
# The atmosphere is the RFMIP (Radiative Forcing Model Intercomparison Project) dataset. This dataset contains atmospheric profiles with various combinations of temperature, pressure, and gas concentrations.

# %%
gas_optics_sw = rrtmgp_gas_optics.load_gas_optics(
    gas_optics_file=GasOpticsFiles.SW_G224
)
atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)

# %% [markdown]
# Level pressures need to be >= 0 but the Fortran reference results
#   were produced using a minimum level pressure matching the minimum 
#   pressure at which the gas optics tables are calculated, so we 
#   need to match that minimum pressure to get the same answers 


# %%
atmosphere["pres_level"] = xr.ufuncs.maximum(
    gas_optics_sw.compute_gas_optics.press_min,
    atmosphere["pres_level"],
)

# %% [markdown]
# # Computing Gas Optics
#
# Next, we define the gas mapping dictionary that specifies the gas names in the atmosphere dataset, the gas optics file names are the keys in the dictionary and are the default names used internally in the package.
#
# With that we compute the gas optics for the atmosphere `two-stream` problem type. The computed gas optics are stored in the `atmosphere` Dataset.

# %%
gas_mapping = {
    "h2o": "water_vapor",
    "co2": "carbon_dioxide_GM",
    "o3": "ozone",
    "n2o": "nitrous_oxide_GM",
    "co": "carbon_monoxide_GM",
    "ch4": "methane_GM",
    "o2": "oxygen_GM",
    "n2": "nitrogen_GM",
    "ccl4": "carbon_tetrachloride_GM",
    "cfc11": "cfc11_GM",
    "cfc12": "cfc12_GM",
    "cfc22": "hcfc22_GM",
    "hfc143a": "hfc143a_GM",
    "hfc125": "hfc125_GM",
    "hfc23": "hfc23_GM",
    "hfc32": "hfc32_GM",
    "hfc134a": "hfc134a_GM",
    "cf4": "cf4_GM",
    "no2": "no2",
}

gas_optics_sw.compute_gas_optics(
    atmosphere,
    problem_type=OpticsProblemTypes.TWO_STREAM,
    gas_name_map=gas_mapping,
)
atmosphere["tau"]

# %% [markdown]
# ## Solving the Radiative Transfer Equation
#
# With the gas optics properties computed, we can now solve the radiative transfer equation using the `rte_solve`. This will calculate the upward and downward shortwave radiative fluxes for each atmospheric profile.

# %%
fluxes = rte_solve(atmosphere, add_to_input=False)
fluxes

# %% [markdown]
# ## Validating Results Against Reference Solutions
#
# Finally, we validate our computed fluxes against reference solutions. The reference data comes from the original RTE-RRTMGP implementation and makes sure that the implementation is correct.
#
# We compare both upward (`rsu`) and downward (`rsd`) shortwave fluxes to ensure our implementation produces accurate results within the specified error tolerance (`ERROR_TOLERANCE = 1e-7`).
#

# %%
rsu = load_example_file(RFMIP_FILES.REFERENCE_RSU)
rsd = load_example_file(RFMIP_FILES.REFERENCE_RSD)

assert np.isclose(
    fluxes["sw_flux_up"].transpose("expt", "site", "level"),
    rsu["rsu"],
    atol=1e-7,
).all(), "Shortwave flux up mismatch"
assert np.isclose(
    fluxes["sw_flux_down"].transpose("expt", "site", "level"),
    rsd["rsd"],
    atol=1e-7,
).all(), "Shortwave flux down mismatch"

print("Shortwave clear-sky (RFMIP) calculations validated successfully!")
