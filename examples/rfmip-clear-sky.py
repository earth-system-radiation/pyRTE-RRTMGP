# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
# # Clear sky example (RFMIP)
#

# %% [markdown]
# ## Overview
#
# This notebook demonstrates the use of pyRTE-RRTMP to solve the simple problem of computing 
#    clear-sky broadband (spectrally-integrated) fluxes. The examples use a set of atmospheric 
#    conditions used in the Radiative Forcing Model Intercomparison Project. The conditions 
#    are described in [this paper](https://doi.org/10.1029/2020JD033483). The conditions, as 
#    well as the results for the reference Fortran implementation of RTE-RRTMGP, are downloaded 
#    by the Python package. 
#    
#  Although they are part of the same Python package we distinguish between `pyRRTMGP`, 
#    which converts a description of the atmosphere into a radiative transfer problem, and 
#    `pyRTE` which solves the radiative transfer problem to determine broadband fluxes
#
# pyRTE-RRTMGP relies on `xarray` representions of data and `dask` for parallelization 

# ## The workflow 
#
# For both longwave and shortwave problems we will 
# 1. Initialize pyRRTMGP by reading the gas optics data
# 2. Read the RFMIP atmospheric conditions
# 3. Compute spectrally-resolved gas optics properties 
# 4. Solve the radiative transfer equation to obtain upward and downward fluxes
# 5. Check the results against the reference solutions generated with the original RTE fortran code
#

# %% [markdown]
# # Setting up the problem 
#
# ## Dependencies

# %%
import numpy as np
import xarray as xr

from pyrte_rrtmgp.rrtmgp_data_files import (
    CloudOpticsFiles,
    GasOpticsFiles,
)
from pyrte_rrtmgp import rte
from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.examples import RFMIP_FILES, load_example_file

# %% [markdown]
# ## Initialize pyRRTMGP gas optics calculations 

# %%
gas_optics_lw = GasOptics(
    gas_optics_file=GasOpticsFiles.LW_G256
)

gas_optics_sw = GasOptics(
    gas_optics_file=GasOpticsFiles.SW_G224
)

# %% [markdown]
# ## Read the RFMIP atmopheric profiles

# %%
atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)

# %% [markdown]
# Layer pressures and temperatures are bounded by range of the empirical 
#   data. Level pressures are only restricted to be > 0 but the reference 
#   results were produced using the minimum allowed layer pressure. 
#   We reproduce that restriction here to get the same answers 
#   as the reference calculation. 

# %%
atmosphere["pres_level"] = xr.ufuncs.maximum(
    gas_optics_sw.press_min,
    atmosphere["pres_level"],
)

# %% [markdown]
# ## Conform to expectations
#
# pyRRTMGP interprets the input `xr.Dataset` by looking for `xr.DataArray`s with specific names. 
#   Those names can be over-ridden via a mapping. 
#   Here we create such a mapping for the gases in the RFMIP dataset. 
#   The names as RRTMGP expects them are the keys in the dictionary; the valus are the names in the RFMIP dataset
#

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

# %% [markdown]
# # Compute the spectrally-dependent optical properties 

# %% [markdown]
# For the longwave problem we will make a new dataset with the optical properties 
#   pyRRTMGP compute the optical properties (just optical depth `tau` for the longwave problem) and three 
#   radiation source functions (on layers, on levels, and at the surface)

# %%
optical_props = gas_optics_lw.compute(
    atmosphere,
    gas_name_map=gas_mapping,
    add_to_input = False,
)
optical_props

# %% [markdown]
# For the shortave problem we will append the optical properties to the original dataset 
#    Shortwave problems have three optical properties (`tau`, `ssa`, and `g`) but a single 
#    source function defined at the top of atmosphere 

# %%
gas_optics_sw.compute(
    atmosphere,
    gas_name_map=gas_mapping,
)
atmosphere


# %% [markdown]
# ## Solve the Radiative Transfer Equation
#
# Before we can solve the radiative transfer equation we need to specify the boundary conditions - 
#   for longwave radiation, the `surface_emissivity` which here comes from the RFMIP conditions 

# %% 
optical_props["surface_emissivity"] = atmosphere.surface_emissivity

# %% [markdown]
# With the problem specified (optical properties, source functions, boundary conditions), 
#   we can now solve the radiative transfer equation to find
#   the upward and downward broadband radiative fluxes for each atmospheric profile.
#
# For the longwave problem we use the dataset containing only the radiative transfer problem. 
# All the arrays for the shortwave are in the same dataset

# %%
lw_fluxes = optical_props.rte.solve(
	add_to_input=False, 
)

atmosphere.rte.solve() 

# %% [markdown]
# ## Check the results against the reference solutions 
#
# We compare all fluxes (up and down, shortwave and longwave) against the results of the reference code to ensure we 
#    have the same results to within some tolerance 
#
# ### Read the reference results 
# %%
ref = xr.merge([
	load_example_file(RFMIP_FILES.REFERENCE_RLU), 
	load_example_file(RFMIP_FILES.REFERENCE_RLD),
	load_example_file(RFMIP_FILES.REFERENCE_RSU), 
	load_example_file(RFMIP_FILES.REFERENCE_RSD),
	]
)

# %% [markdown]
# ### Compare longave results

# %%
assert np.isclose(
    lw_fluxes["lw_flux_up"].transpose("expt", "site", "level"),
    ref["rlu"],
    atol=1e-7,
).all(), "Longwave flux up mismatch"
assert np.isclose(
    lw_fluxes["lw_flux_down"].transpose("expt", "site", "level"),
    ref["rld"],
    atol=1e-7,
).all(), "Longwave flux down mismatch"


# %% [markdown]
# ### Compare shortwave results

# %%
assert np.isclose(
    atmosphere["sw_flux_up"].transpose("expt", "site", "level"),
    ref["rsu"],
    atol=1e-7,
).all(), "Shortwave flux up mismatch"
assert np.isclose(
    atmosphere["sw_flux_down"].transpose("expt", "site", "level"),
    ref["rsd"],
    atol=1e-7,
).all(), "Shortwave flux down mismatch"

# %%
print("RFMIP clear-sky calculations validated")

# %% [markdown]
# # Variants
#
# See the `pyRTE-quick-start notebook for more examples, including how to parallelize computations with `dask` and 
#   how to add clouds to the problem, and how to combine multiple steps of the calculation at once 
