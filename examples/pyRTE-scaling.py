# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: easy
#     language: python
#     name: easy
# ---

# %%
import numpy as np
import xarray as xr

# %%
from pyrte_rrtmgp import rrtmgp_cloud_optics, rrtmgp_gas_optics
from pyrte_rrtmgp.data_types import (
    CloudOpticsFiles,
    GasOpticsFiles,
    OpticsProblemTypes,
)
from pyrte_rrtmgp.rte_solver import rte_solve

# %% [markdown]
# # Read data
#
# Zoom level 5 is 12288 points (roughly 5 degrees); each zoom level (max 11, min 0) is 4x more, fewer points or 2x higher in grid density

# %%
zoom = 5
data = xr.open_dataset(f"https://s3.eu-dkrz-1.dkrz.cloud/wrcp-hackathon/data/ICON/d3hp003feb.zarr/PT15M_inst_z{zoom}_atm", 
                       engine="zarr")

data = xr.open_dataset("/Users/robert/Codes/hk25/data/PT15M_inst_z5_atm", consolidated=True, engine='zarr')
data

# %% [markdown]
# # Transform the data 
#
# - convert `qall` to `lwp`, `iwp`, `rel`, `rei`
#

# %%
data.pressure

# %% [markdown]
# ## Pressure values on levels
#  Top level pressure is arbitrary, bottom level presure would normally be surface pressure but these data have been 
#  interpolated to fixed pressures so the surface pressure can be well below 1000 hPa 

# %%
data["pressure_h"] = xr.concat([
                        xr.DataArray([1], dims="pressure_h"),
                        xr.DataArray(((data.pressure.values[1:] * data.pressure.values[:-1]) ** 0.5), dims="pressure_h"),
                        xr.DataArray([100500], dims="pressure_h")
                    ], dim = "pressure_h")

# %% [markdown]
# ## Temperature on levels
# Linear interpolation of temperature in pressure
#   Temperature at top level is same as top layer 
#   Temperature at bottom level is surface T (might be colder than lowest layer when elevation is high) 
#

# %%
data["ta_h"] = data.ta.interp(pressure=data.pressure_h)
data["ta_h"][:,  0, :] = data.ta[:,  0, :]
data["ta_h"][:, -1, :] = data.ts

# %% [markdown]
# ## Water vapor - convert specific humidity to molar mixing ratio 
#
# Molar mixing ratio assuming specific humidity is water mass/dry air mass (not quite correct)
#   Md, Mw are molar mases of dry air and water vapor 
#

# %%
Md = 0.0289652
Mw = 0.018016
data["h2o"] = data.hus * (Md/Mw)


# %% [markdown]
# ## Ozone from monthly-mean ERA5 interpolated onto HEALPix grid at zoom levels 8 and below

# %%
if zoom <= 8:
    data["o3"] =xr.open_dataset(
        f"https://swift.dkrz.de/v1/dkrz_41aca03ec414c2f95f52b23b775134f/reanalysis/v1/ERA5_P1M_{zoom}.zarr",
        engine="zarr")\
        .sel(time="2020-02-01", method="nearest")\
        .o3.interp(level=data.pressure)

# For zoom > 8 we need to interpolate in space too - probably nearest neighbor 

# %% [markdown]
# ## Well-mixed greenhouse gases (these are pre-industrial values, should update) 

# %%
gas_values = {
        "co2": 348e-6,
        "ch4": 1650e-9,
        "n2o": 306e-9,
        "n2": 0.7808,
        "o2": 0.2095,
        "co": 0.0,
}
for gas_name, value in gas_values.items():
    data[gas_name] = value

# %% [markdown]
# ## Cloud properties 
#   Data set includes only `qall` all hydrometeors 
#   Assume all clouds > 263 are liquid, everything else is ice (could refine)
#   Convert from MMR to vertically-integrated LWP, IWP 

# %%
data["lwp"] = np.where(data.ta >= 263., qall, 0)  
data["iwp"] = np.where(data.ta <  263., qall, 0)  

# Liquid and ice effective sizes in microns 
data["rel"] = np.where(data.lwp > 0., 10., 0)  
data["rei"] = np.where(data.iwp > 0,  35., 0)  

# %%
data

# %%
data.rename_dims({"pressure":"layer", 
                  "pressure_h":"level"})

# %% [markdown]
# # RTE and RRTMPG initialization 
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
# # Compute fluxes - shortwave 

# %%
fluxes = rte_solve(
    xr.merge(
        [cloud_optics_sw.compute_cloud_optics(data).add_to(
            gas_optics_sw.compute_gas_optics(
                data, 
                problem_type=OpticsProblemTypes.TWO_STREAM, 
                add_to_input=False,
            ), 
            delta_scale=True,
        ), 
        xr.Dataset(data_vars = {"surface_albedo":0.06, 
                                "mu0":0.86})],
    ), 
    add_to_input = False,
)

# %% [markdown]
# # Compute fluxes - shortwave 


# %%
fluxes = rte_solve(
    xr.merge(
        [cloud_optics_lw.compute_cloud_optics(data).add_to(
            gas_optics_lw.compute_gas_optics(
                data, 
                problem_type=OpticsProblemTypes.ABSORPTION, 
                add_to_input=False,
            ), 
        ), 
        xr.Dataset(data_vars = {"surface_emissivity":0.98})],
    ), 
    add_to_input = False,
)

