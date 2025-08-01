# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: pyrte-hk25-notebooks
#     language: python
#     name: pyrte-hk25-notebooks
# ---

# %% [markdown]
# # Introduction
#
# This notebook demonstrates how to compute fluxes from global model data. 
#   Data comes from the ICON contribution to the 
#   [WCRP Global KM scale hackathon](https://github.com/digital-earths-global-hackathon). 
#   Data is read from an online Zarr store accessed through an `intake` catalog. 
#
# The environment needed to run the notebook is described in the local `environment.yml` file.
#
# The notebook might be useful as an example of how to a data set into the form needed 
#   by pyRTE. 
#
# We (the developers) are also using the notebook to refine the performance of pyRTE. 
#   Data is on the HEALPix hierarchial equal-area grid so the spatial resolution and 
#   number of points can be changed by setting the zoom level. 
#
# When run with pyRTE v0.1.1 some computations don't work; we are using the notebook 
#   to diagnose and fix the errors. 
#
#

# %%
import numpy as np
import xarray as xr
import intake

# %%
from pyrte_rrtmgp import rrtmgp_cloud_optics, rrtmgp_gas_optics
from pyrte_rrtmgp.rrtmgp_data_files import (
    CloudOpticsFiles,
    GasOpticsFiles,
)
from pyrte_rrtmgp.data_types import OpticsTypes
from pyrte_rrtmgp import rte

# %%
import warnings

# Suppress specific FutureWarnings matching the message pattern when using cat[...].to_dask()
warnings.filterwarnings(
    "ignore",
    message=".*The return type of `Dataset.dims` will be changed.*",
    category=FutureWarning,
)

# %% [markdown]
# # Read data
#
# Zoom level 5 is 12288 points (roughly 5 degrees); each zoom level (max 11, min 0) is 4x change in the number of points 
#   (or 2x change in grid density)
#
# Perhaps chunks should be introduced at this stage? 

# %%
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']

# %%
zoom = 5
data = cat["icon_d3hp003feb"](zoom=zoom).to_dask()

# %%
data

# %% [markdown]
# # Transform the data to the form needed to compute fluxes 

# %% [markdown]
# ## Pressure values on levels
#  Top level pressure is arbitrary, bottom level presure would normally be surface pressure but these data have been 
#  interpolated to fixed pressures so the surface pressure can be well below 1000 hPa 

# %%
data["pressure_h"] = xr.concat([
                        xr.DataArray([1.25], dims="pressure_h"),
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
Mo3 = .047998
if zoom <= 8:
    data["o3"] = cat["ERA5"](zoom=zoom)\
        .to_dask()\
        .sel(time="2020-02-01", method="nearest")\
        .o3.interp(level=data.pressure)\
        .reset_coords(("lat", "lon", "level", "time"))\
        .drop_vars(("lat", "lon", "level", "time"))\
        .o3 * (Md/Mo3)

data.o3.attrs['units'] = "1"   
# This is actually a mass fraction; need to set to vmr 
# also need to change/delete units
    
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
#   Data set includes only `qall` all hydrometeors. 
#   RRTMGP requires liquid and ice water paths (g/m2) and particle sizes (microns)
#   - Assume all clouds > 263 are liquid, everything else is ice (could refine)
#   - Convert from MMR to vertically-integrated LWP, IWP (haven't done this yet)

# %%
#
# Pressure thickness of each layer 
#
dp = xr.DataArray(xr.ufuncs.abs(data.pressure_h.diff(dim="pressure_h")).values, 
                 dims = ("pressure")) 
#
# Gravity  
#
g = 9.8 
data["lwp"] = xr.where(data.ta >= 263., 
                       data.qall * dp/g * 1000, 
                       0)  
data["iwp"] = xr.where(data.ta <  263., 
                       data.qall * dp/g * 1000, 
                       0)  

# Liquid and ice effective sizes in microns 
data["rel"] = xr.where(data.lwp > 0., 10., 0)  
data["rei"] = xr.where(data.iwp > 0,  35., 0)  

# %%
# ## Change variable and coordinate names to those needed by pyRTE 

# Workaround
#    top_at_1 determination assumes 2D pressure arrays
#    we add this array and drop the 1D pressure variable
#    need to revise to use isel(layer=0)[0] and (layer=-1)[0]
data["p2"] = data["pressure"].broadcast_like(data.ta)

var_mapping = {"p2":"pres_layer", 
               "pressure_h":"pres_level", 
               "ta":"temp_layer", 
               "ta_h":"temp_level",
               "ts":"surface_temperature"}

atmosphere = data.rename_dims({"pressure":"layer", 
                               "pressure_h":"level"})\
                 .rename(var_mapping)\
                 .isel(time=6)\
                 .drop_vars(("pressure", "crs"))
atmosphere

# %% [markdown]
# # pyRTE
#
# ## Initialization 

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
# ## Testing
#
# We should be systematic here, exercising gas optics, 
#   cloud optics, and the complete compuation of fluxes 
#   for all variants of the gas and cloud optics input 
#   files. 
# We should also experiement with dask and no dask 


# %% [markdown]
# ## Test gas optics

# %%
# 
# SW gas optics
#   Not clear that this is running in parallel 
#   And there are some NaNs in the tau field... that's bad
#
sw_optics = gas_optics_sw.compute_gas_optics(
                atmosphere,
                problem_type=OpticsTypes.TWO_STREAM, 
                add_to_input=False,
)



# %%
#
# What index values have NaN values? 
#
sw_optics["tau"].where(xr.ufuncs.isnan(sw_optics["tau"]), drop=True)

# Using 112 gpts nans are present at pressure level 0, some of gpts 0-9, 71-95, 102-111, all cells 

# %%
#
# Using 112 gpts this slice, and another from maybe 110-112, are NaNs
#
sw_optics["tau"].isel(cell=100, layer=-1).where(xr.ufuncs.isnan(sw_optics["tau"].isel(cell=100, layer=-1)), drop=True)
# %%
# 
# LW gas optics
#   Produces non-zero values of tau for 256 gpts 
#   Doesn't work full stop with 128 gpts
#
lw_optics = gas_optics_lw.compute_gas_optics(
                atmosphere,
                problem_type=OpticsTypes.ABSORPTION, 
                add_to_input=False,
)

# %% [markdown]
# ## Test cloud optics

# %%
# 
# Shortwave cloud optics
#
sw_cld_optics = cloud_optics_lw.compute_cloud_optics(
    atmosphere, 
    problem_type=OpticsTypes.TWO_STREAM
)
sw_cld_optics["tau"]

# %%
# 
# Longwave cloud optics
#
lw_cld_optics = cloud_optics_lw.compute_cloud_optics(
    atmosphere, 
    problem_type=OpticsTypes.ABSORPTION
)
lw_cld_optics["tau"]


# %% [markdown]
# ## Compute fluxes from atmosphere conditions  

# %%
#
# Shortwave fluxes 
# 
sw_fluxes = xr.merge(
        [cloud_optics_sw.compute_cloud_optics(
            atmosphere, 
            problem_type=OpticsTypes.TWO_STREAM, 
         )\
         .rte.add_to(
             gas_optics_sw.compute_gas_optics(
                    atmosphere,
                    problem_type=OpticsTypes.TWO_STREAM, 
                    add_to_input=False,
             ),
             delta_scale = True,
         ), 
         xr.Dataset(data_vars = {"surface_albedo":0.06, 
                                "mu0":0.86}
                   ),
        ],
    ).
    rte.solve(add_to_input = False,
)

# %%
#
# Longwave fluxes 
# 
lw_fluxes = xr.merge(
        [cloud_optics_lw.compute_cloud_optics(
            atmosphere, 
            problem_type=OpticsTypes.ABSORPTION, 
         )\
         .rte.add_to(
             gas_optics_lw.compute_gas_optics(
                    atmosphere,
                    problem_type=OpticsTypes.ABSORPTION, 
                    add_to_input=False,
             ), 
         ), 
        xr.Dataset(data_vars = {"surface_emissivity":0.98}),
        ],
    ).
    rte.solve(add_to_input = False,
)

