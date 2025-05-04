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

# %%
from datetime import datetime

from pyrte_rrtmgp.external_data_helpers import download_dyamond2_data

# Download the data
downloaded_files = download_dyamond2_data(
    datetime(2020, 2, 1, 9),
    compute_gas_optics=False,
    data_dir="GEOS-DYAMOND2-data",
)

# %%
import xarray as xr
from dask.distributed import Client

from pyrte_rrtmgp import rrtmgp_cloud_optics
from pyrte_rrtmgp.constants import HELMERT1
from pyrte_rrtmgp.data_types import CloudOpticsFiles

nlev = 181
min_lev_liquid = 107
min_lev_ice = 78

# To run in Distributed mode user can uncomment the following lines
client = Client(n_workers=2)
print(f"Dask dashboard available at: {client.dashboard_link}")

# Load the global dataset
atmosphere = (
    xr.open_mfdataset(
        "GEOS-DYAMOND2-data/*inst_01hr_3d_*.nc4",
        drop_variables=[
            "anchor",
            "cubed_sphere",
            "orientation",
            "contacts",
            "corner_lats",
            "corner_lons",
        ],
    )
    .isel(lev=slice(min_lev_ice, nlev))
    .rename({"lev": "layer"})
    .isel(
        Ydim=slice(0, 1000), Xdim=slice(0, 1000), nf=slice(0, 1)
    )  # Ydim=slice(0, 2000), Xdim=slice(0, 2000),
    .chunk({"Xdim": 100, "Ydim": 100, "nf": 1, "layer": -1})
)

# By default the dask arrays are contiguous in Xdim and Ydim (first two dims)
#   This will work for computing optics but not for computing fluxes,
#   where chunks need to include all layers/levels

# Need to convert LWP/IWP to g/m2 and rel/rei to microns
atmosphere["lwp"] = (atmosphere["DELP"] * atmosphere["QL"]) * 1000 / HELMERT1
atmosphere["iwp"] = (atmosphere["DELP"] * atmosphere["QI"]) * 1000 / HELMERT1
atmosphere["rel"] = atmosphere["RL"] * 1e6
atmosphere["rei"] = atmosphere["RI"] * 1e6

needed_vars = ["lwp", "iwp", "rel", "rei"]
cloud_optics_lw = rrtmgp_cloud_optics.load_cloud_optics(
    cloud_optics_file=CloudOpticsFiles.LW_BND
)
tau = cloud_optics_lw.compute_cloud_optics(
    atmosphere[needed_vars], problem_type="absorption", add_to_input=False
)

tau_sum = tau.sum(dim=["layer", "bnd"])["tau"]

# Sort the coordinates to fix the plotting issue
tau_sum = tau_sum.sortby(["Xdim", "Ydim"])

# Plot example
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Create a figure with a cartopy projection
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Plot the data with sorted coordinates
p = tau_sum.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis")

# Add coastlines and gridlines for geographic reference
ax.coastlines()
ax.gridlines(draw_labels=True)

# Add a title
plt.title("Total Cloud Optical Depth")

# Show the plot
plt.tight_layout()


# tau.to_netcdf(
#     "cloud_optics_result.nc",
#     encoding={var: {"zlib": True, "complevel": 9} for var in tau.data_vars},
# )
