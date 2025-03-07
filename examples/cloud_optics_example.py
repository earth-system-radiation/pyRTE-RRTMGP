#! /usr/env/python
"""Compute cloud optics for a large sample of clouds."""
#
#   fetch-DYAMOND-data.py must be run first.
#
import scipy as sc
import xarray as xr

#
# Only about half the levels contain clouds so
#   for cloud optics only we can limit the data we import
#
min_lev_liquid = 107
# There are almost no liquid clouds in levels higher (smaller index is higher) than 107
min_lev_ice = 78
# There are almost no ice clouds in levels higher (smaller index is higher) than 78

atmosphere = xr.open_mfdataset(
    "GEOS-DYAMOND2-data/*.nc4",
    drop_variables=[
        "anchor",
        "cubed_sphere",
        "orientation",
        "contacts",
        "corner_lats",
        "corner_lons",
    ],
).isel(
    lev=slice(
        min_lev_ice,
    )
)
#
# By default the dask arrays are contiguous in Xdim and Ydim (first two dimensions)
#   This will work for computing optics but not for computing fluxes,
#   where chunks need to include all layers/levels
#

atmosphere["LWP"] = (atmosphere.DELP * atmosphere.QL) / sc.constants.g
atmosphere["IWP"] = (atmosphere.DELP * atmosphere.QI) / sc.constants.g

#
# What we'd eventutally like to do...
#
# clouds_optical_props = cloud_optics_sw.compute_cloud_optics(atmosphere)
