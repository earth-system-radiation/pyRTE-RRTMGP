#! /usr/env/python
"""Write a virtual Zaar with icechunk"""

import glob

import icechunk
import xarray as xr
from virtualizarr import open_virtual_dataset

# could replace with URLs
paths = glob.glob("GEOS-DYAMOND2-data/DYAMONDv2_c2880_L181.inst_01hr_3d_*.nc4")

all = xr.merge(
    [
        open_virtual_dataset(
            p,
            drop_variables=[
                "anchor",
                "cubed_sphere",
                "orientation",
                "contacts",
                "corner_lats",
                "corner_lons",
            ],
        )
        for p in paths
    ],
    compat="override",
)

storage = icechunk.local_filesystem_storage("./local/icechunk/store")
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")

all.virtualize.to_icechunk(session.store)
session.commit("Wrote first dataset")
session.close()

ds = xr.open_zarr(
    session.store,
    zarr_version=3,
    consolidated=False,
    chunks={},
)
