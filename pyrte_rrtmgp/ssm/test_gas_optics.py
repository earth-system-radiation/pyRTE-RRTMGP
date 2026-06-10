import numpy as np
import xarray as xr

from gasOptics import GasOptics


"""
First simple longwave GasOptics test.

This script builds one spectral gas-optics object, feeds it a single-layer
atmospheric xarray Dataset, runs compute(), and prints every returned field.
"""



def print_positive_check(name, dataarray):
    """Report whether all values in one output are strictly positive."""
    is_positive = bool((dataarray > 0).all())
    print(f"{name} all positive: {is_positive}")


# Spectral triangle table. Tags are individual absorption components; tags
# with a suffix, such as h2o-rot, share the physical gas name before "-".
params = ["nu0", "l", "kappa0"]
tags = ["co2", "h2o-rot", "h2o-vr", "h2o-cont"]

atmos_data = xr.Dataset(
    coords={
        "tags": tags,
        "params": params,
    },
    data_vars={
        "triangles": (
            ["tags", "params"],
            np.array(
                [
                    [667.5, 10.2, 500.0],
                    [150.0, 58.0, 165.0],
                    [1500.0, 60.0, 15.0],
                    [700.0, 275.0, 0.1],
                ]
            ),
        )
    },
)

# Wavenumber grid and spectral band widths used by the Planck source terms.
nus = xr.DataArray(
    np.array([100.0, 200.0, 500.0, 700.0, 1000.0]),
    dims=("gpt",),
    name="nus",
    attrs={"units": "cm^-1"},
)

dnus = xr.DataArray(
    np.array([100.0, 100.0, 300.0, 200.0, 300.0]),
    dims=("gpt",),
    coords={"gpt": nus["gpt"]},
    name="dnus",
    attrs={"units": "cm^-1"},
)

# Construct the gas-optics calculator from the spectral data.
gas_optics = GasOptics(
    atmos_data=atmos_data,
    nus=nus,
    dnus=dnus,
    pref=1.0e5,
)

# Single-layer atmospheric input. This keeps the original 1D play/plev shape
# instead of adding a synthetic column dimension.
layer = xr.Dataset(
    coords={
        "plev": np.array([1.001e05, 9.99e04]),
        "play": np.array([1.0e05]),
        "species": np.array(["h2o", "co2"]),
    },
    data_vars={
        "Tlay": (["play"], np.array([300.0])),
        "Tlev": (["plev"], np.array([300.0, 300.0])),
        "dp": (["play"], np.array([200.0])),
        "h2o": (["play"], np.array([0.03661])),
        "co2": (["play"], np.array([400e-6])),
    },
)
layer["surface_temperature"] = np.array([305.0])

# Run the longwave gas-optics calculation.
result = gas_optics.compute(layer)

# Print every field returned by compute() with full values for manual inspection.
print(result)



assert bool((result.tau > 0.0).all())
assert bool((result.lay_source > 0.0).all())
assert bool((result.lev_source > 0.0).all())
assert bool((result.sfc_source > 0.0).all())
