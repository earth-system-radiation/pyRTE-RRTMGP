"""
Minimal longwave GasOptics smoke test.

This script builds one spectral gas-optics object, feeds it a small two-layer
atmospheric xarray Dataset, runs compute(), and checks that the returned
compute() fields are strictly positive. Field names and dimension names
(``layer``/``level``) match the RRTMGP convention used elsewhere in
pyRTE-RRTMGP, so the same Dataset works with either gas-optics implementation.
"""

import numpy as np
import xarray as xr
from pyrte_rrtmgp.ssm import GasOptics, SSM_CP26

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
    spectral_data=SSM_CP26,
    nus=nus,
    dnus=dnus,
    pref=SSM_CP26.pref,
)

# Two-layer atmospheric input using the standard layer/level dimension names.
# Pressure decreases with index (surface first), matching rce-states.nc. The
# well-mixed co2 is given once for the column and broadcast across layers.
layer = xr.Dataset(
    data_vars={
        "pres_level": (["level"], np.array([1.001e05, 9.90e04, 8.00e04])),
        "pres_layer": (["layer"], np.array([9.955e04, 8.95e04])),
        "temp_layer": (["layer"], np.array([300.0, 295.0])),
        "temp_level": (["level"], np.array([300.0, 297.0, 293.0])),
        "h2o": (["layer"], np.array([0.03661, 0.02])),
        "co2": np.array(400e-6),
    },
)
layer["surface_temperature"] = np.array(305.0)

# Run the longwave gas-optics calculation.
assert result is not None
result = gas_optics.compute(layer)

assert bool((result.tau > 0.0).all())
assert bool((result.layer_source > 0.0).all())
assert bool((result.level_source > 0.0).all())
assert bool((result.surface_source > 0.0).all())
