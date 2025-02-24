import os

import numpy as np
import xarray as xr

from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data
from pyrte_rrtmgp.rrtmgp_gas_optics import GasOpticsFiles, load_gas_optics
from pyrte_rrtmgp.rte_solver import RTESolver
from pyrte_rrtmgp.rrtmgp_cloud_optics import (
    compute_clouds,
    compute_cloud_optics,
    combine_optical_props,
    delta_scale_optical_props,
)
from pyrte_rrtmgp.utils import compute_profiles, create_gas_dataset

ERROR_TOLERANCE = 1e-7

rte_rrtmgp_dir = download_rrtmgp_data()
rfmip_dir = os.path.join(rte_rrtmgp_dir, "examples", "all-sky")
ref_dir = os.path.join(rfmip_dir, "reference")
sw_clouds = os.path.join(rte_rrtmgp_dir, "rrtmgp-clouds-sw-bnd.nc")


def test_sw_solver_with_clouds() -> None:
    # Set up dimensions
    ncol = 24
    nlay = 72

    # Create atmospheric profiles and gas concentrations
    profiles = compute_profiles(300, ncol, nlay)
    gas_values = {
        "co2": 348e-6,
        "ch4": 1650e-9,
        "n2o": 306e-9,
        "n2": 0.7808,
        "o2": 0.2095,
        "co": 0.0,
    }
    gases = create_gas_dataset(gas_values, dims={"site": ncol, "layer": nlay})

    # Set up atmosphere dataset
    atmosphere = xr.merge([profiles, gases])
    top_at_1 = (
        atmosphere["pres_layer"].values[0, 0] < atmosphere["pres_layer"].values[0, -1]
    )

    # Load SW gas optics and add SW-specific properties
    gas_optics_sw = load_gas_optics(gas_optics_file=GasOpticsFiles.SW_G224)
    ngpt = gas_optics_sw.sizes["gpt"]

    # Add surface albedo and solar angle properties
    atmosphere["surface_albedo_dir"] = xr.DataArray(
        0.06,
        dims=["site", "gpt"],
        coords={"site": np.arange(ncol), "gpt": np.arange(ngpt)},
    )
    atmosphere["surface_albedo_dif"] = xr.DataArray(
        0.06,
        dims=["site", "gpt"],
        coords={"site": np.arange(ncol), "gpt": np.arange(ngpt)},
    )
    atmosphere["mu0"] = xr.DataArray(
        0.86,
        dims=["site", "layer"],
        coords={"site": np.arange(ncol), "layer": np.arange(nlay)},
    )

    # Calculate gas optical properties
    clear_sky_optical_props = gas_optics_sw.gas_optics.compute(
        atmosphere, problem_type="two-stream", add_to_input=False
    )

    # Calculate cloud properties and optical properties
    cloud_optics = xr.load_dataset(sw_clouds)
    cloud_properties = compute_clouds(
        cloud_optics, ncol, nlay, profiles["pres_layer"], profiles["temp_layer"]
    )
    clouds_optical_props = compute_cloud_optics(
        cloud_properties, cloud_optics, lw=False
    )
    clouds_optical_props = delta_scale_optical_props(clouds_optical_props)

    # Combine optical properties and solve RTE
    combined_optical_props = combine_optical_props(
        clouds_optical_props, clear_sky_optical_props
    )
    solver = RTESolver()
    fluxes = solver.solve(combined_optical_props, add_to_input=False)
    assert fluxes is not None

    # Load reference data and verify results
    ref_data = xr.load_dataset(
        os.path.join(ref_dir, "rrtmgp-allsky-sw-no-aerosols.nc"),
        decode_cf=False,
    )

    # Compare results with reference data
    assert np.isclose(
        fluxes["sw_flux_up"], ref_data["sw_flux_up"].T, atol=ERROR_TOLERANCE
    ).all()
    assert np.isclose(
        fluxes["sw_flux_down"], ref_data["sw_flux_dn"].T, atol=ERROR_TOLERANCE
    ).all()
