import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import CloudOpticsFiles, GasOpticsFiles, AllSkyExampleFiles
from pyrte_rrtmgp.rrtmgp_gas_optics import load_gas_optics
from pyrte_rrtmgp.rrtmgp_cloud_optics import (
    load_cloud_optics,
    compute_cloud_optics,
    combine_optical_props,
    delta_scale_optical_props,
)
from pyrte_rrtmgp.utils import compute_profiles, compute_clouds, load_rrtmgp_file
from pyrte_rrtmgp.rte_solver import RTESolver


def test_sw_solver_with_clouds() -> None:
    # Set up dimensions
    ncol = 24
    nlay = 72

    # Create atmospheric profiles and gas concentrations
    atmosphere = compute_profiles(300, ncol, nlay)

    # Add other gas values
    gas_values = {
        "co2": 348e-6,
        "ch4": 1650e-9,
        "n2o": 306e-9,
        "n2": 0.7808,
        "o2": 0.2095,
        "co": 0.0,
    }

    for gas_name, value in gas_values.items():
        atmosphere[gas_name] = xr.DataArray(
            value,
            dims=["site", "layer"],
            coords={"site": range(ncol), "layer": range(nlay)},
        )

    # Load cloud optics data
    cloud_optics_sw = load_cloud_optics(cloud_optics_file=CloudOpticsFiles.SW_BND)

    # Calculate cloud properties and merge into the atmosphere dataset
    cloud_properties = compute_clouds(
        cloud_optics_sw, atmosphere["pres_layer"], atmosphere["temp_layer"]
    )
    atmosphere = atmosphere.merge(cloud_properties)

    # Calculate cloud optical properties
    clouds_optical_props = compute_cloud_optics(atmosphere, cloud_optics_sw, lw=False)
    clouds_optical_props = delta_scale_optical_props(clouds_optical_props)

    # Load gas optics and add SW-specific properties
    gas_optics_sw = load_gas_optics(gas_optics_file=GasOpticsFiles.SW_G224)

    # Add SW-specific surface and angle properties
    ngpt = gas_optics_sw.sizes["gpt"]
    atmosphere["surface_albedo_dir"] = xr.DataArray(
        np.full((ncol, ngpt), 0.06), dims=["site", "gpt"]
    )
    atmosphere["surface_albedo_dif"] = xr.DataArray(
        np.full((ncol, ngpt), 0.06), dims=["site", "gpt"]
    )
    atmosphere["mu0"] = xr.DataArray(
        np.full((ncol, nlay), 0.86), dims=["site", "layer"]
    )

    # Calculate gas optical properties
    clear_sky_optical_props = gas_optics_sw.gas_optics.compute(
        atmosphere, problem_type="two-stream", add_to_input=False
    )

    # Combine optical properties and solve RTE
    combined_optical_props = combine_optical_props(
        clouds_optical_props, clear_sky_optical_props
    )

    solver = RTESolver()
    fluxes = solver.solve(combined_optical_props, add_to_input=False)
    assert fluxes is not None

    # Load reference data and verify results
    ref_data = load_rrtmgp_file(AllSkyExampleFiles.SW_NO_AEROSOL)
    assert np.isclose(fluxes["sw_flux_up"], ref_data["sw_flux_up"].T, atol=1e-7).all()
    assert np.isclose(fluxes["sw_flux_down"], ref_data["sw_flux_dn"].T, atol=1e-7).all()
