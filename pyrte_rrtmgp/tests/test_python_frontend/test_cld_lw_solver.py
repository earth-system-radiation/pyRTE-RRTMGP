import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import CloudOpticsFiles, GasOpticsFiles, AllSkyExampleFiles
from pyrte_rrtmgp.rrtmgp_gas_optics import load_gas_optics
import pyrte_rrtmgp.rrtmgp_cloud_optics
from pyrte_rrtmgp.rrtmgp_cloud_optics import load_cloud_optics
from pyrte_rrtmgp.utils import compute_profiles, compute_clouds, load_rrtmgp_file
from pyrte_rrtmgp.rte_solver import rte_solve


def test_lw_solver_with_clouds() -> None:
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
    cloud_optics_lw = load_cloud_optics(cloud_optics_file=CloudOpticsFiles.LW_BND)

    # Calculate cloud properties and merge into the atmosphere dataset
    cloud_properties = compute_clouds(
        cloud_optics_lw, atmosphere["pres_layer"], atmosphere["temp_layer"]
    )
    atmosphere = atmosphere.merge(cloud_properties)

    # Calculate cloud optical properties
    clouds_optical_props = cloud_optics_lw.compute_cloud_optics(
        atmosphere, problem_type="absorption"
    )

    # Calculate gas optical properties
    gas_optics_lw = load_gas_optics(gas_optics_file=GasOpticsFiles.LW_G256)
    clear_sky_optical_props = gas_optics_lw.compute_gas_optics(
        atmosphere, problem_type="absorption", add_to_input=False
    )
    clear_sky_optical_props["surface_emissivity"] = 0.98

    # Combine optical properties and solve RTE
    clouds_optical_props.add_to(clear_sky_optical_props)

    fluxes = rte_solve(clear_sky_optical_props, add_to_input=False)
    assert fluxes is not None

    # Load reference data and verify results
    ref_data = load_rrtmgp_file(AllSkyExampleFiles.LW_NO_AEROSOL)
    assert np.isclose(fluxes["lw_flux_up"], ref_data["lw_flux_up"].T, atol=1e-7).all()
    assert np.isclose(fluxes["lw_flux_down"], ref_data["lw_flux_dn"].T, atol=1e-7).all()
