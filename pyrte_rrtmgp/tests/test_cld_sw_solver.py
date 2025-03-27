import netCDF4  # noqa (avoids warning https://github.com/pydata/xarray/issues/7259)

from pyrte_rrtmgp import rrtmgp_cloud_optics
from pyrte_rrtmgp import rrtmgp_gas_optics

from pyrte_rrtmgp.data_types import AllSkyExampleFiles
from pyrte_rrtmgp.data_types import CloudOpticsFiles
from pyrte_rrtmgp.data_types import GasOpticsFiles
from pyrte_rrtmgp.data_types import OpticsProblemTypes

from pyrte_rrtmgp.utils import compute_profiles
from pyrte_rrtmgp.utils import compute_clouds
from pyrte_rrtmgp.utils import load_rrtmgp_file

from pyrte_rrtmgp.rte_solver import rte_solve

import xarray as xr
import numpy as np


def test_load_cloud_optics() -> None:

    cloud_optics_sw = rrtmgp_cloud_optics.load_cloud_optics(
        cloud_optics_file=CloudOpticsFiles.SW_BND
    )
    assert cloud_optics_sw is not None


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
    cloud_optics_sw = rrtmgp_cloud_optics.load_cloud_optics(
        cloud_optics_file=CloudOpticsFiles.SW_BND
    )

    # Calculate cloud properties and merge into the atmosphere dataset
    cloud_properties = compute_clouds(
        cloud_optics_sw, atmosphere["pres_layer"], atmosphere["temp_layer"]
    )
    atmosphere = atmosphere.merge(cloud_properties)

    # Calculate cloud optical properties
    clouds_optical_props = cloud_optics_sw.compute_cloud_optics(atmosphere)

    # Load gas optics and add SW-specific properties
    gas_optics_sw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.SW_G224
    )

    # Calculate gas optical properties
    optical_props = gas_optics_sw.compute_gas_optics(
        atmosphere,
        problem_type=OpticsProblemTypes.TWO_STREAM,
        add_to_input=False
    )

    # Add SW-specific surface and angle properties
    optical_props["surface_albedo_direct"] = 0.06
    optical_props["surface_albedo_diffuse"] = 0.06
    optical_props["mu0"] = 0.86

    fluxes = rte_solve(clouds_optical_props.add_to(optical_props, delta_scale=True),
                       add_to_input=False)
    assert fluxes is not None

    # Load reference data and verify results
    ref_data = load_rrtmgp_file(AllSkyExampleFiles.SW_NO_AEROSOL)
    assert np.isclose(fluxes["sw_flux_up"],
                      ref_data["sw_flux_up"].T, atol=1e-7).all()
    assert np.isclose(fluxes["sw_flux_down"],
                      ref_data["sw_flux_dn"].T, atol=1e-7).all()
