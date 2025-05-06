import netCDF4 as nc  # noqa
import xarray as xr
import numpy as np

import dask.array as da

from pyrte_rrtmgp import rrtmgp_cloud_optics
from pyrte_rrtmgp import rrtmgp_gas_optics

from pyrte_rrtmgp.data_types import CloudOpticsFiles
from pyrte_rrtmgp.data_types import GasOpticsFiles
from pyrte_rrtmgp.data_types import OpticsProblemTypes

from pyrte_rrtmgp.examples import ALLSKY_EXAMPLES
from pyrte_rrtmgp.examples import compute_profiles
from pyrte_rrtmgp.examples import compute_clouds
from pyrte_rrtmgp.examples import load_example_file

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
    cloud_optics_lw = rrtmgp_cloud_optics.load_cloud_optics(
        cloud_optics_file=CloudOpticsFiles.LW_BND
    )

    # Calculate cloud properties and merge into the atmosphere dataset
    cloud_properties = compute_clouds(
        cloud_optics_lw, atmosphere["pres_layer"], atmosphere["temp_layer"]
    )
    atmosphere = atmosphere.merge(cloud_properties)

    # Calculate cloud optical properties
    clouds_optical_props = cloud_optics_lw.compute_cloud_optics(
        atmosphere, problem_type=OpticsProblemTypes.ABSORPTION
    )

    # Calculate gas optical properties
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )
    optical_props = gas_optics_lw.compute_gas_optics(
        atmosphere,
        problem_type=OpticsProblemTypes.ABSORPTION,
        add_to_input=False
    )
    optical_props["surface_emissivity"] = 0.98

    fluxes = rte_solve(clouds_optical_props.add_to(optical_props),
                       add_to_input=False)
    assert fluxes is not None

    # Load reference data and verify results
    ref_data = load_example_file(ALLSKY_EXAMPLES.REF_LW_NO_AEROSOL)
    assert np.isclose(fluxes["lw_flux_up"],
                      ref_data["lw_flux_up"].T, atol=1e-7).all()
    assert np.isclose(fluxes["lw_flux_down"],
                      ref_data["lw_flux_dn"].T, atol=1e-7).all()


def test_lw_solver_with_clouds_dask() -> None:

    # Set up dimensions
    ncol = 24
    nlay = 72

    # Create atmospheric profiles and gas concentrations
    atmosphere = compute_profiles(300, ncol, nlay)
    atmosphere = atmosphere.chunk("auto")
    assert isinstance(atmosphere, xr.Dataset)
    assert isinstance(atmosphere["pres_layer"].data, da.Array)

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
            coords={"site": range(ncol), "layer": range(nlay)})

    # Load cloud optics data
    cloud_optics_lw = rrtmgp_cloud_optics.load_cloud_optics(
        cloud_optics_file=CloudOpticsFiles.LW_BND
    )
    cloud_optics_lw = cloud_optics_lw.chunk("auto")
    assert isinstance(cloud_optics_lw, xr.Dataset)
    assert isinstance(cloud_optics_lw["asyice"].data, da.Array)

    # Calculate cloud properties and merge into the atmosphere dataset
    cloud_properties = compute_clouds(
        cloud_optics_lw, atmosphere["pres_layer"], atmosphere["temp_layer"]
    )
    assert isinstance(cloud_properties, xr.Dataset)
    assert isinstance(cloud_properties["lwp"].data, da.Array)

    atmosphere = atmosphere.merge(cloud_properties)

    # Calculate cloud optical properties
    clouds_optical_props = cloud_optics_lw.compute_cloud_optics(
        atmosphere, problem_type=OpticsProblemTypes.ABSORPTION
    )

    assert isinstance(clouds_optical_props, xr.Dataset)
    assert isinstance(clouds_optical_props["tau"].data, da.Array)

    # Calculate gas optical properties
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    # TODO: I don't think dask works with compute_gas_optics
    optical_props = gas_optics_lw.compute_gas_optics(
        atmosphere,
        problem_type=OpticsProblemTypes.ABSORPTION,
        add_to_input=False
    )
    optical_props["surface_emissivity"] = 0.98

    # TODO: When chunking the optical_props values final isclose asserts
    # optical_props = optical_props.chunk("auto")

    problem_ds = clouds_optical_props.add_to(optical_props)
    assert isinstance(problem_ds, xr.Dataset)

    # TODO: tau should probably be dask array?
    # assert isinstance(problem_ds["tau"].data, da.Array)

    fluxes = rte_solve(problem_ds, add_to_input=False)

    assert isinstance(fluxes, xr.Dataset)
    # TODO: fluxes output is not dask array
    # assert isinstance(fluxes["lw_flux_up"].data, da.Array)

    assert fluxes is not None

    # Load reference data and verify results
    ref_data = load_example_file(ALLSKY_EXAMPLES.REF_LW_NO_AEROSOL)
    assert np.isclose(fluxes["lw_flux_up"],
                      ref_data["lw_flux_up"].T, atol=1e-7).all()
    assert np.isclose(fluxes["lw_flux_down"],
                      ref_data["lw_flux_dn"].T, atol=1e-7).all()
