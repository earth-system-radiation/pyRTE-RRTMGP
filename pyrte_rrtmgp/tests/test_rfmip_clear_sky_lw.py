import netCDF4  # noqa
import numpy as np
import dask.array as da

import pytest
import xarray as xr

from pyrte_rrtmgp.data_types import GasOpticsFiles
from pyrte_rrtmgp.data_types import OpticsProblemTypes

from pyrte_rrtmgp.tests import DEFAULT_GAS_MAPPING
from pyrte_rrtmgp.tests import ERROR_TOLERANCE

from pyrte_rrtmgp.examples import load_example_file
from pyrte_rrtmgp.examples import RFMIP_FILES

from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.rte_solver import rte_solve


def test_rfmip_clr_sky_lw() -> None:
    # Load gas optics
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    # Load atmosphere data
    atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)
    atmosphere = atmosphere.sel(expt=0)  # only one experiment

    atmosphere["pres_level"] = xr.where(
        atmosphere["pres_level"] < gas_optics_lw.compute_gas_optics.press_min,
        gas_optics_lw.compute_gas_optics.press_min,
        atmosphere["pres_level"],
    )

    # Compute gas optics for the atmosphere
    gas_optics_lw.compute_gas_optics(
        atmosphere, problem_type=OpticsProblemTypes.ABSORPTION,
        gas_name_map=DEFAULT_GAS_MAPPING
    )

    # Solve RTE
    fluxes = rte_solve(atmosphere, add_to_input=False)
    assert fluxes is not None

    # Load reference data
    rlu = load_example_file(RFMIP_FILES.REFERENCE_RLU)
    rld = load_example_file(RFMIP_FILES.REFERENCE_RLD)
    ref_flux_up = rlu.isel(expt=0)["rlu"]
    ref_flux_down = rld.isel(expt=0)["rld"]

    # Compare results with reference data
    assert np.isclose(fluxes["lw_flux_up"],
                      ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(fluxes["lw_flux_down"],
                      ref_flux_down, atol=ERROR_TOLERANCE).all()


def test_rfmip_clr_sky_lw_dask() -> None:
    # Load gas optics
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    # Load atmosphere data
    atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)
    atmosphere = atmosphere.chunk({"expt": 3})
    atmosphere["pres_level"] = xr.ufuncs.maximum(
        gas_optics_lw.compute_gas_optics.press_min,
        atmosphere["pres_level"],
    )

    assert isinstance(atmosphere, xr.Dataset)
    assert isinstance(atmosphere["lon"].data, da.Array)

    # Compute gas optics for the atmosphere
    gas_optics_lw.compute_gas_optics(
        atmosphere, problem_type=OpticsProblemTypes.ABSORPTION,
        gas_name_map=DEFAULT_GAS_MAPPING
    )

    # Solve RTE
    fluxes = rte_solve(atmosphere, add_to_input=False)
    assert fluxes is not None

    # Load reference data
    rlu = load_example_file(RFMIP_FILES.REFERENCE_RLU)
    rld = load_example_file(RFMIP_FILES.REFERENCE_RLD)

    ref_flux_up = rlu["rlu"]
    ref_flux_down = rld["rld"]

    # Compare results with reference data
    assert np.isclose(fluxes["lw_flux_up"].transpose("expt", "site", "level"),
                      ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(fluxes["lw_flux_down"].transpose("expt", "site", "level"),
                      ref_flux_down, atol=ERROR_TOLERANCE).all()


def test_raises_value_error_if_carbon_monoxide_missing() -> None:

    '''
    Load in LW_G256
    Set up input xarray with/without CO
    Compute gas optics
    compute radiative transfer
    '''

    # Load gas optics
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    # Load atmosphere data
    atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)

    gas_mapping = DEFAULT_GAS_MAPPING.copy()
    del gas_mapping["co"]

    # Compute gas optics for the atmosphere
    with pytest.raises(ValueError):
        gas_optics_lw.compute_gas_optics(
            atmosphere,
            problem_type=OpticsProblemTypes.ABSORPTION,
            gas_name_map=gas_mapping
        )
        _ = rte_solve(atmosphere, add_to_input=False)
