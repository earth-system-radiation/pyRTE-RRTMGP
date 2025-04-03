import netCDF4  # noqa
import numpy as np
import dask.array as da

import pytest
import xarray as xr

from pyrte_rrtmgp.data_types import GasOpticsFiles
from pyrte_rrtmgp.data_types import OpticsProblemTypes
from pyrte_rrtmgp.data_types import RFMIPExampleFiles

from pyrte_rrtmgp.utils import load_rrtmgp_file
from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.rte_solver import rte_solve

ERROR_TOLERANCE = 1e-7

DEFAULT_GAS_MAPPING = {
    "h2o": "water_vapor",
    "co2": "carbon_dioxide_GM",
    "o3": "ozone",
    "n2o": "nitrous_oxide_GM",
    "co": "carbon_monoxide_GM",
    "ch4": "methane_GM",
    "o2": "oxygen_GM",
    "n2": "nitrogen_GM",
    "ccl4": "carbon_tetrachloride_GM",
    "cfc11": "cfc11_GM",
    "cfc12": "cfc12_GM",
    "cfc22": "hcfc22_GM",
    "hfc143a": "hfc143a_GM",
    "hfc125": "hfc125_GM",
    "hfc23": "hfc23_GM",
    "hfc32": "hfc32_GM",
    "hfc134a": "hfc134a_GM",
    "cf4": "cf4_GM",
    "no2": "no2",
}


def test_lw_solver_noscat() -> None:
    # Load gas optics
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    # Load atmosphere data
    atmosphere = load_rrtmgp_file(RFMIPExampleFiles.RFMIP)
    atmosphere = atmosphere.sel(expt=0)  # only one experiment

    # Compute gas optics for the atmosphere
    gas_optics_lw.compute_gas_optics(
        atmosphere, problem_type=OpticsProblemTypes.ABSORPTION,
        gas_name_map=DEFAULT_GAS_MAPPING
    )

    # Solve RTE
    fluxes = rte_solve(atmosphere, add_to_input=False)
    assert fluxes is not None

    # Load reference data
    rlu = load_rrtmgp_file(RFMIPExampleFiles.REFERENCE_RLU)
    rld = load_rrtmgp_file(RFMIPExampleFiles.REFERENCE_RLD)
    ref_flux_up = rlu.isel(expt=0)["rlu"]
    ref_flux_down = rld.isel(expt=0)["rld"]

    # Compare results with reference data
    assert np.isclose(fluxes["lw_flux_up"],
                      ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(fluxes["lw_flux_down"],
                      ref_flux_down, atol=ERROR_TOLERANCE).all()


def test_lw_solver_noscat_dask() -> None:
    # Load gas optics
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    # Load atmosphere data
    atmosphere = load_rrtmgp_file(RFMIPExampleFiles.RFMIP)
    atmosphere = atmosphere.chunk({"expt": 3})

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
    rlu = load_rrtmgp_file(RFMIPExampleFiles.REFERENCE_RLU)
    rld = load_rrtmgp_file(RFMIPExampleFiles.REFERENCE_RLD)

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
    atmosphere = load_rrtmgp_file(RFMIPExampleFiles.RFMIP)

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
