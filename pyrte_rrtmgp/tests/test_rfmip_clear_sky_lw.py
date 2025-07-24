import netCDF4  # noqa
import numpy as np
import dask.array as da

import pytest
import xarray as xr

from pyrte_rrtmgp.data_types import (
    GasOpticsFiles,
    OpticsProblemTypes,
)

from pyrte_rrtmgp.tests import (
    ERROR_TOLERANCE,
)

from pyrte_rrtmgp.examples import (
    load_example_file,
    RFMIP_FILES
)

from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.rte_solver import rte_solve

RFMIP_GAS_MAPPING =  {
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

RFMIP_GAS_MAPPING_SMALL =  {
            "h2o": "water_vapor",
            "co2": "carbon_dioxide_GM",
            "o3": "ozone",
            "n2o": "nitrous_oxide_GM",
            "co": "carbon_monoxide_GM",
            "ch4": "methane_GM",
            "o2": "oxygen_GM",
            "n2": "nitrogen_GM",
        }

def _load_problem_dataset(gas_mapping,
                          use_dask: bool = False) -> xr.Dataset:

    atmosphere: xr.Dataset = load_example_file(RFMIP_FILES.ATMOSPHERE)
    atmosphere["pres_level"] = xr.ufuncs.maximum(
        atmosphere["pres_level"],
        gas_optics_lw.compute_gas_optics.press_min,
    )

    if use_dask:
        atmosphere = atmosphere.chunk({"expt": 3})

    if gas_mapping is None:
        gas_mapping = RFMIP_GAS_MAPPING

    return atmosphere, gas_mapping

def _load_reference_data() -> xr.Dataset:
     return xr.merge([
        load_example_file(RFMIP_FILES.REFERENCE_RLU),
        load_example_file(RFMIP_FILES.REFERENCE_RLD),
        load_example_file(RFMIP_FILES.REFERENCE_RSU),
        load_example_file(RFMIP_FILES.REFERENCE_RSD),
        ])

def _run_lw_test(gas_optics_lw,
                 use_dask: bool = False) -> xr.Dataset:
    """Runs RFMIP clear-sky examples to exercise gas optics, solvers, and gas mapping """
    # Load atmosphere data
    atmosphere: xr.Dataset = load_example_file(RFMIP_FILES.ATMOSPHERE)
    if use_dask:
        atmosphere = atmosphere.chunk({"expt": 3})

    # Compute gas optics for the atmosphere
    gas_optics_lw.compute_gas_optics(
        atmosphere,
        problem_type=OpticsProblemTypes.ABSORPTION,
        gas_name_map=gas_mapping,
    )

    # Solve RTE
    fluxes: xr.Dataset = rte_solve(atmosphere, add_to_input=False)
    assert fluxes is not None
    return fluxes

def _test_verify_rfmip_clr_sky_lw(use_dask: bool = False) -> None:
    """Runs RFMIP clear-sky examples and compares to reference results"""

    # Load gas optics
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file = GasOpticsFiles.LW_G256,
    )

    # Load atmosphere, modify to match
    atmosphere: xr.Dataset = load_example_file(RFMIP_FILES.ATMOSPHERE)
    if use_dask:
        atmosphere = atmosphere.chunk({"expt": 3})
    atmosphere["pres_level"] = xr.ufuncs.maximum(
        atmosphere["pres_level"],
        gas_optics_lw.compute_gas_optics.press_min,
    )

    # Gas optics
    gas_optics_lw.compute_gas_optics(
        atmosphere,
        problem_type=OpticsProblemTypes.ABSORPTION,
        gas_name_map=RFMIP_GAS_MAPPING,
    )
    # Solve RTE
    fluxes: xr.Dataset = rte_solve(atmosphere, add_to_input=False)
    assert fluxes is not None

    # Load reference data (why only a single experiment?)
    ref_fluxes = _load_reference_data()

    # Compare results with reference data
    assert np.allclose(fluxes["lw_flux_up"].transpose("expt", "site", "level"),
                      ref_fluxes["rlu"],
                      atol=ERROR_TOLERANCE)
    assert np.allclose(fluxes["lw_flux_down"].transpose("expt", "site", "level"),
                      ref_fluxes["rld"],
                      atol=ERROR_TOLERANCE)

def test_verify_rfmip() -> None:
    _test_verify_rfmip_clr_sky_lw(use_dask = False)

def test_verify_rfmip_dask() -> None:
    _test_verify_rfmip_clr_sky_lw(use_dask = True)
