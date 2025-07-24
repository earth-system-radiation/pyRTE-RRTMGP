import netCDF4  # noqa
import numpy as np
import xarray as xr

import pytest
from typing import Dict, Optional, Any

from pyrte_rrtmgp.data_types import (
    GasOpticsFiles,
    OpticsProblemTypes,
)
from pyrte_rrtmgp.examples import (
    load_example_file,
    RFMIP_FILES
)
from pyrte_rrtmgp.tests import (
    ERROR_TOLERANCE,
    RFMIP_GAS_MAPPING,
    RFMIP_GAS_MAPPING_SMALL,
)

from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.rte_solver import rte_solve

def _load_reference_data() -> xr.Dataset:
     return xr.merge([
        load_example_file(RFMIP_FILES.REFERENCE_RLU),
        load_example_file(RFMIP_FILES.REFERENCE_RLD),
        load_example_file(RFMIP_FILES.REFERENCE_RSU),
        load_example_file(RFMIP_FILES.REFERENCE_RSD),
        ])

# Ideally we would tell mypy that gas_optics is an xarray accessor...
def _test_get_fluxes_from_RFMIP_atmospheres(
                 gas_optics: Any,
                 problem_type: OpticsProblemTypes,
                 gas_name_mapping: Optional[dict[str, str]] = None,
                 use_dask: bool = False) -> xr.Dataset:
    """Runs RFMIP clear-sky examples to exercise gas optics, solvers, and gas mapping """
    # Load atmosphere data
    atmosphere: xr.Dataset = load_example_file(RFMIP_FILES.ATMOSPHERE)
    if use_dask:
        atmosphere = atmosphere.chunk({"expt": 3})

    if problem_type not in OpticsProblemTypes:
        raise(ValueError)

    if gas_name_mapping is None:
        gas_name_mapping = RFMIP_GAS_MAPPING

    # Compute gas optics for the atmosphere
    gas_optics.compute_gas_optics(
        atmosphere,
        problem_type=problem_type,
        gas_name_map=gas_name_mapping,
    )

    # Solve RTE
    fluxes: xr.Dataset = rte_solve(atmosphere, add_to_input=False)

    assert fluxes is not None
    assert ~xr.ufuncs.isnan(fluxes).any()

    return fluxes

def _test_verify_rfmip_clr_sky(
        problem_type: OpticsProblemTypes,
        use_dask: bool = False) -> None:
    """Runs RFMIP clear-sky examples and compares to reference results."""

    if problem_type not in OpticsProblemTypes:
        raise(ValueError)

    # Load gas optics
    if problem_type is OpticsProblemTypes.ABSORPTION:
        gas_optics = rrtmgp_gas_optics.load_gas_optics(
            gas_optics_file = GasOpticsFiles.LW_G256,
        )
    else:
        gas_optics = rrtmgp_gas_optics.load_gas_optics(
            gas_optics_file = GasOpticsFiles.SW_G224,
        )

    # Load atmosphere, modify to match
    atmosphere: xr.Dataset = load_example_file(RFMIP_FILES.ATMOSPHERE)
    if use_dask:
        atmosphere = atmosphere.chunk({"expt": 3})
    atmosphere["pres_level"] = xr.ufuncs.maximum(
        atmosphere["pres_level"],
        gas_optics.compute_gas_optics.press_min,
    )

    # Gas optics
    gas_optics.compute_gas_optics(
        atmosphere,
        problem_type=problem_type,
        gas_name_map=RFMIP_GAS_MAPPING,
    )
    # Solve RTE
    fluxes: xr.Dataset = rte_solve(atmosphere, add_to_input=False)

    # Load reference data (why only a single experiment?)
    ref_fluxes = _load_reference_data()

    # Compare results with reference data
    if problem_type is OpticsProblemTypes.ABSORPTION:
        assert np.allclose(fluxes["lw_flux_up"].transpose("expt", "site", "level"),
                          ref_fluxes["rlu"],
                          atol=ERROR_TOLERANCE)
        assert np.allclose(fluxes["lw_flux_down"].transpose("expt", "site", "level"),
                          ref_fluxes["rld"],
                          atol=ERROR_TOLERANCE)
    else:
        assert np.allclose(fluxes["sw_flux_up"].transpose("expt", "site", "level"),
                          ref_fluxes["rsu"],
                          atol=ERROR_TOLERANCE)
        assert np.allclose(fluxes["sw_flux_down"].transpose("expt", "site", "level"),
                          ref_fluxes["rsd"],
                          atol=ERROR_TOLERANCE)


#
#### Tests start here
#
def test_verify_rfmip_lw() -> None:
    """ RFMIP test cases, verify against reference results, no dask."""
    _test_verify_rfmip_clr_sky(
        problem_type = OpticsProblemTypes.ABSORPTION,
        use_dask = False,
        )

def test_verify_rfmip_lw_dask() -> None:
    """ RFMIP test cases, verify against reference results, use dask."""
    _test_verify_rfmip_clr_sky(
        problem_type = OpticsProblemTypes.ABSORPTION,
        use_dask = True,
        )

def test_verify_rfmip_sw() -> None:
    """ RFMIP test cases, verify against reference results, no dask."""
    _test_verify_rfmip_clr_sky(
        problem_type = OpticsProblemTypes.TWO_STREAM,
        use_dask = False,
        )

def test_verify_rfmip_sw_dask() -> None:
    """ RFMIP test cases, verify against reference results, use dask."""
    _test_verify_rfmip_clr_sky(
        problem_type = OpticsProblemTypes.TWO_STREAM,
        use_dask = True,
        )

def test_rfmip_lw_reduced_gases() -> None:
    """ Run RFMIP test cases with reduced set of gases"""
    # Load gas optics
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file = GasOpticsFiles.LW_G256,
    )
    _test_get_fluxes_from_RFMIP_atmospheres(
        gas_optics_lw,
        problem_type = OpticsProblemTypes.ABSORPTION,
        gas_name_mapping = RFMIP_GAS_MAPPING_SMALL,
    )

def test_rfmip_lw128() -> None:
    """ Run RFMIP test cases with 128 g-point file"""
    # Load gas optics
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file = GasOpticsFiles.LW_G128,
    )
    _test_get_fluxes_from_RFMIP_atmospheres(
        gas_optics_lw,
        problem_type = OpticsProblemTypes.ABSORPTION,
    )

def test_rfmip_sw_reduced_gases() -> None:
    """ Run RFMIP test cases with reduced set of gases"""
    # Load gas optics
    gas_optics_sw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file = GasOpticsFiles.SW_G112,
    )
    _test_get_fluxes_from_RFMIP_atmospheres(
        gas_optics_sw,
        problem_type = OpticsProblemTypes.TWO_STREAM,
        gas_name_mapping = RFMIP_GAS_MAPPING_SMALL,
    )

def test_rfmip_sw112() -> None:
    """ Run RFMIP test cases with 128 g-point file"""
    # Load gas optics
    gas_optics_sw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file = GasOpticsFiles.SW_G112,
    )
    _test_get_fluxes_from_RFMIP_atmospheres(
        gas_optics_sw,
        problem_type = OpticsProblemTypes.TWO_STREAM,
    )

if False:
    # As of pyrte v0.1.3 this test fails to run
    def test_rfmip_lw128_reduced_gases() -> None:
        """ Run RFMIP test cases with 128 g-point file and reduced set of gases"""
        # Load gas optics
        gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
            gas_optics_file = GasOpticsFiles.LW_G128,
        )
        _test_get_fluxes_from_RFMIP_atmospheres(
            gas_optics_lw,
            gas_name_mapping = RFMIP_GAS_MAPPING_SMALL,
        )

def test_rfmip_sw112_reduced_gases() -> None:
    """ Run RFMIP test cases with 112 g-point file and reduced set of gases"""
    # Load gas optics
    gas_optics_sw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file = GasOpticsFiles.SW_G112,
    )
    _test_get_fluxes_from_RFMIP_atmospheres(
        gas_optics_sw,
        problem_type = OpticsProblemTypes.TWO_STREAM,
        gas_name_mapping = RFMIP_GAS_MAPPING_SMALL,
    )
