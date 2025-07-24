from typing import Dict, Optional
import netCDF4  # noqa
import pytest
import xarray as xr

from pyrte_rrtmgp.data_types import GasOpticsFiles
from pyrte_rrtmgp.data_types import OpticsProblemTypes

from pyrte_rrtmgp.examples import RFMIP_FILES
from pyrte_rrtmgp.examples import load_example_file

from pyrte_rrtmgp.tests import RFMIP_GAS_MAPPING

from pyrte_rrtmgp import rrtmgp_gas_optics


def _load_problem_dataset(gas_mapping: Optional[Dict[str, str]],
                          use_dask: bool = False) -> xr.Dataset:

    gas_optics_lw: xr.Dataset = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    atmosphere: xr.Dataset = load_example_file(RFMIP_FILES.ATMOSPHERE)
    atmosphere["pres_level"] = xr.ufuncs.maximum(
        atmosphere["pres_level"],
        gas_optics_lw.compute_gas_optics.press_min,
    )

    if use_dask:
        atmosphere = atmosphere.chunk({"expt": 3})

    if gas_mapping is None:
        gas_mapping = RFMIP_GAS_MAPPING

    gas_optics_lw.compute_gas_optics(
        atmosphere,
        problem_type=OpticsProblemTypes.ABSORPTION,
        gas_name_map=gas_mapping
    )

    return atmosphere, gas_mapping

def test_validate_problem_dataset_success() -> None:
    """Test gas optics validate_input_data function."""

    ds, gas_mapping = _load_problem_dataset(None)
    gas_optics_lw: xr.Dataset = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )
    _  = gas_optics_lw.compute_gas_optics.validate_input_data(ds, gas_mapping)

def test_dask_validate_problem_dataset_success() -> None:
    """Test gas optics validate_input_data function with dask array."""

    ds, gas_mapping = _load_problem_dataset(None, use_dask=True)
    gas_optics_lw: xr.Dataset = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )
    _  = gas_optics_lw.compute_gas_optics.validate_input_data(ds, gas_mapping)

def test_raises_value_error_if_carbon_monoxide_missing() -> None:
    '''
    Load in LW_G256
    Set up input xarray with/without CO
    Compute gas optics
    '''

    # Load gas optics
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    # Load atmosphere data
    atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)

    gas_mapping = RFMIP_GAS_MAPPING.copy()
    del gas_mapping["co"]

    # Compute gas optics for the atmosphere
    with pytest.raises(ValueError):
        gas_optics_lw.compute_gas_optics(
            atmosphere,
            problem_type=OpticsProblemTypes.ABSORPTION,
            gas_name_map=gas_mapping
        )

def test_raises_value_error_for_invalid_layer_pressure() -> None:
    ds, gas_mapping = _load_problem_dataset(None)
    ds["pres_layer"] = -ds["pres_layer"]
    gas_optics_lw: xr.Dataset = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        _  = gas_optics_lw.compute_gas_optics.validate_input_data(ds, gas_mapping)


def test_dask_raises_value_error_for_invalid_layer_pressure() -> None:
    ds, gas_mapping = _load_problem_dataset(None, use_dask=True)
    ds["pres_layer"] = -ds["pres_layer"]
    gas_optics_lw: xr.Dataset = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        _  = gas_optics_lw.compute_gas_optics.validate_input_data(ds, gas_mapping)


def test_raises_value_error_for_invalid_level_pressure() -> None:
    ds, gas_mapping = _load_problem_dataset(None)
    ds["pres_level"] = -ds["pres_level"]
    gas_optics_lw: xr.Dataset = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        _  = gas_optics_lw.compute_gas_optics.validate_input_data(ds, gas_mapping)


def test_raises_value_error_for_invalid_layer_temperature() -> None:
    ds, gas_mapping = _load_problem_dataset(None)
    ds["temp_layer"] = -ds["temp_layer"]
    gas_optics_lw: xr.Dataset = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        _  = gas_optics_lw.compute_gas_optics.validate_input_data(ds, gas_mapping)

def test_raises_value_error_for_invalid_level_temperature() -> None:
    ds, gas_mapping = _load_problem_dataset(None)
    ds["temp_level"] = -ds["temp_level"]
    gas_optics_lw: xr.Dataset = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        _  = gas_optics_lw.compute_gas_optics.validate_input_data(ds, gas_mapping)
