from typing import Dict, Optional, Tuple
import netCDF4  # noqa
import pytest
import xarray as xr

from pyrte_rrtmgp.rrtmgp_data_files import GasOpticsFiles

from pyrte_rrtmgp.examples import RFMIP_FILES
from pyrte_rrtmgp.examples import load_example_file

from pyrte_rrtmgp.tests import RFMIP_GAS_MAPPING

from pyrte_rrtmgp.rrtmgp import GasOptics


def _load_problem_dataset(gas_mapping: Optional[Dict[str, str]] = None,
                          use_dask: bool = False) -> Tuple[xr.Dataset, Dict[str, str]]:

    atmosphere: xr.Dataset = load_example_file(RFMIP_FILES.ATMOSPHERE)
    if use_dask:
        atmosphere = atmosphere.chunk({"expt": 3})

    if gas_mapping is None:
        mapping: Dict[str, str] = RFMIP_GAS_MAPPING
    else:
        mapping = gas_mapping

    # Robert does not understand why it's necessary to comput gas optics in advance
    #   before checking the data validation function, but not doing so makes many tests
    #   fail because the mapping isn't clear...
    #
    gas_optics_lw = GasOptics( # type: ignore
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    gas_optics_lw.compute( #type: ignore
        atmosphere,
        gas_name_map=mapping,
    )

    return atmosphere, mapping

def test_validate_problem_dataset_success() -> None:
    """Test gas optics validate_input_data function."""

    ds, gas_mapping = _load_problem_dataset(None)
    gas_optics_lw = GasOptics( # type: ignore
        gas_optics_file=GasOpticsFiles.LW_G256
    )
    gas_optics_lw.validate_input_data(ds, gas_mapping) #type: ignore

def test_dask_validate_problem_dataset_success() -> None:
    """Test gas optics validate_input_data function with dask array."""

    ds, gas_mapping = _load_problem_dataset(None, use_dask=True)
    gas_optics_lw = GasOptics( # type: ignore
        gas_optics_file=GasOpticsFiles.LW_G256
    )
    gas_optics_lw.validate_input_data(ds, gas_mapping) #type: ignore

def test_raises_value_error_if_carbon_monoxide_missing() -> None:
    '''
    Load in LW_G256
    Set up input xarray with/without CO
    Compute gas optics
    '''

    # Load gas optics
    gas_optics_lw = GasOptics( # type: ignore
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    # Load atmosphere data
    atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)

    gas_mapping = RFMIP_GAS_MAPPING.copy()
    del gas_mapping["co"]

    # Compute gas optics for the atmosphere
    with pytest.raises(ValueError):
        gas_optics_lw.compute( #type: ignore
            atmosphere,
            gas_name_map=gas_mapping
        )

def test_raises_value_error_for_invalid_layer_pressure() -> None:
    ds, gas_mapping = _load_problem_dataset(None)
    ds["pres_layer"] = -ds["pres_layer"]
    gas_optics_lw = GasOptics( # type: ignore
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        gas_optics_lw.validate_input_data(ds, gas_mapping) #type: ignore


def test_dask_raises_value_error_for_invalid_layer_pressure() -> None:
    ds, gas_mapping = _load_problem_dataset(None, use_dask=True)
    ds["pres_layer"] = -ds["pres_layer"]
    gas_optics_lw = GasOptics( # type: ignore
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        gas_optics_lw.validate_input_data(ds, gas_mapping) #type: ignore


def test_raises_value_error_for_invalid_level_pressure() -> None:
    ds, gas_mapping = _load_problem_dataset(None)
    ds["pres_level"] = -ds["pres_level"]
    gas_optics_lw = GasOptics( # type: ignore
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        gas_optics_lw.validate_input_data(ds, gas_mapping) #type: ignore


def test_raises_value_error_for_invalid_layer_temperature() -> None:
    ds, gas_mapping = _load_problem_dataset(None)
    ds["temp_layer"] = -ds["temp_layer"]
    gas_optics_lw = GasOptics( # type: ignore
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        gas_optics_lw.validate_input_data(ds, gas_mapping) #type: ignore

def test_raises_value_error_for_invalid_level_temperature() -> None:
    ds, gas_mapping = _load_problem_dataset(None)
    ds["temp_level"] = -ds["temp_level"]
    gas_optics_lw = GasOptics( # type: ignore
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    with pytest.raises(ValueError):
        gas_optics_lw.validate_input_data(ds, gas_mapping) #type: ignore
