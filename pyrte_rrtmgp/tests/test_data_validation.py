from typing import Dict, Optional
import netCDF4  # noqa
import pytest
import xarray as xr

from pyrte_rrtmgp.data_types import GasOpticsFiles
from pyrte_rrtmgp.data_types import OpticsProblemTypes
from pyrte_rrtmgp.data_types import RFMIPExampleFiles

from pyrte_rrtmgp.utils import load_rrtmgp_file
from pyrte_rrtmgp import rrtmgp_gas_optics

from pyrte_rrtmgp.data_validation import validate_problem_dataset


def _load_problem_dataset(gas_mapping: Optional[Dict[str, str]],
                          use_dask: bool = False) -> xr.Dataset:

    gas_optics_lw: xr.Dataset = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    atmosphere: xr.Dataset = load_rrtmgp_file(RFMIPExampleFiles.RFMIP)

    if use_dask:
        atmosphere = atmosphere.chunk({"expt": 3})

    if gas_mapping is None:
        gas_mapping = {
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

    gas_optics_lw.compute_gas_optics(
        atmosphere,
        problem_type=OpticsProblemTypes.ABSORPTION,
        gas_name_map=gas_mapping
    )

    return atmosphere


def test_validate_problem_dataset_success() -> None:
    """Test validate_problem_dataset function."""

    ds: xr.Dataset = _load_problem_dataset(None)
    assert validate_problem_dataset(ds)


def test_raises_value_error_for_invalid_pressure() -> None:
    ds: xr.Dataset = _load_problem_dataset(None)
    ds["pres_layer"] = ds["pres_layer"] - 100.0

    with pytest.raises(ValueError):
        assert validate_problem_dataset(ds)


def test_dask_validate_problem_dataset_success() -> None:
    """Test validate_problem_dataset function."""

    ds: xr.Dataset = _load_problem_dataset(None, use_dask=True)
    assert validate_problem_dataset(ds)


def test_dask_raises_value_error_for_invalid_pressure() -> None:
    ds: xr.Dataset = _load_problem_dataset(None, use_dask=True)
    ds["pres_layer"] = ds["pres_layer"] - 100.0

    with pytest.raises(ValueError):
        assert validate_problem_dataset(ds)
