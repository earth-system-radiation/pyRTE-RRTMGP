import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import OpticsProblemTypes
from pyrte_rrtmgp.rrtmgp_gas_optics import GasOpticsFiles, load_gas_optics
from pyrte_rrtmgp.rte_solver import rte_solve
from pyrte_rrtmgp.tests import DEFAULT_GAS_MAPPING
from pyrte_rrtmgp.tests import ERROR_TOLERANCE

from pyrte_rrtmgp.examples import RFMIP_FILES
from pyrte_rrtmgp.examples import load_example_file


def test_rfmip_clr_sky_sw() -> None:
    # Load gas optics
    gas_optics_sw = load_gas_optics(gas_optics_file=GasOpticsFiles.SW_G224)

    # Load atmosphere data
    atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)
    atmosphere = atmosphere.sel(expt=0)  # only one experiment
    atmosphere["pres_level"] = xr.ufuncs.maximum(
        gas_optics_sw.compute_gas_optics.press_min,
        atmosphere["pres_level"],
    )

    # Compute gas optics for the atmosphere
    gas_optics_sw.compute_gas_optics(
        atmosphere,
        problem_type=OpticsProblemTypes.TWO_STREAM,
        gas_name_map=DEFAULT_GAS_MAPPING
    )

    # Solve RTE
    fluxes = rte_solve(atmosphere, add_to_input=False)
    assert fluxes is not None

    # Load reference data
    rsu = load_example_file(RFMIP_FILES.REFERENCE_RSU)
    rsd = load_example_file(RFMIP_FILES.REFERENCE_RSD)
    ref_flux_up = rsu.isel(expt=0)["rsu"]
    ref_flux_down = rsd.isel(expt=0)["rsd"]

    # Compare results with reference data
    assert np.isclose(fluxes["sw_flux_up"],
                      ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(fluxes["sw_flux_down"],
                      ref_flux_down, atol=ERROR_TOLERANCE).all()


def test_rfmip_clr_sky_sw_dask() -> None:
    # Load gas optics
    gas_optics_sw = load_gas_optics(gas_optics_file=GasOpticsFiles.SW_G224)

    # Load atmosphere data
    atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)
    atmosphere = atmosphere.chunk({"expt": 3})
    atmosphere["pres_level"] = xr.ufuncs.maximum(
        gas_optics_sw.compute_gas_optics.press_min,
        atmosphere["pres_level"],
    )

    # Compute gas optics for the atmosphere
    gas_optics_sw.compute_gas_optics(
        atmosphere,
        problem_type=OpticsProblemTypes.TWO_STREAM,
        gas_name_map=DEFAULT_GAS_MAPPING
    )

    # Solve RTE
    fluxes = rte_solve(atmosphere, add_to_input=False)
    assert fluxes is not None

    # Load reference data
    rsu = load_example_file(RFMIP_FILES.REFERENCE_RSU)
    rsd = load_example_file(RFMIP_FILES.REFERENCE_RSD)
    ref_flux_up = rsu["rsu"]
    ref_flux_down = rsd["rsd"]

    # Compare results with reference data
    assert np.isclose(fluxes["sw_flux_up"].transpose("expt", "site", "level"),
                      ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(fluxes["sw_flux_down"].transpose("expt", "site", "level"),
                      ref_flux_down, atol=ERROR_TOLERANCE).all()
