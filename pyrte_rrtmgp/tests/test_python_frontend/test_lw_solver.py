import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import RFMIPExampleFiles
from pyrte_rrtmgp.rrtmgp_gas_optics import GasOpticsFiles, load_gas_optics
from pyrte_rrtmgp.rte_solver import rte_solve
from pyrte_rrtmgp.utils import load_rrtmgp_file

ERROR_TOLERANCE = 1e-7


def test_lw_solver_noscat() -> None:
    # Load gas optics
    gas_optics_lw = load_gas_optics(gas_optics_file=GasOpticsFiles.LW_G256)

    # Load atmosphere data
    atmosphere = load_rrtmgp_file(RFMIPExampleFiles.RFMIP)
    atmosphere = atmosphere.sel(expt=0)  # only one experiment

    # Compute gas optics for the atmosphere
    gas_optics_lw.compute_gas_optics(atmosphere, problem_type="absorption")

    # Solve RTE
    fluxes = rte_solve(atmosphere, add_to_input=False)
    assert fluxes is not None

    # Load reference data
    rlu = load_rrtmgp_file(RFMIPExampleFiles.REFERENCE_RLU)
    rld = load_rrtmgp_file(RFMIPExampleFiles.REFERENCE_RLD)
    ref_flux_up = rlu.isel(expt=0)["rlu"]
    ref_flux_down = rld.isel(expt=0)["rld"]

    # Compare results with reference data
    assert np.isclose(fluxes["lw_flux_up"], ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(fluxes["lw_flux_down"], ref_flux_down, atol=ERROR_TOLERANCE).all()
