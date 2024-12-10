import os

import numpy as np
import xarray as xr

from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data
from pyrte_rrtmgp.rrtmgp_gas_optics import GasOpticsFiles, load_gas_optics
from pyrte_rrtmgp.rte_solver import RTESolver

ERROR_TOLERANCE = 1e-7

rte_rrtmgp_dir = download_rrtmgp_data()
rfmip_dir = os.path.join(rte_rrtmgp_dir, "examples", "rfmip-clear-sky")
input_dir = os.path.join(rfmip_dir, "inputs")
ref_dir = os.path.join(rfmip_dir, "reference")

atmosphere = xr.load_dataset(
    os.path.join(
        input_dir, "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
    )
)
atmosphere = atmosphere.sel(expt=0)  # only one experiment

rlu = xr.load_dataset(
    os.path.join(ref_dir, "rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"),
    decode_cf=False,
)
ref_flux_up = rlu.isel(expt=0)["rlu"]

rld = xr.load_dataset(
    os.path.join(ref_dir, "rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"),
    decode_cf=False,
)
ref_flux_down = rld.isel(expt=0)["rld"]


def test_lw_solver_noscat():
    # Load gas optics with the new API
    gas_optics_lw = load_gas_optics(gas_optics_file=GasOpticsFiles.LW_G256)

    # Compute gas optics for the atmosphere
    gas_optics_lw.gas_optics.compute(atmosphere, problem_type="absorption")

    # Solve RTE with the new API
    solver = RTESolver()
    fluxes = solver.solve(atmosphere, add_to_input=False)

    # Compare results with reference data
    assert np.isclose(
        fluxes["lw_flux_up_broadband"], ref_flux_up, atol=ERROR_TOLERANCE
    ).all()
    assert np.isclose(
        fluxes["lw_flux_down_broadband"], ref_flux_down, atol=ERROR_TOLERANCE
    ).all()
