import os

import numpy as np
import xarray as xr

from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.rrtmgp_gas_optics import GasOpticsFiles, load_gas_optics
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data
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
atmosphere = atmosphere.sel(expt=0)

rsu = xr.load_dataset(
    os.path.join(ref_dir, "rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"),
    decode_cf=False,
)
ref_flux_up = rsu.isel(expt=0)["rsu"]

rsd = xr.load_dataset(
    os.path.join(ref_dir, "rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"),
    decode_cf=False,
)
ref_flux_down = rsd.isel(expt=0)["rsd"]


def test_sw_solver_noscat():
    # Load gas optics with new API
    gas_optics_sw = load_gas_optics(gas_optics_file=GasOpticsFiles.SW_G224)
    
    # Load and compute gas optics with atmosphere data
    gas_optics_sw.gas_optics.compute(atmosphere, problem_type="two-stream")
    
    # Solve using new rte_solve function
    solver = RTESolver()
    fluxes = solver.solve(atmosphere, add_to_input=False)
    
    # Compare results
    assert np.isclose(fluxes["sw_flux_up"], ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(fluxes["sw_flux_down"], ref_flux_down, atol=ERROR_TOLERANCE).all()
