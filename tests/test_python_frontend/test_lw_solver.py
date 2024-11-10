import os

import numpy as np
import xarray as xr
from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.kernels.rte import lw_solver_noscat
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data

ERROR_TOLERANCE = 1e-7

rte_rrtmgp_dir = download_rrtmgp_data()
rfmip_dir = os.path.join(rte_rrtmgp_dir, "examples", "rfmip-clear-sky")
input_dir = os.path.join(rfmip_dir, "inputs")
ref_dir = os.path.join(rfmip_dir, "reference")

rfmip = xr.load_dataset(
    os.path.join(
        input_dir, "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
    )
)
rfmip = rfmip.sel(expt=0)  # only one experiment
kdist = xr.load_dataset(os.path.join(rte_rrtmgp_dir, "rrtmgp-gas-lw-g256.nc"))

rlu = xr.load_dataset(
    os.path.join(ref_dir, "rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"),
    decode_cf=False,
)
ref_flux_up = rlu.isel(expt=0)["rlu"].values

rld = xr.load_dataset(
    os.path.join(ref_dir, "rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"),
    decode_cf=False,
)
ref_flux_down = rld.isel(expt=0)["rld"].values


def test_lw_solver_noscat():
    lw_problem = kdist.gas_optics.load_atmospheric_conditions(rfmip)

    lw_problem.sfc_emis = rfmip["surface_emissivity"].data

    solver_flux_up, solver_flux_down = lw_problem.rte_solve()

    assert np.isclose(solver_flux_up, ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(solver_flux_down, ref_flux_down, atol=ERROR_TOLERANCE).all()
