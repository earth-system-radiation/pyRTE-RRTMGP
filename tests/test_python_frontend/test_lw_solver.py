import os

import numpy as np
import xarray as xr
from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data
from pyrte_rrtmgp.kernels.rte import lw_solver_noscat

ERROR_TOLERANCE = 1e-4

rte_rrtmgp_dir = download_rrtmgp_data()
clear_sky_example_files = f"{rte_rrtmgp_dir}/examples/rfmip-clear-sky/inputs"
clear_sky_ref_files = f"{rte_rrtmgp_dir}/examples/rfmip-clear-sky/reference"

rfmip = xr.load_dataset(
    f"{clear_sky_example_files}/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
)
rfmip = rfmip.sel(expt=0)  # only one experiment
kdist = xr.load_dataset(f"{rte_rrtmgp_dir}/rrtmgp-gas-lw-g256.nc")

rlu = xr.load_dataset(
    f"{clear_sky_ref_files}/rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
)
ref_flux_up = rlu.isel(expt=0)["rlu"].values

rld = xr.load_dataset(
    f"{clear_sky_ref_files}/rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
)
ref_flux_down = rld.isel(expt=0)["rld"].values


def test_lw_solver_noscat():
    rrtmgp_gas_optics = kdist.gas_optics.load_atmosferic_conditions(rfmip)

    _, solver_flux_up, solver_flux_down, _, _ = lw_solver_noscat(
        tau=rrtmgp_gas_optics.tau,
        lay_source=rrtmgp_gas_optics.lay_src,
        lev_source=rrtmgp_gas_optics.lev_src,
        sfc_emis=rfmip["surface_emissivity"].data,
        sfc_src=rrtmgp_gas_optics.sfc_src,
        sfc_src_jac=rrtmgp_gas_optics.sfc_src_jac,
    )

    assert np.isclose(solver_flux_up, ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(solver_flux_down, ref_flux_down, atol=ERROR_TOLERANCE).all()
