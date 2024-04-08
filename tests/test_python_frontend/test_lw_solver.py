import numpy as np
import pytest
import xarray as xr
from pyrte_rrtmgp.gas_optics import GasOptics
from pyrte_rrtmgp.rte import lw_solver_noscat

ERROR_TOLERANCE = 1e-4

rte_rrtmgp_dir = "rrtmgp-data"
clear_sky_example_files = f"{rte_rrtmgp_dir}/examples/rfmip-clear-sky/inputs"

rfmip = xr.load_dataset(
    f"{clear_sky_example_files}/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
)
rfmip = rfmip.sel(expt=0)  # only one experiment
kdist = xr.load_dataset(f"{rte_rrtmgp_dir}/rrtmgp-gas-lw-g256.nc")

rlu = xr.load_dataset(
    "tests/test_python_frontend/rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
)
ref_flux_up = rlu.isel(expt=0)["rlu"].values

rld = xr.load_dataset(
    "tests/test_python_frontend/rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
)
ref_flux_down = rld.isel(expt=0)["rld"].values


def test_lw_solver_noscat():
    min_index = np.argmin(rfmip["pres_level"].values)
    rfmip["pres_level"][:, min_index] = 1.0051835744630002

    gas_optics = GasOptics(kdist, rfmip)
    tau, _, _, layer_src, level_src, sfc_src, sfc_src_jac = gas_optics.gas_optics()

    sfc_emis = rfmip["surface_emissivity"].values
    sfc_emis = np.stack([sfc_emis] * tau.shape[2]).T

    _, solver_flux_up, solver_flux_down, _, _ = lw_solver_noscat(
        tau=tau,
        lay_source=layer_src,
        lev_source=level_src,
        sfc_emis=sfc_emis,
        sfc_src=sfc_src,
        sfc_src_jac=sfc_src_jac,
    )

    assert np.isclose(solver_flux_up, ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(solver_flux_down, ref_flux_down, atol=ERROR_TOLERANCE).all()
