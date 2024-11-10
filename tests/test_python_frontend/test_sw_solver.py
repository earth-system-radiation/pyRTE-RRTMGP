import os

import numpy as np
import pytest
import xarray as xr
from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.kernels.rte import sw_solver_2stream
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data
from pyrte_rrtmgp.utils import compute_mu0, compute_toa_flux, get_usecols

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
kdist = xr.load_dataset(os.path.join(rte_rrtmgp_dir, "rrtmgp-gas-sw-g224.nc"))

rsu = xr.load_dataset(
    os.path.join(ref_dir, "rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"),
    decode_cf=False,
)
ref_flux_up = rsu.isel(expt=0)["rsu"].values

rsd = xr.load_dataset(
    os.path.join(ref_dir, "rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"),
    decode_cf=False,
)
ref_flux_down = rsd.isel(expt=0)["rsd"].values


def test_sw_solver_noscat():
    sw_problem = kdist.gas_optics.load_atmospheric_conditions(rfmip)

    sw_problem.sfc_alb_dir = rfmip["surface_albedo"].data
    sw_problem.total_solar_irradiance = rfmip["total_solar_irradiance"].data
    sw_problem.solar_zenith_angle = rfmip["solar_zenith_angle"].values

    solver_flux_up, solver_flux_down = sw_problem.solve()

    assert np.isclose(solver_flux_up, ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(solver_flux_down, ref_flux_down, atol=ERROR_TOLERANCE).all()
