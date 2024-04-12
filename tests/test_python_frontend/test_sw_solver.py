import os

import numpy as np
import pytest
import xarray as xr
from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data
from pyrte_rrtmgp.kernels.rte import sw_solver_2stream
from pyrte_rrtmgp.utils import compute_mu0, compute_toa_flux, get_usecols

ERROR_TOLERANCE = 1e-4

rte_rrtmgp_dir = download_rrtmgp_data()
clear_sky_example_files = f"{rte_rrtmgp_dir}/examples/rfmip-clear-sky/inputs"

rfmip = xr.load_dataset(
    f"{clear_sky_example_files}/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
)
rfmip = rfmip.sel(expt=0)  # only one experiment
kdist = xr.load_dataset(f"{rte_rrtmgp_dir}/rrtmgp-gas-sw-g224.nc")

rsu = xr.load_dataset(
    "tests/test_python_frontend/rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
)
ref_flux_up = rsu.isel(expt=0)["rsu"].values

rsd = xr.load_dataset(
    "tests/test_python_frontend/rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
)
ref_flux_down = rsd.isel(expt=0)["rsd"].values


def test_sw_solver_noscat():
    gas_optics = kdist.gas_optics.load_atmosferic_conditions(rfmip)

    surface_albedo = rfmip["surface_albedo"].data
    total_solar_irradiance = rfmip["total_solar_irradiance"].data

    nlayer = len(rfmip["layer"])
    mu0 = compute_mu0(rfmip["solar_zenith_angle"].values, nlayer=nlayer)

    toa_flux = compute_toa_flux(total_solar_irradiance, gas_optics.solar_source)

    _, _, _, solver_flux_up, solver_flux_down, _ = sw_solver_2stream(
        kdist.gas_optics.top_at_1,
        gas_optics.tau,
        gas_optics.ssa,
        gas_optics.g,
        mu0,
        sfc_alb_dir=surface_albedo,
        sfc_alb_dif=surface_albedo,
        inc_flux_dir=toa_flux,
        inc_flux_dif=None,
        has_dif_bc=False,
        do_broadband=True,
    )

    # RTE will fail if passed solar zenith angles greater than 90 degree. We replace any with
    #   nighttime columns with a default solar zenith angle. We'll mask these out later, of
    #   course, but this gives us more work and so a better measure of timing.
    usecol = get_usecols(rfmip["solar_zenith_angle"].values)
    solver_flux_up = solver_flux_up * usecol[:, np.newaxis]
    solver_flux_down = solver_flux_down * usecol[:, np.newaxis]

    assert np.isclose(solver_flux_up, ref_flux_up, atol=ERROR_TOLERANCE).all()
    assert np.isclose(solver_flux_down, ref_flux_down, atol=ERROR_TOLERANCE).all()
