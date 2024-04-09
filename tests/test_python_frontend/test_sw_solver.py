import os
import sys

import numpy as np
import xarray as xr
from pyrte_rrtmgp.rrtmgp_gas_optics import GasOptics
from pyrte_rrtmgp.kernels.rte import sw_solver_2stream
from pyrte_rrtmgp.utils import compute_mu0, get_usecols

ERROR_TOLERANCE = 1e-4

rte_rrtmgp_dir = os.environ.get("RRTMGP_DATA", "rrtmgp-data")
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


def test_lw_solver_noscat():
    min_index = np.argmin(rfmip["pres_level"].data)
    min_press = kdist["press_ref"].min().item() + sys.float_info.epsilon
    rfmip["pres_level"][:, min_index] = min_press

    gas_optics = GasOptics(kdist, rfmip)
    gas_optics.source_is_internal
    tau, g, ssa, toa_flux = gas_optics.gas_optics()

    pres_layers = rfmip["pres_layer"]["layer"]
    top_at_1 = (pres_layers[0] < pres_layers[-1]).values.item()

    # Expand the surface albedo to ngpt
    ngpt = len(kdist["gpt"])
    surface_albedo = rfmip["surface_albedo"].values
    surface_albedo = np.stack([surface_albedo] * ngpt)
    sfc_alb_dir = surface_albedo.T.copy()
    sfc_alb_dif = surface_albedo.T.copy()

    nlayer = len(rfmip["layer"])
    mu0 = compute_mu0(rfmip["solar_zenith_angle"].values, nlayer=nlayer)

    total_solar_irradiance = rfmip["total_solar_irradiance"].values
    toa_flux = np.stack([toa_flux] * mu0.shape[0])
    def_tsi = toa_flux.sum(axis=1)
    toa_flux = (toa_flux.T * (total_solar_irradiance / def_tsi)).T

    _, _, _, solver_flux_up, solver_flux_down, _ = sw_solver_2stream(
        top_at_1,
        tau,
        ssa,
        g,
        mu0,
        sfc_alb_dir,
        sfc_alb_dif,
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
