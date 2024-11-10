import os

import numpy as np
import pytest
import xarray as xr
from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.kernels.rrtmgp import (
    compute_planck_source,
    compute_tau_absorption,
    compute_tau_rayleigh,
    interpolation,
)
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data

from utils import convert_args_arrays

ERROR_TOLERANCE = 1e-4

rte_rrtmgp_dir = download_rrtmgp_data()
clear_sky_example_files = f"{rte_rrtmgp_dir}/examples/rfmip-clear-sky/inputs"

rfmip = xr.load_dataset(
    f"{clear_sky_example_files}/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
)
rfmip = rfmip.sel(expt=0)  # only one experiment
kdist = xr.load_dataset(f"{rte_rrtmgp_dir}/rrtmgp-gas-lw-g256.nc")
kdist_sw = xr.load_dataset(f"{rte_rrtmgp_dir}/rrtmgp-gas-sw-g224.nc")

rrtmgp_gas_optics = kdist.gas_optics.load_atmospheric_conditions(rfmip)
rrtmgp_gas_optics_sw = kdist_sw.gas_optics.load_atmospheric_conditions(rfmip)

# Prepare the arguments for the interpolation function
interpolation_args = [
    len(kdist["mixing_fraction"]),
    kdist.gas_optics.flavors_sets,
    kdist["press_ref"].values,
    kdist["temp_ref"].values,
    kdist["press_ref_trop"].values.item(),
    kdist.gas_optics.vmr_ref,
    rfmip["pres_layer"].values,
    rfmip["temp_layer"].values,
    kdist.gas_optics.column_gases,
]

expected_output = (
    kdist.gas_optics._interpolated.jtemp,
    kdist.gas_optics._interpolated.fmajor,
    kdist.gas_optics._interpolated.fminor,
    kdist.gas_optics._interpolated.col_mix,
    kdist.gas_optics._interpolated.tropo,
    kdist.gas_optics._interpolated.jeta,
    kdist.gas_optics._interpolated.jpress,
)


@pytest.mark.parametrize(
    "args, expected",
    [(i, expected_output) for i in convert_args_arrays(interpolation_args)],
)
def test_compute_interpoaltion(args, expected):
    result = interpolation(*args)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert r.shape == e.shape
        assert np.isclose(r, e, atol=ERROR_TOLERANCE).all()


# Prepare the arguments for the compute_planck_source function
planck_source_args = [
    rfmip["temp_layer"].data,
    rfmip["temp_level"].data,
    rfmip["surface_temperature"].data,
    kdist.gas_optics.top_at_1,
    kdist.gas_optics._interpolated.fmajor,
    kdist.gas_optics._interpolated.jeta,
    kdist.gas_optics._interpolated.tropo,
    kdist.gas_optics._interpolated.jtemp,
    kdist.gas_optics._interpolated.jpress,
    kdist["bnd_limits_gpt"].data.T,
    kdist["plank_fraction"].data.transpose(0, 2, 1, 3),
    kdist["temp_ref"].data.min(),
    kdist["temp_ref"].data.max(),
    kdist["totplnk"].data.T,
    kdist.gas_optics.gpoint_flavor,
]

expected_output = (
    rrtmgp_gas_optics.sfc_src,
    rrtmgp_gas_optics.lay_source,
    rrtmgp_gas_optics.lev_source,
    rrtmgp_gas_optics.sfc_src_jac,
)


@pytest.mark.parametrize(
    "args, expected",
    [(i, expected_output) for i in convert_args_arrays(planck_source_args)],
)
def test_compute_planck_source(args, expected):
    result = compute_planck_source(*args)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert r.shape == e.shape
        assert np.isclose(r, e, atol=ERROR_TOLERANCE).all()


# Prepare the arguments for the compute_tau_absorption function
minor_gases_lower = kdist.gas_optics.extract_names(kdist["minor_gases_lower"].data)
minor_gases_upper = kdist.gas_optics.extract_names(kdist["minor_gases_upper"].data)
idx_minor_lower = kdist.gas_optics.get_idx_minor(
    kdist.gas_optics.gas_names, minor_gases_lower
)
idx_minor_upper = kdist.gas_optics.get_idx_minor(
    kdist.gas_optics.gas_names, minor_gases_upper
)

scaling_gas_lower = kdist.gas_optics.extract_names(kdist["scaling_gas_lower"].data)
scaling_gas_upper = kdist.gas_optics.extract_names(kdist["scaling_gas_upper"].data)
idx_minor_scaling_lower = kdist.gas_optics.get_idx_minor(
    kdist.gas_optics.gas_names, scaling_gas_lower
)
idx_minor_scaling_upper = kdist.gas_optics.get_idx_minor(
    kdist.gas_optics.gas_names, scaling_gas_upper
)

tau_absorption_args = [
    kdist.gas_optics.idx_h2o,
    kdist.gas_optics.gpoint_flavor,
    kdist["bnd_limits_gpt"].values.T,
    kdist["kmajor"].values,
    kdist["kminor_lower"].values,
    kdist["kminor_upper"].values,
    kdist["minor_limits_gpt_lower"].values.T,
    kdist["minor_limits_gpt_upper"].values.T,
    kdist["minor_scales_with_density_lower"].values.astype(bool),
    kdist["minor_scales_with_density_upper"].values.astype(bool),
    kdist["scale_by_complement_lower"].values.astype(bool),
    kdist["scale_by_complement_upper"].values.astype(bool),
    idx_minor_lower,
    idx_minor_upper,
    idx_minor_scaling_lower,
    idx_minor_scaling_upper,
    kdist["kminor_start_lower"].values,
    kdist["kminor_start_upper"].values,
    kdist.gas_optics._interpolated.tropo,
    kdist.gas_optics._interpolated.col_mix,
    kdist.gas_optics._interpolated.fmajor,
    kdist.gas_optics._interpolated.fminor,
    rfmip["pres_layer"].values,
    rfmip["temp_layer"].values,
    kdist.gas_optics.column_gases,
    kdist.gas_optics._interpolated.jeta,
    kdist.gas_optics._interpolated.jtemp,
    kdist.gas_optics._interpolated.jpress,
]


@pytest.mark.parametrize(
    "args, expected",
    [(i, rrtmgp_gas_optics.tau) for i in convert_args_arrays(tau_absorption_args)],
)
def test_compute_tau_absorption(args, expected):
    result = compute_tau_absorption(*args)
    assert np.isclose(result, expected, atol=ERROR_TOLERANCE).all()


# Prepare the arguments for the compute_tau_rayleigh function
tau_rayleigh_args = [
    kdist_sw.gas_optics.gpoint_flavor,
    kdist_sw["bnd_limits_gpt"].values.T,
    np.stack([kdist_sw["rayl_lower"].values, kdist_sw["rayl_upper"].values], axis=-1),
    kdist_sw.gas_optics.idx_h2o,
    kdist_sw.gas_optics.column_gases[:, :, 0],
    kdist_sw.gas_optics.column_gases,
    kdist_sw.gas_optics._interpolated.fminor,
    kdist_sw.gas_optics._interpolated.jeta,
    kdist_sw.gas_optics._interpolated.tropo,
    kdist_sw.gas_optics._interpolated.jtemp,
]
