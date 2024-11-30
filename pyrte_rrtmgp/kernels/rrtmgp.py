from typing import Tuple

import numpy as np
import numpy.typing as npt

from pyrte_rrtmgp.pyrte_rrtmgp import (
    rrtmgp_compute_Planck_source,
    rrtmgp_compute_tau_absorption,
    rrtmgp_compute_tau_rayleigh,
    rrtmgp_interpolation,
)
from pyrte_rrtmgp.utils import convert_xarray_args


@convert_xarray_args
def interpolation(
    neta: int,
    flavor: npt.NDArray,
    press_ref: npt.NDArray,
    temp_ref: npt.NDArray,
    press_ref_trop: float,
    vmr_ref: npt.NDArray,
    play: npt.NDArray,
    tlay: npt.NDArray,
    col_gas: npt.NDArray,
) -> Tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    """Interpolate the RRTMGP coefficients.

    Args:
        neta (int): Number of mixing_fraction.
        flavor (np.ndarray): Index into vmr_ref of major gases for each flavor.
        press_ref (np.ndarray): Reference pressure grid.
        temp_ref (np.ndarray): Reference temperature grid.
        press_ref_trop (float): Reference pressure at the tropopause.
        vmr_ref (np.ndarray): Reference volume mixing ratio.
        play (np.ndarray): Pressure layers.
        tlay (np.ndarray): Temperature layers.
        col_gas (np.ndarray): Gas concentrations.

    Returns:
        Tuple: A tuple containing the following arrays:
            - jtemp (np.ndarray): Temperature interpolation index.
            - fmajor (np.ndarray): Major gas interpolation fraction.
            - fminor (np.ndarray): Minor gas interpolation fraction.
            - col_mix (np.ndarray): Mixing fractions.
            - tropo (np.ndarray): Use lower (or upper) atmosphere tables.
            - jeta (np.ndarray): Index for binary species interpolation.
            - jpress (np.ndarray): Pressure interpolation index.
    """
    npres = press_ref.shape[0]
    ntemp = temp_ref.shape[0]
    ncol, nlay, ngas = col_gas.shape
    ngas = ngas - 1  # Fortran uses index 0 here
    nflav = flavor.shape[1]

    press_ref_log = np.log(press_ref)
    press_ref_log_delta = (press_ref_log.min() - press_ref_log.max()) / (
        len(press_ref_log) - 1
    )
    press_ref_trop_log = np.log(press_ref_trop)

    temp_ref_min = temp_ref.min()
    temp_ref_delta = (temp_ref.max() - temp_ref.min()) / (len(temp_ref) - 1)

    # outputs
    jtemp = np.ndarray([ncol, nlay], dtype=np.int32, order="F")
    fmajor = np.ndarray([2, 2, 2, ncol, nlay, nflav], dtype=np.float64, order="F")
    fminor = np.ndarray([2, 2, ncol, nlay, nflav], dtype=np.float64, order="F")
    col_mix = np.ndarray([2, ncol, nlay, nflav], dtype=np.float64, order="F")
    tropo = np.ndarray([ncol, nlay], dtype=np.int32, order="F")
    jeta = np.ndarray([2, ncol, nlay, nflav], dtype=np.int32, order="F")
    jpress = np.ndarray([ncol, nlay], dtype=np.int32, order="F")

    args = [
        ncol,
        nlay,
        ngas,
        nflav,
        neta,
        npres,
        ntemp,
        np.asfortranarray(flavor),
        np.asfortranarray(press_ref_log),
        np.asfortranarray(temp_ref),
        press_ref_log_delta,
        temp_ref_min,
        temp_ref_delta,
        press_ref_trop_log,
        np.asfortranarray(vmr_ref),
        np.asfortranarray(play),
        np.asfortranarray(tlay),
        np.asfortranarray(col_gas),
        jtemp,
        fmajor,
        fminor,
        col_mix,
        tropo,
        jeta,
        jpress,
    ]

    rrtmgp_interpolation(*args)

    tropo = tropo != 0  # Convert to boolean
    return jtemp, fmajor, fminor, col_mix, tropo, jeta, jpress


@convert_xarray_args
def compute_planck_source(
    tlay,
    tlev,
    tsfc,
    top_at_1,
    fmajor,
    jeta,
    tropo,
    jtemp,
    jpress,
    band_lims_gpt,
    pfracin,
    temp_ref_min,
    temp_ref_max,
    totplnk,
    gpoint_flavor,
):
    """Compute the Planck source function for a radiative transfer calculation.

    Args:
        tlay (numpy.ndarray): Temperature at layer centers (K), shape (ncol, nlay).
        tlev (numpy.ndarray): Temperature at layer interfaces (K), shape (ncol, nlay+1).
        tsfc (numpy.ndarray): Surface temperature, shape (ncol,).
        top_at_1 (bool): Flag indicating if the top layer is at index 0.
        sfc_lay (int): Index of the surface layer.
        fmajor (numpy.ndarray): Interpolation weights for major gases, shape (2, 2, 2, ncol, nlay, nflav).
        jeta (numpy.ndarray): Interpolation indexes in eta, shape (2, ncol, nlay, nflav).
        tropo (numpy.ndarray): Use upper- or lower-atmospheric tables, shape (ncol, nlay).
        jtemp (numpy.ndarray): Interpolation indexes in temperature, shape (ncol, nlay).
        jpress (numpy.ndarray): Interpolation indexes in pressure, shape (ncol, nlay).
        band_lims_gpt (numpy.ndarray): Start and end g-point for each band, shape (2, nbnd).
        pfracin (numpy.ndarray): Fraction of the Planck function in each g-point, shape (ntemp, neta, npres+1, ngpt).
        temp_ref_min (float): Minimum reference temperature for Planck function interpolation.
        totplnk (numpy.ndarray): Total Planck function by band at each temperature, shape (nPlanckTemp, nbnd).
        gpoint_flavor (numpy.ndarray): Major gas flavor (pair) by upper/lower, g-point, shape (2, ngpt).

    Returns:
        sfc_src (numpy.ndarray): Planck emission from the surface, shape (ncol, ngpt).
        lay_src (numpy.ndarray): Planck emission from layer centers, shape (ncol, nlay, ngpt).
        lev_src (numpy.ndarray): Planck emission from layer boundaries, shape (ncol, nlay+1, ngpt).
        sfc_source_Jac (numpy.ndarray): Jacobian (derivative) of the surface Planck source with respect to surface temperature, shape (ncol, ngpt).
    """

    _, ncol, nlay, nflav = jeta.shape
    nPlanckTemp, nbnd = totplnk.shape
    ntemp, neta, npres_e, ngpt = pfracin.shape
    npres = npres_e - 1

    sfc_lay = nlay if top_at_1 else 1

    gpoint_bands = []

    totplnk_delta = (temp_ref_max - temp_ref_min) / (nPlanckTemp - 1)

    # outputs
    sfc_src = np.ndarray((ncol, ngpt), dtype=np.float64, order="F")
    lay_src = np.ndarray((ncol, nlay, ngpt), dtype=np.float64, order="F")
    lev_src = np.ndarray((ncol, nlay + 1, ngpt), dtype=np.float64, order="F")
    sfc_src_jac = np.ndarray((ncol, ngpt), dtype=np.float64, order="F")

    args = [
        ncol,
        nlay,
        nbnd,
        ngpt,
        nflav,
        neta,
        npres,
        ntemp,
        nPlanckTemp,
        np.asfortranarray(tlay),
        np.asfortranarray(tlev),
        np.asfortranarray(tsfc),
        sfc_lay,
        np.asfortranarray(fmajor),
        np.asfortranarray(jeta),
        np.asfortranarray(tropo),
        np.asfortranarray(jtemp),
        np.asfortranarray(jpress),
        gpoint_bands,
        np.asfortranarray(band_lims_gpt),
        np.asfortranarray(pfracin),
        temp_ref_min,
        totplnk_delta,
        np.asfortranarray(totplnk),
        np.asfortranarray(gpoint_flavor),
        sfc_src,
        lay_src,
        lev_src,
        sfc_src_jac,
    ]

    rrtmgp_compute_Planck_source(*args)

    return sfc_src, lay_src, lev_src, sfc_src_jac


@convert_xarray_args
def compute_tau_absorption(
    idx_h2o,
    gpoint_flavor,
    band_lims_gpt,
    kmajor,
    kminor_lower,
    kminor_upper,
    minor_limits_gpt_lower,
    minor_limits_gpt_upper,
    minor_scales_with_density_lower,
    minor_scales_with_density_upper,
    scale_by_complement_lower,
    scale_by_complement_upper,
    idx_minor_lower,
    idx_minor_upper,
    idx_minor_scaling_lower,
    idx_minor_scaling_upper,
    kminor_start_lower,
    kminor_start_upper,
    tropo,
    col_mix,
    fmajor,
    fminor,
    play,
    tlay,
    col_gas,
    jeta,
    jtemp,
    jpress,
):
    """Compute the absorption optical depth for a set of atmospheric profiles.

    Args:
        idx_h2o (int): Index of the water vapor gas species.
        gpoint_flavor (np.ndarray): Spectral g-point flavor indices.
        band_lims_gpt (np.ndarray): Spectral band limits in g-point space.
        kmajor (np.ndarray): Major gas absorption coefficients.
        kminor_lower (np.ndarray): Minor gas absorption coefficients for the lower atmosphere.
        kminor_upper (np.ndarray): Minor gas absorption coefficients for the upper atmosphere.
        minor_limits_gpt_lower (np.ndarray): Spectral g-point limits for minor contributors in the lower atmosphere.
        minor_limits_gpt_upper (np.ndarray): Spectral g-point limits for minor contributors in the upper atmosphere.
        minor_scales_with_density_lower (np.ndarray): Flags indicating if minor contributors in the lower atmosphere scale with density.
        minor_scales_with_density_upper (np.ndarray): Flags indicating if minor contributors in the upper atmosphere scale with density.
        scale_by_complement_lower (np.ndarray): Flags indicating if minor contributors in the lower atmosphere should be scaled by the complement.
        scale_by_complement_upper (np.ndarray): Flags indicating if minor contributors in the upper atmosphere should be scaled by the complement.
        idx_minor_lower (np.ndarray): Indices of minor contributors in the lower atmosphere.
        idx_minor_upper (np.ndarray): Indices of minor contributors in the upper atmosphere.
        idx_minor_scaling_lower (np.ndarray): Indices of minor contributors in the lower atmosphere that require scaling.
        idx_minor_scaling_upper (np.ndarray): Indices of minor contributors in the upper atmosphere that require scaling.
        kminor_start_lower (np.ndarray): Starting indices of minor absorption coefficients in the lower atmosphere.
        kminor_start_upper (np.ndarray): Starting indices of minor absorption coefficients in the upper atmosphere.
        tropo (np.ndarray): Flags indicating if a layer is in the troposphere.
        col_mix (np.ndarray): Column-dependent gas mixing ratios.
        fmajor (np.ndarray): Major gas absorption coefficient scaling factors.
        fminor (np.ndarray): Minor gas absorption coefficient scaling factors.
        play (np.ndarray): Pressure in each layer.
        tlay (np.ndarray): Temperature in each layer.
        col_gas (np.ndarray): Column-dependent gas concentrations.
        jeta (np.ndarray): Indices of temperature/pressure levels.
        jtemp (np.ndarray): Indices of temperature levels.
        jpress (np.ndarray): Indices of pressure levels.

    Returns:
        np.ndarray): tau Absorption optical depth.
    """

    ntemp, neta, npres_e, ngpt = kmajor.shape
    npres = npres_e - 1
    nbnd = band_lims_gpt.shape[1]
    _, ncol, nlay, nflav = jeta.shape
    ngas = col_gas.shape[2] - 1
    nminorlower = minor_scales_with_density_lower.shape[0]
    nminorupper = minor_scales_with_density_upper.shape[0]
    nminorklower = kminor_lower.shape[2]
    nminorkupper = kminor_upper.shape[2]

    # outputs
    tau = np.zeros((ncol, nlay, ngpt), dtype=np.float64, order="F")

    args = [
        ncol,
        nlay,
        nbnd,
        ngpt,
        ngas,
        nflav,
        neta,
        npres,
        ntemp,
        nminorlower,
        nminorklower,
        nminorupper,
        nminorkupper,
        idx_h2o,
        np.asfortranarray(gpoint_flavor),
        np.asfortranarray(band_lims_gpt),
        np.asfortranarray(kmajor),
        np.asfortranarray(kminor_lower),
        np.asfortranarray(kminor_upper),
        np.asfortranarray(minor_limits_gpt_lower),
        np.asfortranarray(minor_limits_gpt_upper),
        np.asfortranarray(minor_scales_with_density_lower),
        np.asfortranarray(minor_scales_with_density_upper),
        np.asfortranarray(scale_by_complement_lower),
        np.asfortranarray(scale_by_complement_upper),
        np.asfortranarray(idx_minor_lower),
        np.asfortranarray(idx_minor_upper),
        np.asfortranarray(idx_minor_scaling_lower),
        np.asfortranarray(idx_minor_scaling_upper),
        np.asfortranarray(kminor_start_lower),
        np.asfortranarray(kminor_start_upper),
        np.asfortranarray(tropo),
        np.asfortranarray(col_mix),
        np.asfortranarray(fmajor),
        np.asfortranarray(fminor),
        np.asfortranarray(play),
        np.asfortranarray(tlay),
        np.asfortranarray(col_gas),
        np.asfortranarray(jeta),
        np.asfortranarray(jtemp),
        np.asfortranarray(jpress),
        tau,
    ]

    rrtmgp_compute_tau_absorption(*args)

    return tau


@convert_xarray_args
def compute_tau_rayleigh(
    gpoint_flavor,
    band_lims_gpt,
    krayl,
    idx_h2o,
    col_dry,
    col_gas,
    fminor,
    jeta,
    tropo,
    jtemp,
):
    """Compute Rayleigh optical depth.

    Args:
        gpoint_flavor (numpy.ndarray): Major gas flavor (pair) by upper/lower, g-point (shape: (2, ngpt)).
        band_lims_gpt (numpy.ndarray): Start and end g-point for each band (shape: (2, nbnd)).
        krayl (numpy.ndarray): Rayleigh scattering coefficients (shape: (ntemp, neta, ngpt, 2)).
        idx_h2o (int): Index of water vapor in col_gas.
        col_dry (numpy.ndarray): Column amount of dry air (shape: (ncol, nlay)).
        col_gas (numpy.ndarray): Input column gas amount (molecules/cm^2) (shape: (ncol, nlay, 0:ngas)).
        fminor (numpy.ndarray): Interpolation weights for major gases - computed in interpolation() (shape: (2, 2, ncol, nlay, nflav)).
        jeta (numpy.ndarray): Interpolation indexes in eta - computed in interpolation() (shape: (2, ncol, nlay, nflav)).
        tropo (numpy.ndarray): Use upper- or lower-atmospheric tables? (shape: (ncol, nlay)).
        jtemp (numpy.ndarray): Interpolation indexes in temperature - computed in interpolation() (shape: (ncol, nlay)).

    Returns:
        numpy.ndarray: Rayleigh optical depth (shape: (ncol, nlay, ngpt)).
    """

    ncol, nlay, ngas = col_gas.shape
    ntemp, neta, ngpt, _ = krayl.shape
    nflav = jeta.shape[3]
    nbnd = band_lims_gpt.shape[1]

    # outputs
    tau_rayleigh = np.ndarray((ncol, nlay, ngpt), dtype=np.float64, order="F")

    args = [
        ncol,
        nlay,
        nbnd,
        ngpt,
        ngas,
        nflav,
        neta,
        0,  # not used in fortran
        ntemp,
        np.asfortranarray(gpoint_flavor),
        np.asfortranarray(band_lims_gpt),
        np.asfortranarray(krayl),
        idx_h2o,
        np.asfortranarray(col_dry),
        np.asfortranarray(col_gas),
        np.asfortranarray(fminor),
        np.asfortranarray(jeta),
        np.asfortranarray(tropo),
        np.asfortranarray(jtemp),
        tau_rayleigh,
    ]

    rrtmgp_compute_tau_rayleigh(*args)

    return tau_rayleigh
