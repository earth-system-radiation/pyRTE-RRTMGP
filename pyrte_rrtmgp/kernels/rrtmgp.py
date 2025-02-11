from typing import Tuple

import numpy as np
import numpy.typing as npt

from pyrte_rrtmgp.pyrte_rrtmgp import (
    rrtmgp_compute_Planck_source,
    rrtmgp_compute_tau_absorption,
    rrtmgp_compute_tau_rayleigh,
    rrtmgp_interpolation,
)


def interpolation(
    ncol: int,
    nlay: int,
    ngas: int,
    nflav: int,
    neta: int,
    npres: int,
    ntemp: int,
    flavor: npt.NDArray[np.int32],
    press_ref: npt.NDArray[np.float64],
    temp_ref: npt.NDArray[np.float64],
    press_ref_trop: float,
    vmr_ref: npt.NDArray[np.float64],
    play: npt.NDArray[np.float64],
    tlay: npt.NDArray[np.float64],
    col_gas: npt.NDArray[np.float64],
) -> Tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Interpolate the RRTMGP coefficients to the current atmospheric state.

    This function performs interpolation of gas optics coefficients based on the current
    atmospheric temperature and pressure profiles.

    Args:
        ncol: Number of atmospheric columns
        nlay: Number of atmospheric layers
        ngas: Number of gases
        nflav: Number of gas flavors
        neta: Number of mixing fraction points
        npres: Number of reference pressure grid points
        ntemp: Number of reference temperature grid points
        flavor: Index into vmr_ref of major gases for each flavor with shape (nflav,)
        press_ref: Reference pressure grid with shape (npres,)
        temp_ref: Reference temperature grid with shape (ntemp,)
        press_ref_trop: Reference pressure at the tropopause
        vmr_ref: Reference volume mixing ratios with shape (ngas,)
        play: Layer pressures with shape (ncol, nlay)
        tlay: Layer temperatures with shape (ncol, nlay)
        col_gas: Gas concentrations with shape (ncol, nlay, ngas)

    Returns:
        Tuple containing:
            - jtemp: Temperature interpolation indices with shape (ncol, nlay)
            - fmajor: Major gas interpolation fractions with shape (2, 2, 2, ncol, nlay, nflav)
            - fminor: Minor gas interpolation fractions with shape (2, 2, ncol, nlay, nflav)
            - col_mix: Mixing fractions with shape (2, ncol, nlay, nflav)
            - tropo: Boolean mask for troposphere with shape (ncol, nlay)
            - jeta: Binary species interpolation indices with shape (2, ncol, nlay, nflav)
            - jpress: Pressure interpolation indices with shape (ncol, nlay)
    """
    press_ref_log = np.log(press_ref)
    press_ref_log_delta = (press_ref_log.min() - press_ref_log.max()) / (
        len(press_ref_log) - 1
    )
    press_ref_trop_log = np.log(press_ref_trop)

    temp_ref_min = temp_ref.min()
    temp_ref_delta = (temp_ref.max() - temp_ref.min()) / (len(temp_ref) - 1)

    ngas = ngas - 1  # Fortran uses index 0 here

    # Initialize output arrays
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


def compute_planck_source(
    ncol: int,
    nlay: int,
    nbnd: int,
    ngpt: int,
    nflav: int,
    neta: int,
    npres: int,
    ntemp: int,
    nPlanckTemp: int,
    tlay: npt.NDArray[np.float64],
    tlev: npt.NDArray[np.float64],
    tsfc: npt.NDArray[np.float64],
    top_at_1: bool,
    fmajor: npt.NDArray[np.float64],
    jeta: npt.NDArray[np.int32],
    tropo: npt.NDArray[np.bool_],
    jtemp: npt.NDArray[np.int32],
    jpress: npt.NDArray[np.int32],
    band_lims_gpt: npt.NDArray[np.int32],
    pfracin: npt.NDArray[np.float64],
    temp_ref_min: float,
    temp_ref_max: float,
    totplnk: npt.NDArray[np.float64],
    gpoint_flavor: npt.NDArray[np.int32],
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute the Planck source function for radiative transfer calculations.

    This function calculates the Planck blackbody emission source terms needed for
    longwave radiative transfer calculations.

    Args:
        ncol: Number of atmospheric columns
        nlay: Number of atmospheric layers
        nbnd: Number of spectral bands
        ngpt: Number of g-points
        nflav: Number of gas flavors
        neta: Number of eta points
        npres: Number of pressure points
        ntemp: Number of temperature points
        nPlanckTemp: Number of temperatures for Planck function
        tlay: Layer temperatures with shape (ncol, nlay)
        tlev: Level temperatures with shape (ncol, nlay+1)
        tsfc: Surface temperatures with shape (ncol,)
        top_at_1: Whether the top of the atmosphere is at index 1
        fmajor: Major gas interpolation weights with shape (2, 2, 2, ncol, nlay, nflav)
        jeta: Eta interpolation indices with shape (2, ncol, nlay, nflav)
        tropo: Troposphere mask with shape (ncol, nlay)
        jtemp: Temperature interpolation indices with shape (ncol, nlay)
        jpress: Pressure interpolation indices with shape (ncol, nlay)
        gpoint_bands: TODO: Add information (ngpt)
        band_lims_gpt: Band limits in g-point space with shape (2, nbnd)
        pfracin: Planck fractions with shape (ntemp, neta, npres+1, ngpt)
        temp_ref_min: Minimum reference temperature
        temp_ref_max: Maximum reference temperature
        totplnk: Total Planck function by band with shape (nPlanckTemp, nbnd)
        gpoint_flavor: G-point flavors with shape (2, ngpt)

    Returns:
        Tuple containing:
            - sfc_src: Surface emission with shape (ncol, ngpt)
            - lay_src: Layer emission with shape (ncol, nlay, ngpt)
            - lev_src: Level emission with shape (ncol, nlay+1, ngpt)
            - sfc_src_jac: Surface emission Jacobian with shape (ncol, ngpt)
    """
    sfc_lay = nlay if top_at_1 else 1
    gpoint_bands = np.ndarray((ngpt), dtype=np.int32, order="F")
    totplnk_delta = (temp_ref_max - temp_ref_min) / (nPlanckTemp - 1)

    # Initialize output arrays
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


def compute_tau_absorption(
    ncol: int,
    nlay: int,
    nbnd: int,
    ngpt: int,
    ngas: int,
    nflav: int,
    neta: int,
    npres: int,
    ntemp: int,
    nminorlower: int,
    nminorklower: int,
    nminorupper: int,
    nminorkupper: int,
    idx_h2o: int,
    gpoint_flavor: npt.NDArray[np.int32],
    band_lims_gpt: npt.NDArray[np.int32],
    kmajor: npt.NDArray[np.float64],
    kminor_lower: npt.NDArray[np.float64],
    kminor_upper: npt.NDArray[np.float64],
    minor_limits_gpt_lower: npt.NDArray[np.int32],
    minor_limits_gpt_upper: npt.NDArray[np.int32],
    minor_scales_with_density_lower: npt.NDArray[np.bool_],
    minor_scales_with_density_upper: npt.NDArray[np.bool_],
    scale_by_complement_lower: npt.NDArray[np.bool_],
    scale_by_complement_upper: npt.NDArray[np.bool_],
    idx_minor_lower: npt.NDArray[np.int32],
    idx_minor_upper: npt.NDArray[np.int32],
    idx_minor_scaling_lower: npt.NDArray[np.int32],
    idx_minor_scaling_upper: npt.NDArray[np.int32],
    kminor_start_lower: npt.NDArray[np.int32],
    kminor_start_upper: npt.NDArray[np.int32],
    tropo: npt.NDArray[np.bool_],
    col_mix: npt.NDArray[np.float64],
    fmajor: npt.NDArray[np.float64],
    fminor: npt.NDArray[np.float64],
    play: npt.NDArray[np.float64],
    tlay: npt.NDArray[np.float64],
    col_gas: npt.NDArray[np.float64],
    jeta: npt.NDArray[np.int32],
    jtemp: npt.NDArray[np.int32],
    jpress: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Compute the absorption optical depth for atmospheric profiles.

    This function calculates the total absorption optical depth by combining contributions
    from major and minor gas species in both the upper and lower atmosphere.

    Args:
        ncol: Number of atmospheric columns
        nlay: Number of atmospheric layers
        nbnd: Number of spectral bands
        ngpt: Number of g-points
        ngas: Number of gases
        nflav: Number of gas flavors
        neta: Number of eta points
        npres: Number of pressure points
        ntemp: Number of temperature points
        nminorlower: Number of minor species in lower atmosphere
        nminorklower: Number of minor absorption coefficients in lower atmosphere
        nminorupper: Number of minor species in upper atmosphere
        nminorkupper: Number of minor absorption coefficients in upper atmosphere
        idx_h2o: Index of water vapor
        gpoint_flavor: G-point flavors with shape (2, ngpt)
        band_lims_gpt: Band limits in g-point space with shape (2, nbnd)
        kmajor: Major gas absorption coefficients
        kminor_lower: Minor gas absorption coefficients for lower atmosphere
        kminor_upper: Minor gas absorption coefficients for upper atmosphere
        minor_limits_gpt_lower: G-point limits for minor gases in lower atmosphere
        minor_limits_gpt_upper: G-point limits for minor gases in upper atmosphere
        minor_scales_with_density_lower: Density scaling flags for lower atmosphere
        minor_scales_with_density_upper: Density scaling flags for upper atmosphere
        scale_by_complement_lower: Complement scaling flags for lower atmosphere
        scale_by_complement_upper: Complement scaling flags for upper atmosphere
        idx_minor_lower: Minor gas indices for lower atmosphere
        idx_minor_upper: Minor gas indices for upper atmosphere
        idx_minor_scaling_lower: Minor gas scaling indices for lower atmosphere
        idx_minor_scaling_upper: Minor gas scaling indices for upper atmosphere
        kminor_start_lower: Starting indices for minor gases in lower atmosphere
        kminor_start_upper: Starting indices for minor gases in upper atmosphere
        tropo: Troposphere mask with shape (ncol, nlay)
        col_mix: Gas mixing ratios with shape (2, ncol, nlay, nflav)
        fmajor: Major gas interpolation weights
        fminor: Minor gas interpolation weights
        play: Layer pressures with shape (ncol, nlay)
        tlay: Layer temperatures with shape (ncol, nlay)
        col_gas: Gas concentrations with shape (ncol, nlay, ngas)
        jeta: Eta interpolation indices
        jtemp: Temperature interpolation indices
        jpress: Pressure interpolation indices

    Returns:
        Absorption optical depth with shape (ncol, nlay, ngpt)
    """
    ngas = ngas - 1  # Fortran uses index 0 here

    # Initialize output array
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


def compute_tau_rayleigh(
    ncol: int,
    nlay: int,
    nbnd: int,
    ngpt: int,
    ngas: int,
    nflav: int,
    neta: int,
    ntemp: int,
    gpoint_flavor: npt.NDArray[np.int32],
    band_lims_gpt: npt.NDArray[np.int32],
    krayl: npt.NDArray[np.float64],
    idx_h2o: int,
    col_dry: npt.NDArray[np.float64],
    col_gas: npt.NDArray[np.float64],
    fminor: npt.NDArray[np.float64],
    jeta: npt.NDArray[np.int32],
    tropo: npt.NDArray[np.bool_],
    jtemp: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Compute Rayleigh scattering optical depth.

    This function calculates the optical depth due to Rayleigh scattering by air molecules.

    Args:
        ncol: Number of atmospheric columns
        nlay: Number of atmospheric layers
        nbnd: Number of spectral bands
        ngpt: Number of g-points
        ngas: Number of gases
        nflav: Number of gas flavors
        neta: Number of eta points
        ntemp: Number of temperature points
        gpoint_flavor: G-point flavors with shape (2, ngpt)
        band_lims_gpt: Band limits in g-point space with shape (2, nbnd)
        krayl: Rayleigh scattering coefficients with shape (ntemp, neta, ngpt, 2)
        idx_h2o: Index of water vapor
        col_dry: Dry air column amounts with shape (ncol, nlay)
        col_gas: Gas concentrations with shape (ncol, nlay, ngas + 1)
        fminor: Minor gas interpolation weights
        jeta: Eta interpolation indices
        tropo: Troposphere mask with shape (ncol, nlay)
        jtemp: Temperature interpolation indices

    Returns:
        Rayleigh scattering optical depth with shape (ncol, nlay, ngpt)
    """
    # Initialize output array
    tau_rayleigh = np.ndarray((ncol, nlay, ngpt), dtype=np.float64, order="F")

    ngas = ngas - 1  # Fortran uses index 0 here

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
