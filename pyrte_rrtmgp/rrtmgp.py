from typing import Tuple

import numpy as np
import numpy.typing as npt

from pyrte_rrtmgp.pyrte_rrtmgp import (
    rrtmgp_compute_Planck_source,
    rrtmgp_compute_tau_absorption,
    rrtmgp_interpolation,
    rrtmgp_compute_tau_rayleigh,
)


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
    nflav = flavor.shape[1]

    press_ref_log = np.log(press_ref)
    press_ref_log_delta = round(
        (press_ref_log.min() - press_ref_log.max()) / (len(press_ref_log) - 1), 9
    )
    press_ref_trop_log = np.log(press_ref_trop)

    temp_ref_min = temp_ref.min()
    temp_ref_delta = (temp_ref.max() - temp_ref.min()) / (len(temp_ref) - 1)

    # outputs
    jtemp = np.ndarray([ncol, nlay], dtype=np.int32)
    fmajor = np.ndarray([2, 2, 2, ncol, nlay, nflav], dtype=np.float64)
    fminor = np.ndarray([2, 2, ncol, nlay, nflav], dtype=np.float64)
    col_mix = np.ndarray([2, ncol, nlay, nflav], dtype=np.float64)
    tropo = np.ndarray([ncol, nlay], dtype=np.int32)
    jeta = np.ndarray([2, ncol, nlay, nflav], dtype=np.int32)
    jpress = np.ndarray([ncol, nlay], dtype=np.int32)

    args = [
        ncol,
        nlay,
        ngas,
        nflav,
        neta,
        npres,
        ntemp,
        flavor,
        press_ref_log,
        temp_ref,
        press_ref_log_delta,
        temp_ref_min,
        temp_ref_delta,
        press_ref_trop_log,
        vmr_ref,
        play,
        tlay,
        col_gas,
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
    _, ncol, nlay, nflav = jeta.shape
    nPlanckTemp, nbnd = totplnk.shape
    ntemp, neta, npres_e, ngpt = pfracin.shape
    npres = npres_e - 1

    sfc_lay = nlay if top_at_1 else 1

    band_ranges = [[i] * (r[1] - r[0] + 1) for i, r in enumerate(band_lims_gpt, 1)]
    gpoint_bands = np.concatenate(band_ranges)

    totplnk_delta = (temp_ref_max - temp_ref_min) / (nPlanckTemp - 1)

    # outputs
    sfc_src = np.ndarray([ncol, ngpt], dtype=np.float64)
    lay_src = np.ndarray([ncol, nlay, ngpt], dtype=np.float64)
    lev_src = np.ndarray([ncol, nlay + 1, ngpt], dtype=np.float64)
    sfc_src_jac = np.ndarray([ncol, ngpt], dtype=np.float64)

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
        tlay,
        tlev,
        tsfc,
        sfc_lay,
        fmajor,
        jeta,
        tropo,
        jtemp,
        jpress,
        gpoint_bands,
        band_lims_gpt,
        pfracin,
        temp_ref_min,
        totplnk_delta,
        totplnk,
        gpoint_flavor,
        sfc_src,
        lay_src,
        lev_src,
        sfc_src_jac,
    ]

    rrtmgp_compute_Planck_source(*args)

    return sfc_src, lay_src, lev_src, sfc_src_jac


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
    ntemp, neta, npres_e, ngpt = kmajor.shape
    npres = npres_e - 1
    nbnd = band_lims_gpt.shape[0]
    _, ncol, nlay, nflav = jeta.shape
    ngas = col_gas.shape[2]
    nminorlower = minor_scales_with_density_lower.shape[0]
    nminorupper = minor_scales_with_density_upper.shape[0]
    nminorklower = kminor_lower.shape[2]
    nminorkupper = kminor_upper.shape[2]
    
    
    # outputs
    tau = np.ndarray([ncol, nlay, nbnd], dtype=np.float64)

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
        tau,
    ]

    rrtmgp_compute_tau_absorption(*args)

    return tau


def compute_tau_rayleigh(
    npres,
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

    ncol, nlay, ngas = col_gas.shape
    ntemp, neta, ngpt = krayl.shape
    nflav = jeta.shape[3]
    nbnd = band_lims_gpt.shape[1]

    # outputs
    tau_rayleigh = np.ndarray((ncol, nlay, ngpt), dtype=np.float64)

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
        tau_rayleigh,
    ]

    rrtmgp_compute_tau_rayleigh(*args)

    return tau


