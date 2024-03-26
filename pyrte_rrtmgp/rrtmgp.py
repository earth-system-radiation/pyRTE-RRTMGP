from typing import Tuple

import numpy as np
import numpy.typing as npt

from pyrte_rrtmgp.pyrte_rrtmgp import rrtmgp_interpolation


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
        jtemp (np.ndarray): Temperature interpolation index.
        fmajor (np.ndarray): Major gas interpolation fraction.
        fminor (np.ndarray): Minor gas interpolation fraction.
        col_mix (np.ndarray): Mixing fractions.
        tropo (np.ndarray): Use lower (or upper) atmosphere tables.
        jeta (np.ndarray): Index for binary species interpolation.
        jpress (np.ndarray): Pressure interpolation index.
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

    arg_list = [
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

    rrtmgp_interpolation(*arg_list)

    return jtemp, fmajor, fminor, col_mix, tropo, jeta, jpress
