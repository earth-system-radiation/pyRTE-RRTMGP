from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from pyrte_rrtmgp.pyrte_rrtmgp import rte_lw_solver_noscat, rte_sw_solver_noscat

GAUSS_DS = np.array(
    [
        [1.66, 0.0, 0.0, 0.0],  # Diffusivity angle, not Gaussian angle
        [1.18350343, 2.81649655, 0.0, 0.0],
        [1.09719858, 1.69338507, 4.70941630, 0.0],
        [1.06056257, 1.38282560, 2.40148179, 7.15513024],
    ]
)


GAUSS_WTS = np.array(
    [
        [0.5, 0.0, 0.0, 0.0],
        [0.3180413817, 0.1819586183, 0.0, 0.0],
        [0.2009319137, 0.2292411064, 0.0698269799, 0.0],
        [0.1355069134, 0.2034645680, 0.1298475476, 0.0311809710],
    ]
)


def lw_solver_noscat(
    tau: npt.NDArray,
    lay_source: npt.NDArray,
    lev_source: npt.NDArray,
    sfc_emis: npt.NDArray,
    sfc_src: npt.NDArray,
    top_at_1: bool = True,
    nmus: int = 1,
    inc_flux: Optional[npt.NDArray] = None,
    ds: Optional[npt.NDArray] = None,
    weights: Optional[npt.NDArray] = None,
    do_broadband: Optional[bool] = True,
    do_Jacobians: Optional[bool] = False,
    sfc_src_jac: Optional[npt.NDArray] = [],
    do_rescaling: Optional[bool] = False,
    ssa: Optional[npt.NDArray] = None,
    g: Optional[np.ndarray] = None,
) -> Tuple:
    """
    Perform longwave radiation transfer calculations without scattering.

    Args:
        top_at_1 (bool): Flag indicating whether the top of the atmosphere is at level 1.
        nmus (int): Number of quadrature points.
        tau (npt.NDArray): Array of optical depths.
        lay_source (npt.NDArray): Array of layer sources.
        lev_source (npt.NDArray): Array of level sources.
        sfc_emis (npt.NDArray): Array of surface emissivities.
        sfc_src (npt.NDArray): Array of surface sources.
        inc_flux (npt.NDArray): Array of incoming fluxes.
        ds (Optional[npt.NDArray], optional): Array of integration weights. Defaults to None.
        weights (Optional[npt.NDArray], optional): Array of Gaussian quadrature weights. Defaults to None.
        do_broadband (Optional[bool], optional): Flag indicating whether to compute broadband fluxes. Defaults to None.
        do_Jacobians (Optional[bool], optional): Flag indicating whether to compute Jacobians. Defaults to None.
        sfc_src_jac (Optional[npt.NDArray], optional): Array of surface source Jacobians. Defaults to None.
        do_rescaling (Optional[bool], optional): Flag indicating whether to perform flux rescaling. Defaults to None.
        ssa (Optional[npt.NDArray], optional): Array of single scattering albedos. Defaults to None.
        g (Optional[np.ndarray], optional): Array of asymmetry parameters. Defaults to None.

    Returns:
        Tuple: A tuple containing the following arrays:
            - flux_up_jac (np.ndarray): Array of upward flux Jacobians.
            - broadband_up (np.ndarray): Array of upward broadband fluxes.
            - broadband_dn (np.ndarray): Array of downward broadband fluxes.
            - flux_up (np.ndarray): Array of upward fluxes.
            - flux_dn (np.ndarray): Array of downward fluxes.
    """

    ncol, nlay, ngpt = tau.shape

    # default values
    n_quad_angs = nmus

    if inc_flux is None:
        inc_flux = np.zeros(sfc_src.shape)

    if ds is None:
        ds = np.empty((ncol, ngpt, n_quad_angs))
        for imu in range(n_quad_angs):
            for igpt in range(ngpt):
                for icol in range(ncol):
                    ds[icol, igpt, imu] = GAUSS_DS[imu, n_quad_angs - 1]

    if weights is None:
        weights = GAUSS_WTS[0:n_quad_angs, n_quad_angs - 1]

    ssa = ssa or tau
    g = g or tau

    # outputs
    flux_up_jac = np.full([nlay + 1, ncol], np.nan, dtype=np.float64)
    broadband_up = np.full([nlay + 1, ncol], np.nan, dtype=np.float64)
    broadband_dn = np.full([nlay + 1, ncol], np.nan, dtype=np.float64)
    flux_up = np.full([ngpt, nlay + 1, ncol], np.nan, dtype=np.float64)
    flux_dn = np.full([ngpt, nlay + 1, ncol], np.nan, dtype=np.float64)

    args = [
        ncol,
        nlay,
        ngpt,
        top_at_1,
        nmus,
        ds.flatten("F"),
        weights.flatten("F"),
        tau.flatten("F"),
        lay_source.flatten("F"),
        lev_source.flatten("F"),
        sfc_emis.flatten("F"),
        sfc_src.flatten("F"),
        inc_flux.flatten("F"),
        flux_up,
        flux_dn,
        do_broadband,
        broadband_up,
        broadband_dn,
        do_Jacobians,
        sfc_src_jac.flatten("F"),
        flux_up_jac,
        do_rescaling,
        ssa.flatten("F"),
        g.flatten("F"),
    ]

    rte_lw_solver_noscat(*args)

    return flux_up_jac.T, broadband_up.T, broadband_dn.T, flux_up.T, flux_dn.T


def sw_solver_noscat(
    top_at_1,
    tau,
    mu0,
    inc_flux_dir,
):
    """
    Computes the direct-beam flux for a shortwave radiative transfer problem without scattering.

    Args:
        top_at_1 (bool): Logical flag indicating if the top layer is at index 1.
        tau (numpy.ndarray): Absorption optical thickness of size (ncol, nlay, ngpt).
        mu0 (numpy.ndarray): Cosine of solar zenith angle of size (ncol, nlay).
        inc_flux_dir (numpy.ndarray): Direct beam incident flux of size (ncol, ngpt).

    Returns:
        numpy.ndarray: Direct-beam flux of size (ncol, nlay+1, ngpt).
    """

    ncol, nlay, ngpt = tau.shape

    # outputs
    flux_dir = np.ndarray((ncol, nlay + 1, ngpt), dtype=np.float64)

    args = [ncol, nlay, ngpt, top_at_1, tau, mu0, inc_flux_dir, flux_dir]

    rte_lw_solver_noscat(*args)

    return flux_dir
