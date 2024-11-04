from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from pyrte_rrtmgp.pyrte_rrtmgp import (
    rte_lw_solver_noscat,
    rte_sw_solver_2stream,
    rte_sw_solver_noscat,
)

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

    if len(sfc_emis.shape) == 1:
        sfc_emis = np.stack([sfc_emis] * ngpt).T

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
    flux_up_jac = np.full([ncol, nlay + 1], np.nan, dtype=np.float64, order='F')
    broadband_up = np.full([ncol, nlay + 1], np.nan, dtype=np.float64, order='F')
    broadband_dn = np.full([ncol, nlay + 1], np.nan, dtype=np.float64, order='F')
    flux_up = np.full([ncol, nlay + 1, ngpt], np.nan, dtype=np.float64, order='F')
    flux_dn = np.full([ncol, nlay + 1, ngpt], np.nan, dtype=np.float64, order='F')

    args = [
        ncol,
        nlay,
        ngpt,
        top_at_1,
        nmus,
        np.asfortranarray(ds),
        np.asfortranarray(weights),
        np.asfortranarray(tau),
        np.asfortranarray(lay_source),
        np.asfortranarray(lev_source),
        np.asfortranarray(sfc_emis),
        np.asfortranarray(sfc_src),
        np.asfortranarray(inc_flux),
        flux_up,
        flux_dn,
        do_broadband,
        broadband_up,
        broadband_dn,
        do_Jacobians,
        np.asfortranarray(sfc_src_jac),
        flux_up_jac,
        do_rescaling,
        np.asfortranarray(ssa),
        np.asfortranarray(g),
    ]

    rte_lw_solver_noscat(*args)

    return flux_up_jac, broadband_up, broadband_dn, flux_up, flux_dn


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
    flux_dir = np.ndarray((ncol, nlay + 1, ngpt), dtype=np.float64, order='F')

    args = [ncol, nlay, ngpt, top_at_1, tau, mu0, inc_flux_dir, flux_dir]

    rte_sw_solver_noscat(*args)

    return flux_dir


def sw_solver_2stream(
    top_at_1,
    tau,
    ssa,
    g,
    mu0,
    sfc_alb_dir,
    sfc_alb_dif,
    inc_flux_dir,
    inc_flux_dif=None,
    has_dif_bc=False,
    do_broadband=False,
):
    """
    Solve the shortwave radiative transfer equation using the 2-stream approximation.

    Args:
        top_at_1 (bool): Flag indicating whether the top of the atmosphere is at level 1.
        tau (ndarray): Array of optical depths with shape (ncol, nlay, ngpt).
        ssa (ndarray): Array of single scattering albedos with shape (ncol, nlay, ngpt).
        g (ndarray): Array of asymmetry parameters with shape (ncol, nlay, ngpt).
        mu0 (ndarray): Array of cosine of solar zenith angles with shape (ncol, ngpt).
        sfc_alb_dir (ndarray): Array of direct surface albedos with shape (ncol, ngpt).
        sfc_alb_dif (ndarray): Array of diffuse surface albedos with shape (ncol, ngpt).
        inc_flux_dir (ndarray): Array of direct incident fluxes with shape (ncol, ngpt).
        inc_flux_dif (ndarray, optional): Array of diffuse incident fluxes with shape (ncol, ngpt).
            Defaults to None.
        has_dif_bc (bool, optional): Flag indicating whether the boundary condition includes diffuse fluxes.
            Defaults to False.
        do_broadband (bool, optional): Flag indicating whether to compute broadband fluxes.
            Defaults to False.

    Returns:
        Tuple of ndarrays: Tuple containing the following arrays:
            - flux_up: Array of upward fluxes with shape (ngpt, nlay + 1, ncol).
            - flux_dn: Array of downward fluxes with shape (ngpt, nlay + 1, ncol).
            - flux_dir: Array of direct fluxes with shape (ngpt, nlay + 1, ncol).
            - broadband_up: Array of broadband upward fluxes with shape (nlay + 1, ncol).
            - broadband_dn: Array of broadband downward fluxes with shape (nlay + 1, ncol).
            - broadband_dir: Array of broadband direct fluxes with shape (nlay + 1, ncol).
    """
    ncol, nlay, ngpt = tau.shape

    if len(sfc_alb_dir.shape) == 1:
        sfc_alb_dir = np.stack([sfc_alb_dir] * ngpt).T
    if len(sfc_alb_dif.shape) == 1:
        sfc_alb_dif = np.stack([sfc_alb_dif] * ngpt).T

    if inc_flux_dif is None:
        inc_flux_dif = np.zeros((ncol, ngpt), dtype=np.float64)

    # outputs
    flux_up = np.zeros((ncol, nlay + 1, ngpt), dtype=np.float64, order='F')
    flux_dn = np.zeros((ncol, nlay + 1, ngpt), dtype=np.float64, order='F')
    flux_dir = np.zeros((ncol, nlay + 1, ngpt), dtype=np.float64, order='F')
    broadband_up = np.zeros((ncol, nlay + 1), dtype=np.float64, order='F')
    broadband_dn = np.zeros((ncol, nlay + 1), dtype=np.float64, order='F')
    broadband_dir = np.zeros((ncol, nlay + 1), dtype=np.float64, order='F')

    args = [
        ncol,
        nlay,
        ngpt,
        top_at_1,
        np.asfortranarray(tau),
        np.asfortranarray(ssa),
        np.asfortranarray(g),
        np.asfortranarray(mu0),
        np.asfortranarray(sfc_alb_dir),
        np.asfortranarray(sfc_alb_dif),
        np.asfortranarray(inc_flux_dir),
        flux_up,
        flux_dn,
        flux_dir,
        has_dif_bc,
        np.asfortranarray(inc_flux_dif),
        do_broadband,
        broadband_up,
        broadband_dn,
        broadband_dir,
    ]

    rte_sw_solver_2stream(*args)

    return flux_up, flux_dn, flux_dir, broadband_up, broadband_dn, broadband_dir
