from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from pyrte_rrtmgp.pyrte_rrtmgp import (
    rte_lw_solver_noscat,
    rte_sw_solver_2stream,
    rte_sw_solver_noscat,
    rte_lw_solver_2stream,
)


GAUSS_DS: npt.NDArray[np.float64] = np.reciprocal(
    np.array(
        [
            [0.6096748751, np.inf, np.inf, np.inf],
            [0.2509907356, 0.7908473988, np.inf, np.inf],
            [0.1024922169, 0.4417960320, 0.8633751621, np.inf],
            [0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973],
        ]
    )
)
"""Gaussian quadrature secants (1/μ) for different numbers of streams"""

GAUSS_WTS: npt.NDArray[np.float64] = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.2300253764, 0.7699746236, 0.0, 0.0],
        [0.0437820218, 0.3875796738, 0.5686383044, 0.0],
        [0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087],
    ]
)
"""Gaussian quadrature weights for different numbers of streams"""


def lw_solver_noscat(
    tau: npt.NDArray[np.float64],
    lay_source: npt.NDArray[np.float64],
    lev_source: npt.NDArray[np.float64],
    sfc_emis: npt.NDArray[np.float64],
    sfc_src: npt.NDArray[np.float64],
    top_at_1: bool = True,
    nmus: int = 1,
    inc_flux: Optional[npt.NDArray[np.float64]] = None,
    ds: Optional[npt.NDArray[np.float64]] = None,
    weights: Optional[npt.NDArray[np.float64]] = None,
    do_broadband: bool = True,
    do_Jacobians: bool = False,
    sfc_src_jac: Optional[npt.NDArray[np.float64]] = None,
    do_rescaling: bool = False,
    ssa: Optional[npt.NDArray[np.float64]] = None,
    g: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], 
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Perform longwave radiation transfer calculations without scattering.

    Args:
        tau: Optical depths with shape (ncol, nlay, ngpt)
        lay_source: Layer source terms with shape (ncol, nlay, ngpt)
        lev_source: Level source terms with shape (ncol, nlay+1, ngpt)
        sfc_emis: Surface emissivities with shape (ncol, ngpt) or (ncol,)
        sfc_src: Surface source terms with shape (ncol, ngpt)
        top_at_1: Whether the top of the atmosphere is at index 1
        nmus: Number of quadrature points (1-4)
        inc_flux: Incident fluxes with shape (ncol, ngpt)
        ds: Integration weights with shape (ncol, ngpt, n_quad_angs)
        weights: Gaussian quadrature weights with shape (n_quad_angs,)
        do_broadband: Whether to compute broadband fluxes
        do_Jacobians: Whether to compute Jacobians
        sfc_src_jac: Surface source Jacobians with shape (ncol, nlay+1)
        do_rescaling: Whether to perform flux rescaling
        ssa: Single scattering albedos with shape (ncol, nlay, ngpt)
        g: Asymmetry parameters with shape (ncol, nlay, ngpt)

    Returns:
        Tuple containing:
            flux_up_jac: Upward flux Jacobians (ncol, nlay+1)
            broadband_up: Upward broadband fluxes (ncol, nlay+1)
            broadband_dn: Downward broadband fluxes (ncol, nlay+1)
            flux_up: Upward fluxes (ncol, nlay+1, ngpt)
            flux_dn: Downward fluxes (ncol, nlay+1, ngpt)
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
    flux_up_jac = np.full([ncol, nlay + 1], np.nan, dtype=np.float64, order="F")
    broadband_up = np.full([ncol, nlay + 1], np.nan, dtype=np.float64, order="F")
    broadband_dn = np.full([ncol, nlay + 1], np.nan, dtype=np.float64, order="F")
    flux_up = np.full([ncol, nlay + 1, ngpt], np.nan, dtype=np.float64, order="F")
    flux_dn = np.full([ncol, nlay + 1, ngpt], np.nan, dtype=np.float64, order="F")

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
    tau: npt.NDArray[np.float64],
    mu0: npt.NDArray[np.float64],
    inc_flux_dir: npt.NDArray[np.float64],
    top_at_1: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Perform shortwave radiation transfer calculations without scattering.

    Args:
        tau: Optical depths with shape (ncol, nlay, ngpt)
        mu0: Cosine of solar zenith angles with shape (ncol, nlay)
        inc_flux_dir: Direct beam incident fluxes with shape (ncol, ngpt)
        top_at_1: Whether the top of the atmosphere is at index 1

    Returns:
        Direct-beam fluxes with shape (ncol, nlay+1, ngpt)
    """
    ncol, nlay, ngpt = tau.shape

    # outputs
    flux_dir = np.zeros((ncol, nlay + 1, ngpt), dtype=np.float64, order="F")

    args = [
        ncol,
        nlay,
        ngpt,
        top_at_1,
        np.asfortranarray(tau),
        np.asfortranarray(mu0),
        np.asfortranarray(inc_flux_dir),
        flux_dir,
    ]

    rte_sw_solver_noscat(*args)

    return flux_dir


def sw_solver_2stream(
    tau: npt.NDArray[np.float64],
    ssa: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    mu0: npt.NDArray[np.float64],
    sfc_alb_dir: npt.NDArray[np.float64],
    sfc_alb_dif: npt.NDArray[np.float64],
    inc_flux_dir: npt.NDArray[np.float64],
    top_at_1: bool = True,
    inc_flux_dif: Optional[npt.NDArray[np.float64]] = None,
    has_dif_bc: bool = False,
    do_broadband: bool = True,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64],
          npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Perform shortwave radiation transfer calculations using the 2-stream approximation.

    Args:
        tau: Optical depths with shape (ncol, nlay, ngpt)
        ssa: Single scattering albedos with shape (ncol, nlay, ngpt)
        g: Asymmetry parameters with shape (ncol, nlay, ngpt)
        mu0: Cosine of solar zenith angles with shape (ncol, ngpt)
        sfc_alb_dir: Direct surface albedos with shape (ncol, ngpt) or (ncol,)
        sfc_alb_dif: Diffuse surface albedos with shape (ncol, ngpt) or (ncol,)
        inc_flux_dir: Direct incident fluxes with shape (ncol, ngpt)
        top_at_1: Whether the top of the atmosphere is at index 1
        inc_flux_dif: Diffuse incident fluxes with shape (ncol, ngpt)
        has_dif_bc: Whether the boundary condition includes diffuse fluxes
        do_broadband: Whether to compute broadband fluxes

    Returns:
        Tuple containing:
            flux_up: Upward fluxes with shape (ncol, nlay+1, ngpt)
            flux_dn: Downward fluxes with shape (ncol, nlay+1, ngpt)
            flux_dir: Direct fluxes with shape (ncol, nlay+1, ngpt)
            broadband_up: Broadband upward fluxes with shape (ncol, nlay+1)
            broadband_dn: Broadband downward fluxes with shape (ncol, nlay+1)
            broadband_dir: Broadband direct fluxes with shape (ncol, nlay+1)
    """
    ncol, nlay, ngpt = tau.shape

    if len(sfc_alb_dir.shape) == 1:
        sfc_alb_dir = np.stack([sfc_alb_dir] * ngpt).T
    if len(sfc_alb_dif.shape) == 1:
        sfc_alb_dif = np.stack([sfc_alb_dif] * ngpt).T

    if inc_flux_dif is None:
        inc_flux_dif = np.zeros((ncol, ngpt), dtype=np.float64)

    # outputs
    flux_up = np.zeros((ncol, nlay + 1, ngpt), dtype=np.float64, order="F")
    flux_dn = np.zeros((ncol, nlay + 1, ngpt), dtype=np.float64, order="F")
    flux_dir = np.zeros((ncol, nlay + 1, ngpt), dtype=np.float64, order="F")
    broadband_up = np.zeros((ncol, nlay + 1), dtype=np.float64, order="F")
    broadband_dn = np.zeros((ncol, nlay + 1), dtype=np.float64, order="F")
    broadband_dir = np.zeros((ncol, nlay + 1), dtype=np.float64, order="F")

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


def lw_solver_2stream(
    tau: npt.NDArray[np.float64],
    ssa: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lay_source: npt.NDArray[np.float64],
    lev_source: npt.NDArray[np.float64],
    sfc_emis: npt.NDArray[np.float64],
    sfc_src: npt.NDArray[np.float64],
    inc_flux: npt.NDArray[np.float64],
    top_at_1: bool = True,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Solve the longwave radiative transfer equation using the 2-stream approximation.

    Args:
        tau: Optical depths with shape (ncol, nlay, ngpt)
        ssa: Single-scattering albedos with shape (ncol, nlay, ngpt)
        g: Asymmetry parameters with shape (ncol, nlay, ngpt)
        lay_source: Layer source terms with shape (ncol, nlay, ngpt)
        lev_source: Level source terms with shape (ncol, nlay+1, ngpt)
        sfc_emis: Surface emissivities with shape (ncol, ngpt) or (ncol,)
        sfc_src: Surface source terms with shape (ncol, ngpt)
        inc_flux: Incident fluxes with shape (ncol, ngpt)
        top_at_1: Whether the top of the atmosphere is at index 1

    Returns:
        Tuple containing:
            flux_up: Upward fluxes with shape (ncol, nlay+1, ngpt)
            flux_dn: Downward fluxes with shape (ncol, nlay+1, ngpt)
    """
    ncol, nlay, ngpt = tau.shape

    if len(sfc_emis.shape) == 1:
        sfc_emis = np.stack([sfc_emis] * ngpt).T

    # outputs
    flux_up = np.zeros((ncol, nlay + 1, ngpt), dtype=np.float64, order="F")
    flux_dn = np.zeros((ncol, nlay + 1, ngpt), dtype=np.float64, order="F")

    args = [
        ncol,
        nlay,
        ngpt,
        top_at_1,
        np.asfortranarray(tau),
        np.asfortranarray(ssa),
        np.asfortranarray(g),
        np.asfortranarray(lay_source),
        np.asfortranarray(lev_source),
        np.asfortranarray(sfc_emis),
        np.asfortranarray(sfc_src),
        np.asfortranarray(inc_flux),
        flux_up,
        flux_dn,
    ]

    rte_lw_solver_2stream(*args)

    return flux_up, flux_dn
