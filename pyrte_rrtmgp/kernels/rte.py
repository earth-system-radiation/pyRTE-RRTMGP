from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from pyrte_rrtmgp.pyrte_rrtmgp import (
    rte_lw_solver_2stream,
    rte_lw_solver_noscat,
    rte_sw_solver_2stream,
    rte_sw_solver_noscat,
)


def lw_solver_noscat(
    ncol: int,
    nlay: int,
    ngpt: int,
    ds: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    ssa: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lay_source: npt.NDArray[np.float64],
    lev_source: npt.NDArray[np.float64],
    sfc_emis: npt.NDArray[np.float64],
    sfc_src: npt.NDArray[np.float64],
    sfc_src_jac: npt.NDArray[np.float64],
    inc_flux: npt.NDArray[np.float64],
    top_at_1: bool = True,
    nmus: int = 1,
    do_broadband: bool = True,
    do_Jacobians: bool = False,
    do_rescaling: bool = False,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Perform longwave radiation transfer calculations without scattering.

    This function solves the longwave radiative transfer equation in the absence of scattering,
    computing fluxes and optionally their Jacobians.

    Args:
        ncol: Number of columns
        nlay: Number of layers
        ngpt: Number of g-points
        ds: Integration weights with shape (ncol, ngpt, n_quad_angs)
        weights: Gaussian quadrature weights with shape (n_quad_angs,)
        tau: Optical depths with shape (ncol, nlay, ngpt)
        ssa: Single scattering albedos with shape (ncol, nlay, ngpt)
        g: Asymmetry parameters with shape (ncol, nlay, ngpt)
        lay_source: Layer source terms with shape (ncol, nlay, ngpt)
        lev_source: Level source terms with shape (ncol, nlay+1, ngpt)
        sfc_emis: Surface emissivities with shape (ncol, ngpt) or (ncol,)
        sfc_src: Surface source terms with shape (ncol, ngpt)
        sfc_src_jac: Surface source Jacobians with shape (ncol, nlay+1)
        inc_flux: Incident fluxes with shape (ncol, ngpt)
        top_at_1: Whether the top of the atmosphere is at index 1
        nmus: Number of quadrature points (1-4)
        do_broadband: Whether to compute broadband fluxes
        do_Jacobians: Whether to compute Jacobians
        do_rescaling: Whether to perform flux rescaling

    Returns:
        Tuple containing:
            flux_up_jac: Upward flux Jacobians with shape (ncol, nlay+1)
            broadband_up: Upward broadband fluxes with shape (ncol, nlay+1)
            broadband_dn: Downward broadband fluxes with shape (ncol, nlay+1)
            flux_up: Upward fluxes with shape (ncol, nlay+1, ngpt)
            flux_dn: Downward fluxes with shape (ncol, nlay+1, ngpt)
    """
    # Initialize output arrays
    flux_up_jac = np.full((ncol, nlay + 1), np.nan, dtype=np.float64, order="F")
    broadband_up = np.full((ncol, nlay + 1), np.nan, dtype=np.float64, order="F")
    broadband_dn = np.full((ncol, nlay + 1), np.nan, dtype=np.float64, order="F")
    flux_up = np.full((ncol, nlay + 1, ngpt), np.nan, dtype=np.float64, order="F")
    flux_dn = np.full((ncol, nlay + 1, ngpt), np.nan, dtype=np.float64, order="F")

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


def lw_solver_2stream(
    ncol: int,
    nlay: int,
    ngpt: int,
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
    """Solve the longwave radiative transfer equation using the 2-stream approximation.

    This function implements the two-stream approximation for longwave radiative transfer,
    accounting for both absorption and scattering processes.

    Args:
        ncol: Number of columns
        nlay: Number of layers
        ngpt: Number of g-points
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
    # Initialize output arrays
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


def sw_solver_noscat(
    ncol: int,
    nlay: int,
    ngpt: int,
    tau: npt.NDArray[np.float64],
    mu0: npt.NDArray[np.float64],
    inc_flux_dir: npt.NDArray[np.float64],
    top_at_1: bool = True,
) -> npt.NDArray[np.float64]:
    """Perform shortwave radiation transfer calculations without scattering.

    This function solves the shortwave radiative transfer equation in the absence of
    scattering, computing direct beam fluxes only.

    Args:
        tau: Optical depths with shape (ncol, nlay, ngpt)
        mu0: Cosine of solar zenith angles with shape (ncol, nlay)
        inc_flux_dir: Direct beam incident fluxes with shape (ncol, ngpt)
        top_at_1: Whether the top of the atmosphere is at index 1

    Returns:
        Direct-beam fluxes with shape (ncol, nlay+1, ngpt)
    """
    # Initialize output array
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
    ncol: int,
    nlay: int,
    ngpt: int,
    tau: npt.NDArray[np.float64],
    ssa: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    mu0: npt.NDArray[np.float64],
    sfc_alb_dir: npt.NDArray[np.float64],
    sfc_alb_dif: npt.NDArray[np.float64],
    inc_flux_dir: npt.NDArray[np.float64],
    inc_flux_dif: npt.NDArray[np.float64],
    top_at_1: bool = True,
    has_dif_bc: bool = False,
    do_broadband: bool = True,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Perform shortwave radiation transfer calculations using the 2-stream approximation.

    This function implements the two-stream approximation for shortwave radiative transfer,
    computing direct, diffuse upward and downward fluxes, as well as optional broadband fluxes.

    Args:
        ncol: Number of columns
        nlay: Number of layers
        ngpt: Number of g-points
        tau: Optical depths with shape (ncol, nlay, ngpt)
        ssa: Single scattering albedos with shape (ncol, nlay, ngpt)
        g: Asymmetry parameters with shape (ncol, nlay, ngpt)
        mu0: Cosine of solar zenith angles with shape (ncol, ngpt)
        sfc_alb_dir: Direct surface albedos with shape (ncol, ngpt) or (ncol,)
        sfc_alb_dif: Diffuse surface albedos with shape (ncol, ngpt) or (ncol,)
        inc_flux_dir: Direct incident fluxes with shape (ncol, ngpt)
        inc_flux_dif: Diffuse incident fluxes with shape (ncol, ngpt)
        top_at_1: Whether the top of the atmosphere is at index 1
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
    # Initialize output arrays
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
