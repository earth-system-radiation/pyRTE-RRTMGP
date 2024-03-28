from pyrte_rrtmgp.pyrte_rrtmgp import rte_lw_solver_noscat
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional


GAUSS_DS = np.array([1.66, 0., 0., 0.],  # Diffusivity angle, not Gaussian angle
                    [1.18350343, 2.81649655, 0., 0.],
                    [1.09719858, 1.69338507, 4.70941630, 0.],
                    [1.06056257, 1.38282560, 2.40148179, 7.15513024])


GAUSS_WTS = np.array([0.5, 0., 0., 0.],
                    [0.3180413817, 0.1819586183, 0., 0.],
                    [0.2009319137, 0.2292411064, 0.0698269799, 0.],
                    [0.1355069134, 0.2034645680, 0.1298475476, 0.0311809710])


def lw_solver_noscat(
    top_at_1: bool,
    nmus: int,
    tau: npt.NDArray,
    lay_source: npt.NDArray,
    lev_source: npt.NDArray,
    sfc_emis: npt.NDArray,
    sfc_src: npt.NDArray,
    inc_flux: npt.NDArray,
    ds: Optional[npt.NDArray] = None,
    weights: Optional[npt.NDArray] = None,
    do_broadband: Optional[bool] = None,
    do_Jacobians: Optional[bool] = None,
    sfc_src_jac: Optional[npt.NDArray] = None,
    do_rescaling: Optional[bool] = None,
    ssa: Optional[npt.NDArray] = None,
    g: Optional[np.ndarray] = None
) -> Tuple:
    
    ncol, nlay, ngpt = tau.shape

    # default values
    n_quad_angs = nmus

    if ds is None:
        ds = np.empty((ncol, ngpt, n_quad_angs))
        for imu in range(n_quad_angs):
            for igpt in range(ngpt):
                for icol in range(ncol):
                    ds[icol, igpt, imu] = GAUSS_DS[imu, n_quad_angs]

    if weights is None:
        weights = GAUSS_WTS[1:n_quad_angs,n_quad_angs]

    # outputs
    flux_up_jac = np.ndarray([ncol, nlay+1], dtype=np.float64)
    broadband_up = np.ndarray([ncol, nlay+1], dtype=np.float64)
    broadband_dn = np.ndarray([ncol, nlay+1], dtype=np.float64)
    flux_up = np.ndarray([ncol, nlay+1, ngpt], dtype=np.float64)
    flux_dn = np.ndarray([ncol, nlay+1, ngpt], dtype=np.float64)

    args = [
        ncol,
        nlay,
        ngpt,
        top_at_1,
        nmus,
        ds,
        weights,
        tau,
        lay_source,
        lev_source,
        sfc_emis,
        sfc_src,
        inc_flux,
        flux_up,
        flux_dn,
        do_broadband,
        broadband_up,
        broadband_dn,
        do_Jacobians,
        sfc_src_jac,
        flux_up_jac,
        do_rescaling,
        ssa,
        g
    ]

    rte_lw_solver_noscat(*args)

    return flux_up_jac, broadband_up, broadband_dn, flux_up, flux_dn