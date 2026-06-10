"""
Core xarray-based numerical kernels for the Simple Spectral Model (SSM).

References
----------
Williams, A. I. L. (2026). Bridging clarity and accuracy: A simple spectral
longwave radiation scheme for idealized climate modeling.
Journal of Advances in Modeling Earth Systems, 18, e2025MS005405.
https://doi.org/10.1029/2025MS005405

"""

from typing import Final

import numpy as np
import scipy.constants as sc
import xarray as xr

from pyrte_rrtmgp.utils import B_nu

GRAV: Final[float] = sc.g


def compute_absorption_coeffs(
    triangles: xr.DataArray,
    nus: xr.DataArray,
) -> xr.DataArray:
    """
    Compute reference absorption coefficients for each spectral tag.

    Parameters
    ----------
    triangles:
        DataArray with dims ("tag", "param"). Expected params are
        "nu0", "l", and "kappa0".

    nus:
        DataArray with shape (?) ", containing wavenumber grid points [cm^-1].

    Returns
    -------
    xr.DataArray
        Absorption coefficients with dims ("tag", "gpt").
    """
    nu0 = triangles.sel(param="nu0")
    ell = triangles.sel(param="l")
    kappa0 = triangles.sel(param="kappa0")

    absorption_coeffs = kappa0 * np.exp(-abs(nus - nu0) / ell)
    absorption_coeffs = absorption_coeffs.rename("absorption_coeffs")
    absorption_coeffs.attrs["units"] = "m2 kg-1"

    return absorption_coeffs


def compute_layer_mass(
    vmr: xr.DataArray,
    plev: xr.DataArray,
    play: xr.DataArray,
    mol_weights: xr.DataArray,
    m_dry: float = 0.029,
) -> xr.DataArray:
    """
    Convert volume mixing ratios and pressure levels to gas layer masses.

    Parameters
    ----------
    vmr:
        Volume mixing ratio with dims ("tag", column_dim, layer_dim).

    plev:
        Pressure at layer interfaces with dims (column_dim, level_dim).

    play:
        Layer pressure with dims (column_dim, layer_dim). Used to recover
        the layer dimension name and coordinates.

    mol_weights:
        Molecular weights aligned over dim "tag".

    m_dry: float
      Molecular weight of dry air [kg/mol]

    Returns
    -------
    xr.DataArray
        Layer mass with dims ("tag", column_dim, layer_dim).
        Mass of each gas in each layer [kg m^-2]
    """
    lev_dim = plev.dims[-1]
    lay_dim = play.dims[-1]

    dp = abs(plev.diff(lev_dim))
    dp = dp.rename({lev_dim: lay_dim})

    if lay_dim in play.coords:
        dp = dp.assign_coords({lay_dim: play[lay_dim]})

    mmr = vmr * (mol_weights / m_dry)
    layer_mass = mmr * dp / GRAV

    layer_mass = layer_mass.rename("layer_mass")
    layer_mass.attrs["units"] = "kg m-2"

    return layer_mass


"""
    Compute absorption optical depth.

    Optical depth is computed as the pressure-scaled sum over spectral tags:

    ``tau = (play / pref) * sum_tag(layer_mass * absorption_coeffs)``.

    Parameters
    ----------
    absorption_coeffs:
        Absorption coefficients with dimensions ``("tag", "gpt")``.
        Reference absorption coefficients [m^2 kg^-1] from compute_absorption_coeffs()

    play:
        Layer pressure with dimensions ``(column_dim, layer_dim)``.
        Layer pressures [Pa]

    pref:
        Reference pressure in Pa. If zero, pressure scaling is set to one.

    layer_mass:
        Gas layer mass with dimensions ``("tag", column_dim, layer_dim)``.
        Gas layer masses [kg m^-2] from compute_layer_mass()

    Returns
    -------
    xr.DataArray
        Optical depth with dimensions ``(column_dim, layer_dim, "gpt")``.
"""


def compute_tau(
    absorption_coeffs: xr.DataArray,
    play: xr.DataArray,
    pref: float,
    layer_mass: xr.DataArray,
) -> xr.DataArray:
    """Compute optical depth from simple spectral model."""
    if pref != 0.0:
        p_scaling = play / pref
    else:
        p_scaling = xr.ones_like(play)

    tau = p_scaling * (layer_mass * absorption_coeffs).sum("tag")
    tau = tau.rename("tau")

    return tau


def compute_planck_source(
    T: xr.DataArray,
    nus: xr.DataArray,
    dnus: xr.DataArray,
) -> xr.DataArray:
    """Compute and annotate radiation source function."""
    source = (B_nu(T, nus) * dnus).rename("planck_source")
    source.attrs["units"] = "W m-2 sr-1"

    return source
