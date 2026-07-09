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

    absorption_coeffs = (
        kappa0 * np.exp(-abs(nus - nu0) / ell)
    ).rename("absorption_coeffs").assign_attrs({"units": "m2 kg-1"})

    return absorption_coeffs


def compute_layer_mass(
    vmr: xr.Dataset,
    plev: xr.DataArray,
    play: xr.DataArray,
    mol_weights: xr.DataArray,
    tags,
    species_by_tag,
    m_dry: float = 0.029,
) -> xr.DataArray:
    """
    Convert volume mixing ratios and pressure levels to gas layer masses.

    Parameters
    ----------
    vmr:
        Dataset containing one volume mixing ratio variable per species.
        ``tags`` and ``species_by_tag`` are used to build the tag-indexed VMR
        array with ``xr.concat``. Well-mixed species given as a single value
        per column are broadcast across the layer grid.

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

    vmr = (
        xr.concat(
            [
                vmr[str(species)].broadcast_like(play)
                for species in species_by_tag.values
            ],
            dim=xr.IndexVariable("tag", list(tags)),
        )
        .assign_coords(species=("tag", species_by_tag.values))
    )

    dp = abs(plev.diff(lev_dim)).rename({lev_dim: lay_dim})

    return (
        vmr
        * (mol_weights / m_dry)
        * dp
        / GRAV
    ).rename("layer_mass").assign_attrs({"units": "kg m-2"})


def compute_tau(
    absorption_coeffs: xr.DataArray,
    play: xr.DataArray,
    pref: float,
    layer_mass: xr.DataArray,
) -> xr.DataArray:
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
        Reference pressure in Pa.

    layer_mass:
        Gas layer mass with dimensions ``("tag", column_dim, layer_dim)``.
        Gas layer masses [kg m^-2] from compute_layer_mass()

    Returns
    -------
    xr.DataArray
        Optical depth with dimensions ``(column_dim, layer_dim, "gpt")``.
    """
    return (
        (play / pref) * xr.dot(layer_mass, absorption_coeffs, dim="tag")
    ).rename("tau")


#
# Underlying B_nu function is numba vectorized
#
def compute_planck_source(
    T: xr.DataArray,
    nus: xr.DataArray,
    dnus: xr.DataArray,
) -> xr.DataArray:
    """Compute and annotate radiation source function."""
    source = (B_nu(T, nus) * dnus).rename("planck_source")
    source.attrs["units"] = "W m-2 sr-1"

    return source
