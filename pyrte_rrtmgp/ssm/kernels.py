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

def _as_layer_array(values: xr.DataArray, layer_dim: str) -> xr.DataArray:
    """Return values with its final dimension named like the layer grid."""
    if values.dims[-1] == layer_dim:
        return values

    return values.rename({values.dims[-1]: layer_dim})

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
    tags=None,
    species_by_tag=None,
    m_dry: float = 0.029,
) -> xr.DataArray:
    """
    Convert volume mixing ratios and pressure levels to gas layer masses.

    Parameters
    ----------
    vmr:
        Dataset containing one volume mixing ratio variable per species.
        ``tags`` and ``species_by_tag`` are used to build the tag-indexed VMR
        array with ``xr.concat``.

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

    if tags is None or species_by_tag is None:
        raise ValueError("tags and species_by_tag are required")

    vmr = (
        xr.concat(
            [
                _as_layer_array(vmr[str(species)], lay_dim)
                for species in species_by_tag.values
            ],
            dim=xr.IndexVariable("tag", list(tags)),
        )
        .assign_coords(species=("tag", species_by_tag.values))
    )

    dp = abs(plev.diff(lev_dim)).rename({lev_dim: lay_dim})

    if lay_dim in play.coords:
        dp = dp.assign_coords({lay_dim: play[lay_dim]})

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
        Reference pressure in Pa. If zero, pressure scaling is set to one.

    layer_mass:
        Gas layer mass with dimensions ``("tag", column_dim, layer_dim)``.
        Gas layer masses [kg m^-2] from compute_layer_mass()

    Returns
    -------
    xr.DataArray
        Optical depth with dimensions ``(column_dim, layer_dim, "gpt")``.
    """
    return (
        (play / pref if pref != 0.0 else xr.ones_like(play))
        * xr.dot(layer_mass, absorption_coeffs, dims="tag")
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
