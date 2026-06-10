"""
Core xarray-based numerical kernels for the Simple Spectral Model (SSM).

References
----------
Williams, A. I. L. (2026). Bridging clarity and accuracy: A simple spectral
longwave radiation scheme for idealized climate modeling.
Journal of Advances in Modeling Earth Systems, 18, e2025MS005405.
https://doi.org/10.1029/2025MS005405

"""

import numpy as np
import xarray as xr

from defaults import PLANCK_H, LIGHTSPEED, BOLTZMANN_K, GRAV

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
def compute_absorption_coeffs(
    triangles: xr.DataArray,
    nus: xr.DataArray,
) -> xr.DataArray:
    
    nu0 = triangles.sel(param="nu0")
    ell = triangles.sel(param="l")
    kappa0 = triangles.sel(param="kappa0")

    absorption_coeffs = kappa0 * np.exp(-abs(nus - nu0) / ell)
    absorption_coeffs = absorption_coeffs.rename("absorption_coeffs")
    absorption_coeffs.attrs["units"] = "m2 kg-1"

    return absorption_coeffs

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
        Layer mass with dims ("tag", column_dim, layer_dim). Mass of each gas in each layer [kg m^-2]
"""

def compute_layer_mass(
    vmr: xr.DataArray,
    plev: xr.DataArray,
    play: xr.DataArray,
    mol_weights: xr.DataArray,
    m_dry: float = 0.029,
) -> xr.DataArray:
   
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
    
    if pref != 0.0:
        p_scaling = play / pref
    else:
        p_scaling = xr.ones_like(play)

    tau = p_scaling * (layer_mass * absorption_coeffs).sum("tag")
    tau = tau.rename("tau")

    return tau


"""
planck_function() calculates how much radiation a perfect blackbody emits at wavelength nu given temperature T, 
per unit wavelength interval, per unit solid angle. In other words, the spectral radiance of a blackbody per unit wavenumber:
B_nu(T, nu) = 100 * 2*h*c^2 * (100*nu)^3 
                  / [exp(h*c*100*nu / (k_B * T)) - 1]

    Parameters
    ----------
    T:
        Temperature in K. May have any atmospheric dimensions.

    nu:
        Wavenumber grid in cm^-1, typically with dimension ``"gpt"``.

    Returns
    -------
    xr.DataArray
        Spectral radiance [ W m^-2 sr^-1 (cm^-1)^-1]
        Spectral radiance broadcast over the dimensions of ``T`` and ``nu``.
"""

def planck_function(
    T: xr.DataArray,
    nu: xr.DataArray,
) -> xr.DataArray:
    
    nu_si = nu / 100.0
    numerator = 100.0 * 2.0 * PLANCK_H * (nu_si ** 3) * (LIGHTSPEED ** 2)
    exponent = (PLANCK_H * LIGHTSPEED * nu_si) / (BOLTZMANN_K * T)
    return numerator / (np.exp(exponent) - 1.0)

"""
    compute_planck_source() computes the spectrally integrated Planck source function.
    For each wavenumber band, the source is:
      source(..., nu) = B_nu(T, nu) * dnu

      where dnu is the width of the spectral band. This is the band-integrated radiance, or the total emission in the wavenumber interval


    The source is evaluated as ``planck_function(T, nus) * dnus``. xarray
    broadcasting expands atmospheric temperature fields over the ``"gpt"``
    spectral dimension.

    Parameters
    ----------
    T:  Temperature [K]
        Temperature field in K.

    nus:Wavenumber at each spectral point [cm^-1]
        Wavenumber grid with dimension ``"gpt"``.

    dnus:Width of each spectral band [cm^-1]
         Spectral band widths with dimension ``"gpt"``.

    Returns
    -------
    xr.DataArray
        Band-integrated Planck source with the dimensions of ``T`` plus
        ``"gpt"``.
        Band-integrated Planck radiance [W m^-2 sr^-1]   
    """

def compute_planck_source(
    T: xr.DataArray,
    nus: xr.DataArray,
    dnus: xr.DataArray,
) -> xr.DataArray:
    
    source = planck_function(T, nus) * dnus
    source = source.rename("planck_source")
    source.attrs["units"] = "W m-2 sr-1"

    return source
