import numpy as np
import xarray as xr


def get_usecols(solar_zenith_angle):
    """Get the usecols values

    Args:
        solar_zenith_angle (np.ndarray): Solar zenith angle in degrees

    Returns:
        np.ndarray: Usecols values
    """
    return solar_zenith_angle < 90.0 - 2.0 * np.spacing(90.0)


def compute_mu0(solar_zenith_angle, nlayer=None):
    """Calculate the cosine of the solar zenith angle

    Args:
        solar_zenith_angle (np.ndarray): Solar zenith angle in degrees
        nlayer (int, optional): Number of layers. Defaults to None.
    """
    usecol_values = get_usecols(solar_zenith_angle)
    mu0 = np.where(usecol_values, np.cos(np.radians(solar_zenith_angle)), 1.0)
    if nlayer is not None:
        mu0 = np.stack([mu0] * nlayer).T
    return mu0


def compute_toa_flux(total_solar_irradiance, solar_source):
    """Compute the top of atmosphere flux

    Args:
        total_solar_irradiance (np.ndarray): Total solar irradiance
        solar_source (np.ndarray): Solar source

    Returns:
        np.ndarray: Top of atmosphere flux
    """
    ncol = total_solar_irradiance.shape[0]
    toa_flux = np.stack([solar_source] * ncol)
    def_tsi = toa_flux.sum(axis=1)
    return (toa_flux.T * (total_solar_irradiance / def_tsi)).T


def convert_xarray_args(func):
    def wrapper(*args, **kwargs):
        output_args = []
        for x in args:
            if isinstance(x, xr.DataArray):
                output_args.append(x.data)
            else:
                output_args.append(x)
        for k, v in kwargs.items():
            if isinstance(v, xr.DataArray):
                kwargs[k] = v.data
            else:
                kwargs[k] = v
        return func(*output_args, **kwargs)

    return wrapper
