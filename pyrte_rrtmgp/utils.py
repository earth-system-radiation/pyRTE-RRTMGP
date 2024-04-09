import numpy as np


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
