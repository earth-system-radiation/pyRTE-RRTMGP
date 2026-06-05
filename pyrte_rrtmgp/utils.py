"""Constants and functions for radiative transfer and gas optics calculations."""

import numpy as np
from numba import float64, vectorize
from scipy.constants import Boltzmann as k_B
from scipy.constants import h
from scipy.constants import speed_of_light as c


#######################
#
# Planck function and its inverse.
#
@vectorize([float64(float64, float64)])
def B_nu(T: float64, nu: float64) -> float64:
    """
    Planck function.

    In:
    T [K]: temperature
    nu [cm-1]: multiply all nu's by 100 to convert to m-1 in formula, then multiply
               B by 100 for units of W/m^2/sr/cm-1

    Out:
        Planck function in units of W/m^2/sr/cm-1
    """
    nu = nu * 100
    return ((2 * h * (c**2) * (nu**3)) / (np.exp((h * c * nu) / (k_B * T)) - 1)) * 100


@vectorize([float64(float64, float64)])
def Tb_nu(B: float64, nu: float64) -> float64:
    """
    Inverse Planck function.

    In:
    B [W/m^2/sr/cm-1]: Spectral flux
    nu [cm-1]: multiply all nu's by 100 to convert to m-1 in formula, then multiply
               B by 100 for units of W/m^2/sr/cm-1

    Out:
        Brightness temperature [K]
    """
    nu = nu * 100
    return (h * c * nu) / k_B * 1 / np.log((2 * h * (c**2) * nu**3) * 100.0 / B + 1)
