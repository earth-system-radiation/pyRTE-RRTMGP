"""Constants and functions for radiative transfer and gas optics calculations."""

from typing import Dict, Final

import numpy as np
from molmass import Formula
from numba import float64, vectorize
from scipy.constants import Boltzmann as k_B
from scipy.constants import h
from scipy.constants import speed_of_light as c

#
# Chemical formulae for gases known to RRTMGP that aren't clear from the name
#   Formulae from Wikipedia
#
halocarbons: Final[Dict[str, str]] = {
    "ccl4": "CCl4",
    "cfc11": "CCl3F",
    "cfc12": "CCl2F2",
    "cfc22": "CHClF2",
    "hfc23": "CHF3",
    "hfc32": "CH2F2",
    "hfc125": "C2HF5",
    "hfc143a": "C2H3F3",
}
#
# International standard atmosphere:
#  https://en.wikipedia.org/wiki/International_Standard_Atmosphere
#
M_DRY: Final[float64] = 28.966e-3


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


###
#
# Molar mass of gases
#
def get_molmass(gas_name: str) -> float64:
    """Molar mass in MKS units."""
    f = halocarbons[gas_name] if gas_name in halocarbons.keys() else gas_name.upper()
    return Formula(f).mass * 1.0e-3
