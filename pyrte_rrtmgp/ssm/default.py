"""Physical constants and default for simple spectral model."""

# Default constants and triangle parameters from mo_optics_ssm.F90
# Each triangle is defined by (kappa_0, nu_0, l)
from typing import Final

import numpy as np
from scipy.constants import g

from .. import utils

GRAV: Final[float] = g  # Gravitational acceleration [m/s^2]

TSUN_SSM = 5760.0  # default sun temeprature for SSM (K)
TSI = 1360.0  # default total solar irradiance  (W/m^2)

# Molecular weights (kg/mol)
MW_H2O: Final[float] = utils.get_molmass("H2O")
MW_CO2: Final[float] = utils.get_molmass("CO2")
MW_O3: Final[float] = utils.get_molmass("O3")
M_DRY: Final[float] = utils.M_DRY  # dry air

# default cloud absorption coefficients (m2/kg)
KAPPA_CLD_LW = 50.0
KAPPA_CLD_SW = 0.0001

# default cloud single scattering albedo
SSA_CLD_LW = 0.0
SSA_CLD_SW = 0.9999

# default for cloud asymmetry
G_CLD_LW = 0.0
G_CLD_SW = 0.85

# default nnu
NNU_DEF = 41

# wavelength ranges (cm^-1)
NU_MIN_LW_DEF = 0.0
NU_MAX_LW_DEF = 3500.0

NU_MIN_SW_DEF = 0.0
NU_MAX_SW_DEF = 50000.0

# default wavenumber arrays
# np.linspace reproduces the Fortran array constructor exactly
NUS_LW_DEF = np.linspace(50.0, 3000.0, NNU_DEF)
NUS_SW_DEF = np.linspace(1000.0, 45000.0, NNU_DEF)


def band_widths(
    nus: np.ndarray,
    nu_min: float = NU_MIN_LW_DEF,
    nu_max: float = NU_MAX_LW_DEF,
) -> np.ndarray:
    """
    Spectral width to credit to each wavenumber in ``nus`` [cm^-1].

    Reproduces the Fortran band-limit rule of mo_optics_ssm.F90: band edges sit
    midway between neighbouring ``nus``, with the outermost edges clamped to
    ``nu_min`` and ``nu_max``. The widths sum exactly to ``nu_max - nu_min``.

    Parameters
    ----------
    nus:
        Wavenumber grid points [cm^-1], strictly increasing.

    nu_min, nu_max:
        Outer edges of the spectral range [cm^-1].

    Returns
    -------
    np.ndarray
        Band widths [cm^-1], same length as ``nus``.
    """
    mids = 0.5 * (nus[:-1] + nus[1:])
    return np.diff(np.concatenate(([nu_min], mids, [nu_max])))


# default band widths matching NUS_LW_DEF
DNUS_LW_DEF = band_widths(NUS_LW_DEF)

# default spectroscopic params
# shape is (3 triangles, 4 parameters) - same as Fortran shape=[3,4]
# columns: [gas_index, kappa_0, nu_0, l]
TRIANGLE_PARAMS_DEF_LW = np.array(
    [[1.0, 282.0, 0.0, 64.0], [1.0, 24.0, 1600.0, 52.0], [2.0, 110.0, 667.0, 12.0]]
)

GAS_NAMES_DEF_LW = ["h2o", "co2"]

# shape is (2 triangles, 4 parameters)
# columns: [gas_index, kappa_0, nu_0, l]
TRIANGLE_PARAMS_DEF_SW = np.array(
    [[1.0, 1.0, 0.0, 1200.0], [2.0, 0.0, 0.0, 1000000.0]]  # h2o  # o3
)

GAS_NAMES_DEF_SW = ["h2o", "o3"]

P_REF = 50000.0  # reference pressure [Pa]

MOL_WEIGHTS = {
    "h2o": MW_H2O,
    "co2": MW_CO2,
    "o3": MW_O3,
}
