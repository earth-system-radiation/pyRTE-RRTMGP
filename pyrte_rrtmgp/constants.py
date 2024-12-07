"""Physical and mathematical constants used in radiative transfer calculations.

This module contains various physical and mathematical constants needed for
radiative transfer calculations, including gravitational parameters, molecular
masses, and Gaussian quadrature weights and points.
"""

from typing import Dict, Final
import numpy as np
from numpy.typing import NDArray

# Gravitational parameters from Helmert's equation (m/s^2)
HELMERT1: Final[float] = 9.80665  # Standard gravity at sea level
HELMERT2: Final[float] = 0.02586  # Gravity variation with latitude

# Molecular masses (kg/mol)
M_DRY: Final[float] = 0.028964  # Dry air
M_H2O: Final[float] = 0.018016  # Water vapor

# Avogadro's number (molecules/mol)
AVOGAD: Final[float] = 6.02214076e23

# Solar constants for orbit calculations
SOLAR_CONSTANTS: Final[Dict[str, float]] = {
    "A_OFFSET": 0.1495954,  # Semi-major axis offset (AU)
    "B_OFFSET": 0.00066696,  # Orbital eccentricity factor
}

# Gaussian quadrature constants for radiative transfer
GAUSS_DS: NDArray[np.float64] = np.reciprocal(
    np.array(
        [
            [0.6096748751, np.inf, np.inf, np.inf],
            [0.2509907356, 0.7908473988, np.inf, np.inf],
            [0.1024922169, 0.4417960320, 0.8633751621, np.inf],
            [0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973],
        ]
    )
)

GAUSS_WTS: NDArray[np.float64] = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.2300253764, 0.7699746236, 0.0, 0.0],
        [0.0437820218, 0.3875796738, 0.5686383044, 0.0],
        [0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087],
    ]
)
