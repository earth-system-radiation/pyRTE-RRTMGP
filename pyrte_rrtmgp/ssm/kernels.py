"""SSM kernel functions: column gas amounts, optical depth, and Planck emission."""

import numpy as np
import numpy.typing as npt

from .defaults import BOLTZMANN_K, GRAV, LIGHTSPEED, M_DRY, PLANCK_H

# Radiation constants for Planck function with wavenumber in cm^-1.
# Derived by converting B(nu_m, T) [W/m^2/sr/m^-1] to B(nu, T) [W/m^2/sr/cm^-1]:
#   B = 2*h*c^2 * (100*nu)^3 / (exp(h*c*100*nu / (k*T)) - 1) * 100
#     = C1 * nu^3 / (exp(C2*nu/T) - 1)
_C1 = 2.0 * PLANCK_H * LIGHTSPEED**2 * 1e8   # W m^2 sr^-1 (normalised for cm^-1 grid)
_C2 = PLANCK_H * LIGHTSPEED * 100.0 / BOLTZMANN_K  # cm K  (hc/k in cm·K units)


def compute_col_gas(
    plev: npt.NDArray[np.float64],
    vmr: npt.NDArray[np.float64],
    mol_weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute column gas amounts for each atmospheric layer.

    Converts pressure-level boundaries and volume mixing ratios into column
    mass densities (kg/m²) using the hydrostatic relation col = Δp / g.

    Args:
        plev: Pressure at layer interfaces (ncol, nlay+1) [Pa]
        vmr: Volume mixing ratios (ncol, nlay, ngas) [mol/mol]
        mol_weights: Molecular weight for each gas (ngas,) [kg/mol]

    Returns:
        Column gas amounts (ncol, nlay, ngas) [kg/m²]
    """
    delta_p = np.abs(np.diff(plev, axis=-1))  # (ncol, nlay) [Pa]
    col_dry = delta_p / GRAV  # (ncol, nlay) [kg/m²]
    # mass mixing ratio = VMR * MW_gas / M_dry_air
    mass_mix = vmr * mol_weights[np.newaxis, np.newaxis, :] / M_DRY  # (ncol, nlay, ngas)
    return mass_mix * col_dry[:, :, np.newaxis]  # (ncol, nlay, ngas) [kg/m²]


def compute_tau(
    col_gas: npt.NDArray[np.float64],
    nus: npt.NDArray[np.float64],
    triangle_params: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute absorption optical depth using triangular spectral profiles.

    Each row of triangle_params defines one spectral triangle:
        [gas_index (1-based), kappa_0 (m²/kg), nu_0 (cm⁻¹), l (cm⁻¹)]

    The absorption cross-section at wavenumber nu is:
        kappa(nu) = kappa_0 * max(0, 1 - |nu - nu_0| / l)

    Multiple triangles for the same gas are summed.

    Args:
        col_gas: Column gas amounts (ncol, nlay, ngas) [kg/m²]
        nus: Wavenumber grid (nnu,) [cm⁻¹]
        triangle_params: Triangle spectral parameters (ntri, 4)

    Returns:
        Absorption optical depth (ncol, nlay, nnu) [dimensionless]
    """
    ncol, nlay, _ = col_gas.shape
    nnu = len(nus)
    tau = np.zeros((ncol, nlay, nnu), dtype=np.float64)

    for row in triangle_params:
        gas_idx = int(row[0]) - 1  # convert 1-based gas index to 0-based
        kappa_0, nu_0, l = row[1], row[2], row[3]
        # triangular cross-section shape (nnu,)
        kappa = kappa_0 * np.maximum(0.0, 1.0 - np.abs(nus - nu_0) / l)
        # accumulate: col_gas (ncol, nlay, 1) * kappa (1, 1, nnu)
        tau += col_gas[:, :, gas_idx : gas_idx + 1] * kappa[np.newaxis, np.newaxis, :]

    return tau


def compute_planck(
    temperature: npt.NDArray[np.float64],
    nus: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute Planck blackbody spectral radiance.

    Evaluates B(nu, T) = C1 * nu^3 / (exp(C2 * nu / T) - 1) where
    C1 and C2 are derived from fundamental constants (see module header).

    Args:
        temperature: Temperature array of arbitrary shape (...) [K]
        nus: Wavenumber grid (nnu,) [cm⁻¹]

    Returns:
        Planck spectral radiance with shape (..., nnu) [W m⁻² sr⁻¹ (cm⁻¹)⁻¹]
    """
    T = temperature[..., np.newaxis]  # (..., 1) — broadcasts over nnu
    return _C1 * nus**3 / (np.exp(_C2 * nus / T) - 1.0)
