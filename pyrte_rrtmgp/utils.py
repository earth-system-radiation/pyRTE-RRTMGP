from typing import List

import numpy as np
import numpy.typing as npt
import xarray as xr

# Constants
HELMERT1 = 9.80665
HELMERT2 = 0.02586
M_DRY = 0.028964
M_H2O = 0.018016
AVOGAD = 6.02214076e23


def flavors_from_kdist(kdist: xr.Dataset) -> npt.NDArray:
    """Get the unique flavors from the k-distribution file.

    Args:
        kdist (xr.Dataset): K-distribution file.

    Returns:
        np.ndarray: Unique flavors.
    """
    key_species = kdist["key_species"].values
    tot_flav = len(kdist["bnd"]) * len(kdist["atmos_layer"])
    npairs = len(kdist["pair"])
    all_flav = np.reshape(key_species, (tot_flav, npairs))
    # (0,0) becomes (2,2) because absorption coefficients for these g-points will be 0.
    all_flav[np.all(all_flav == [0, 0], axis=1)] = [2, 2]
    # we do that instead of unique to preserv the order
    _, idx = np.unique(all_flav, axis=0, return_index=True)
    return all_flav[np.sort(idx)].T


def get_col_dry(vmr_h2o, plev, latitude=None):
    """Calculate the dry column of the atmosphere

    Args:
        vmr_h2o (np.ndarray): Water vapor volume mixing ratio
        plev (np.ndarray): Pressure levels
        latitude (np.ndarray): Latitude of the location

    Returns:
        np.ndarray: Dry column of the atmosphere
    """
    ncol = plev.shape[0]
    nlev = plev.shape[1]
    col_dry = np.zeros((ncol, nlev - 1))

    if latitude is not None:
        g0 = HELMERT1 - HELMERT2 * np.cos(2.0 * np.pi * latitude / 180.0)
    else:
        g0 = np.full(ncol, HELMERT1)  # Assuming grav is a constant value

    # TODO: use numpy instead of loops
    for ilev in range(nlev - 1):
        for icol in range(ncol):
            delta_plev = abs(plev[icol, ilev] - plev[icol, ilev + 1])
            fact = 1.0 / (1.0 + vmr_h2o[icol, ilev])
            m_air = (M_DRY + M_H2O * vmr_h2o[icol, ilev]) * fact
            col_dry[icol, ilev] = (
                10.0 * delta_plev * AVOGAD * fact / (1000.0 * m_air * 100.0 * g0[icol])
            )
    return col_dry


def rfmip_2_col_gas(rfmip: xr.Dataset, gas_names: List[str], dry_air: bool = False):
    """Convert RFMIP data to column gas concentrations.

    Args:
        rfmip (xr.Dataset): RFMIP data.
        gas_names (list): List of gas names.
        dry_air (bool, optional): Include dry air. Defaults to False.

    Returns:
        np.ndarray: Column gas concentrations.
    """

    ncol = len(rfmip["site"])
    nlay = len(rfmip["layer"])
    col_gas = []
    for gas_name in gas_names:
        # if gas_name is not available, fill it with zeros
        if gas_name not in rfmip.data_vars.keys():
            gas_values = np.zeros((ncol, nlay))
        else:
            try:
                scale = float(rfmip[gas_name].units)
            except AttributeError:
                scale = 1.0
            gas_values = rfmip[gas_name].values * scale

        if gas_values.ndim == 0:
            gas_values = np.full((ncol, nlay), gas_values)
        col_gas.append(gas_values)

    if dry_air:
        if "h2o" not in gas_names and "water_vapor" not in gas_names:
            raise ValueError(
                "h2o gas must be included in gas_names to calculate dry air"
            )
        if "h2o" in gas_names:
            h2o_idx = gas_names.index("h2o")
        else:
            h2o_idx = gas_names.index("water_vapor")
        vmr_h2o = col_gas[h2o_idx]
        col_dry = get_col_dry(vmr_h2o, rfmip["pres_level"].values, latitude=None)
        col_gas = [col_dry] + col_gas

    col_gas = np.stack(col_gas, axis=-1).astype(np.float64)
    col_gas[:, :, 1:] = col_gas[:, :, 1:] * col_gas[:, :, :1]

    return col_gas


def gpoint_flavor_from_kdist(kdist: xr.Dataset) -> npt.NDArray:
    """Get the g-point flavors from the k-distribution file.

    Each g-point is associated with a flavor, which is a pair of key species.

    Args:
        kdist (xr.Dataset): K-distribution file.

    Returns:
        np.ndarray: G-point flavors.
    """
    key_species = kdist["key_species"].values
    flavors = flavors_from_kdist(kdist)

    band_ranges = [
        [i] * (r.values[1] - r.values[0] + 1)
        for i, r in enumerate(kdist["bnd_limits_gpt"], 1)
    ]
    gpoint_bands = np.concatenate(band_ranges)

    key_species_rep = key_species.copy()
    key_species_rep[np.all(key_species_rep == [0, 0], axis=2)] = [2, 2]

    # unique flavors
    flist = flavors.T.tolist()

    def key_species_pair2flavor(key_species_pair):
        return flist.index(key_species_pair.tolist()) + 1

    flavors_bands = np.apply_along_axis(
        key_species_pair2flavor, 2, key_species_rep
    ).tolist()
    gpoint_flavor = np.array([flavors_bands[gp - 1] for gp in gpoint_bands]).T

    return gpoint_flavor


def extract_gas_names(gas_names):
    """Extract gas names from the gas_names array, decoding and removing the suffix

    Args:
        gas_names (np.ndarray): Gas names

    Returns:
        list: List of gas names
    """
    output = []
    for gas in gas_names:
        output.append(gas.tobytes().decode().strip().split("_")[0])
    return output


def get_idx_minor(gas_names, minor_gases):
    """Index of each minor gas in col_gas

    Args:
        gas_names (list): Gas names
        minor_gases (list): List of minor gases

    Returns:
        list: Index of each minor gas in col_gas
    """
    idx_minor_gas = []
    for gas in minor_gases:
        try:
            gas_idx = gas_names.index(gas) + 1
        except ValueError:
            gas_idx = -1
        idx_minor_gas.append(gas_idx)
    return np.array(idx_minor_gas, dtype=np.int32)


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


def krayl_from_kdist(kdist: xr.Dataset) -> npt.NDArray:
    """Get the Rayleigh scattering coefficients from the k-distribution file.

    Args:
        kdist (xr.Dataset): K-distribution file.

    Returns:
        np.ndarray: Rayleigh scattering coefficients.
    """
    return np.stack([kdist["rayl_lower"].values, kdist["rayl_upper"].values], axis=-1)


def combine_abs_and_rayleigh(tau_absorption, tau_rayleigh):
    """Combine absorption and Rayleigh scattering optical depths.

    Args:
        tau_absorption (np.ndarray): Absorption optical depth.
        tau_rayleigh (np.ndarray): Rayleigh scattering optical depth.

    Returns:
        np.ndarray: Combined optical depth.
    """

    tau = tau_absorption + tau_rayleigh
    ssa = np.where(tau > 2.0 * np.finfo(float).tiny, tau_rayleigh / tau, 0.0)
    g = np.zeros(tau.shape)

    return tau, ssa, g
