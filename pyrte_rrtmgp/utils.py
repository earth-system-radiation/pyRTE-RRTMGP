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
    return np.unique(all_flav, axis=0).T


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
        gas_values = rfmip[gas_name].values
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
        dryair = get_col_dry(vmr_h2o, rfmip["pres_level"].values, latitude=None)
        col_gas = [dryair] + col_gas
    return np.stack(col_gas, axis=-1)


def gpoint_flavor_from_kdist(kdist: xr.Dataset) -> npt.NDArray:
    """Get the g-point flavors from the k-distribution file.

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
    return idx_minor_gas


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

    for ilev in range(nlev - 1):
        for icol in range(ncol):
            delta_plev = abs(plev[icol, ilev] - plev[icol, ilev + 1])
            fact = 1.0 / (1.0 + vmr_h2o[icol, ilev])
            m_air = (M_DRY + M_H2O * vmr_h2o[icol, ilev]) * fact
            col_dry[icol, ilev] = (
                10.0 * delta_plev * AVOGAD * fact / (1000.0 * m_air * 100.0 * g0[icol])
            )

    return col_dry


def calculate_mu0(solar_zenith_angle):
    """Calculate the cosine of the solar zenith angle

    Args:
        solar_zenith_angle (np.ndarray): Solar zenith angle in degrees
    """
    usecol_values = solar_zenith_angle < 90.0 - 2.0 * np.spacing(90.0)
    return np.where(usecol_values, np.cos(np.radians(solar_zenith_angle)), 1.0)
