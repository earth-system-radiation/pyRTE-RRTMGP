from typing import Iterable

import numpy as np
import numpy.typing as npt
import xarray as xr


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


def rfmip_2_col_gas(rfmip: xr.Dataset, gas_names: Iterable, dry_air: bool = False):
    """Convert RFMIP data to column gas concentrations.

    Args:
        rfmip (xr.Dataset): RFMIP data.
        gas_names (list): List of gas names.
        dry_air (bool, optional): Include dry air. Defaults to False.

    Returns:
        np.ndarray: Column gas concentrations.
    """
    if dry_air:
        raise NotImplementedError("Dry air is not implemented yet")

    ncol = len(rfmip["site"])
    nlay = len(rfmip["layer"])
    # TODO: need to add dry air
    col_gas = []
    for gas_name in gas_names:
        gas_values = rfmip[gas_name].values
        if gas_values.ndim == 0:
            gas_values = np.full((ncol, nlay), gas_values)
        col_gas.append(gas_values)
    return np.stack(col_gas, axis=-1)


def gpoint_flavor_from_kdist(kdist: xr.Dataset) -> npt.NDArray:
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
    """Extract gas names from the gas_names array, decoding and removing the suffix"""
    output = []
    for gas in gas_names:
        output.append(gas.tobytes().decode().strip().split("_")[0])
    return output

def get_idx_minor(gas_names, minor_gases):
    """Index of each minor gas in col_gas"""
    idx_minor_gas = []
    for gas in minor_gases:
        try:
            gas_idx = gas_names.index(gas) + 1
        except ValueError:
            gas_idx = -1
        idx_minor_gas.append(gas_idx)
    return idx_minor_gas
