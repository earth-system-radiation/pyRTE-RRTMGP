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
