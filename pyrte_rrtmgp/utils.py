"""Utility functions for pyRTE-RRTMGP."""

from typing import Any

import numpy as np
import xarray as xr


def expand_variable_dims(
    dataset: xr.Dataset, var_name: str, needed_dims: list
) -> xr.Dataset:
    """Expand dimensions of a variable in the dataset if needed.

    Args:
        dataset: Dataset containing the variable to expand
        var_name: Name of the variable to expand
        needed_dims: List of dimensions that the variable should have

    Returns:
        Dataset with the variable expanded to include needed dimensions
    """
    original_dims = dataset[var_name].dims

    # Create dictionary of dimensions to expand with their coordinates
    expand_dims = {}
    for dim in needed_dims:
        if dim not in original_dims and dim in dataset.dims:
            expand_dims[dim] = dataset[dim]

    # Expand dimensions with proper coordinates
    if expand_dims:
        expanded_var = dataset[var_name].expand_dims(expand_dims)
        dataset = dataset.drop_vars(var_name)
        dataset[var_name] = expanded_var

    return dataset


def safer_divide(a: Any, b: Any) -> Any:
    """Safer np.divide util func."""
    return np.divide(a, b, out=np.zeros_like(a), where=b > np.finfo(float).eps)
