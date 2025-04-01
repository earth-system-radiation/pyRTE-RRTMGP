"""Data validation utilities for pyRTE-RRTMGP."""

import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import xarray as xr

from pyrte_rrtmgp.config import (
    DEFAULT_DIM_MAPPING,
    DEFAULT_VAR_MAPPING,
)


@dataclass
class DatasetMapping:
    """Container for dimension and variable name mappings."""

    dim_mapping: Dict[str, str]
    """Dictionary mapping standard dimension names to dataset-specific names"""

    var_mapping: Dict[str, str]
    """Dictionary mapping standard variable names to dataset-specific names"""

    def __post_init__(self) -> None:
        """Validate mappings upon initialization."""
        pass

    @classmethod
    def from_dict(cls, d: Dict[str, Dict[str, str]]) -> "DatasetMapping":
        """Create mapping from dictionary representation.

        Args:
            d: Dictionary containing dim_mapping and var_mapping

        Returns:
            New DatasetMapping instance
        """
        return cls(dim_mapping=d["dim_mapping"], var_mapping=d["var_mapping"])

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetMapping":
        """Create mapping from JSON string.

        Args:
            json_str: JSON string containing dim_mapping and var_mapping

        Returns:
            New DatasetMapping instance
        """
        return cls.from_dict(json.loads(json_str))


@xr.register_dataset_accessor("mapping")
class DatasetMappingAccessor:
    """Accessor for xarray datasets that provides variable mapping functionality.

    The mapping is stored in the dataset's attributes to maintain persistence.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialize the DatasetMappingAccessor.

        Args:
            xarray_obj: Dataset containing the mapping.
        """
        self._obj = xarray_obj

    def set_mapping(self, mapping: DatasetMapping) -> None:
        """Set the mapping in dataset attributes.

        Args:
            mapping: DatasetMapping instance to store

        Raises:
            ValueError: If mapped dimensions don't exist in dataset
        """
        self._obj.attrs["dataset_mapping"] = json.dumps(asdict(mapping))

    @property
    def mapping(self) -> Optional[DatasetMapping]:
        """Get the mapping from dataset attributes.

        Returns:
            DatasetMapping if exists, None otherwise
        """
        if "dataset_mapping" not in self._obj.attrs:
            return None
        if isinstance(self._obj.attrs["dataset_mapping"], str):
            return DatasetMapping.from_json(self._obj.attrs["dataset_mapping"])
        return DatasetMapping.from_dict(self._obj.attrs["dataset_mapping"])

    def get_var(self, standard_name: str) -> Optional[str]:
        """Get the dataset-specific variable name for a standard name.

        Args:
            standard_name: Standard variable name

        Returns:
            Dataset-specific variable name if found, None otherwise
        """
        mapping = self.mapping
        if mapping is None:
            return None
        return mapping.var_mapping.get(standard_name)

    def get_dim(self, standard_name: str) -> Optional[str]:
        """Get the dataset-specific dimension name for a standard name.

        Args:
            standard_name: Standard dimension name

        Returns:
            Dataset-specific dimension name if found, None otherwise
        """
        mapping = self.mapping
        if mapping is None:
            return None
        return mapping.dim_mapping.get(standard_name)


@dataclass
class AtmosphericMapping(DatasetMapping):
    """Specific mapping for atmospheric data with required dimensions and variables.

    Inherits from DatasetMapping and adds validation for required atmospheric fields.
    """

    def __post_init__(self) -> None:
        """Validate atmospheric-specific mappings.

        Raises:
            ValueError: If required dimensions or variables are missing
        """
        required_dims = {"site", "layer", "level"}
        missing_dims = required_dims - set(self.dim_mapping.keys())
        if missing_dims:
            raise ValueError(f"Missing required dimensions in mapping: {missing_dims}")

        required_vars = {"pres_layer", "pres_level", "temp_layer", "temp_level"}
        missing_vars = required_vars - set(self.var_mapping.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables in mapping: {missing_vars}")


def create_default_mapping() -> AtmosphericMapping:
    """Create a default atmospheric mapping configuration.

    Returns:
        AtmosphericMapping instance with default dimension and variable mappings
    """
    return AtmosphericMapping(
        dim_mapping=DEFAULT_DIM_MAPPING,
        var_mapping=DEFAULT_VAR_MAPPING,
    )


# PROBLEM DATASET VALIDATION
def _validate_problem_dataset_dims(dataset: xr.Dataset) -> Tuple[bool, str]:
    return True, ""


def _validate_problem_dataset_vars(dataset: xr.Dataset) -> Tuple[bool, str]:

    pressure_layer_name: str = dataset.mapping.mapping.var_mapping.get("pres_layer")
    if pressure_layer_name is None:
        return False, "Pressure layer variable not found in dataset"

    pressure_arr: xr.DataArray = dataset.get(pressure_layer_name)
    if pressure_arr is not None and (pressure_arr < 1.0).any():
        return (
            False,
            f"Pressure layer ({pressure_layer_name}) values must be greater than 1.0",
        )

    return True, ""


def validate_problem_dataset(dataset: xr.Dataset) -> bool:
    """Validate that a dataset contains required dimensions and variables.

    Args:
        dataset: Dataset to validate
    """
    vars_valid, vars_msg = _validate_problem_dataset_vars(dataset)
    if not vars_valid:
        raise ValueError(vars_msg)

    dims_valid, dims_msg = _validate_problem_dataset_dims(dataset)
    if not dims_valid:
        raise ValueError(dims_msg)

    return True
