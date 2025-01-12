from dataclasses import asdict, dataclass
from typing import Dict, Optional, Set

import xarray as xr

from pyrte_rrtmgp.config import (
    DEFAULT_DIM_MAPPING,
    DEFAULT_GAS_MAPPING,
    DEFAULT_VAR_MAPPING,
)


@dataclass
class GasMapping:
    """Class for managing gas name mappings between standard and dataset-specific names.

    Attributes:
        _mapping: Dictionary mapping standard gas names to dataset-specific names
        _required_gases: Set of required gas names that must be present
    """

    _mapping: Dict[str, str]
    _required_gases: Set[str]

    @classmethod
    def create(
        cls, gas_names: Set[str], custom_mapping: Optional[Dict[str, str]] = None
    ) -> "GasMapping":
        """Create a new GasMapping instance with default and custom mappings.

        Args:
            gas_names: Set of required gas names
            custom_mapping: Optional custom mapping to override defaults

        Returns:
            New GasMapping instance
        """
        mapping = DEFAULT_GAS_MAPPING.copy()
        if custom_mapping:
            mapping.update(custom_mapping)

        return cls(mapping, gas_names)

    def validate(self) -> Dict[str, str]:
        """Validate and return the final gas name mapping.

        Returns:
            Dictionary mapping standard gas names to dataset-specific names

        Raises:
            ValueError: If a required gas is not found in any mapping
        """
        validated_mapping = {}

        for gas in self._required_gases:
            if gas not in self._mapping:
                if gas not in DEFAULT_GAS_MAPPING:
                    raise ValueError(f"Gas {gas} not found in any mapping")
                validated_mapping[gas] = DEFAULT_GAS_MAPPING[gas]
            else:
                validated_mapping[gas] = self._mapping[gas]

        return validated_mapping


@dataclass
class DatasetMapping:
    """Container for dimension and variable name mappings.

    Attributes:
        dim_mapping: Dictionary mapping standard dimension names to dataset-specific names
        var_mapping: Dictionary mapping standard variable names to dataset-specific names
    """

    dim_mapping: Dict[str, str]
    var_mapping: Dict[str, str]

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


@xr.register_dataset_accessor("mapping")
class DatasetMappingAccessor:
    """Accessor for xarray datasets that provides variable mapping functionality.

    The mapping is stored in the dataset's attributes to maintain persistence.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    def set_mapping(self, mapping: DatasetMapping) -> None:
        """Set the mapping in dataset attributes.

        Args:
            mapping: DatasetMapping instance to store

        Raises:
            ValueError: If mapped dimensions don't exist in dataset
        """
        missing_dims = set(mapping.dim_mapping.values()) - set(self._obj.dims)
        if missing_dims:
            raise ValueError(f"Dataset missing required dimensions: {missing_dims}")

        self._obj.attrs["dataset_mapping"] = asdict(mapping)

    @property
    def mapping(self) -> Optional[DatasetMapping]:
        """Get the mapping from dataset attributes.

        Returns:
            DatasetMapping if exists, None otherwise
        """
        if "dataset_mapping" not in self._obj.attrs:
            return None
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
