from dataclasses import asdict, dataclass
from typing import Dict, Optional, Set

import xarray as xr

from pyrte_rrtmgp.config import DEFAULT_GAS_MAPPING


@dataclass
class GasMapping:
    _mapping: Dict[str, str]
    _required_gases: Set[str]

    @classmethod
    def create(
        cls, gas_names: Set[str], custom_mapping: Dict[str, str] | None = None
    ) -> "GasMapping":
        mapping = DEFAULT_GAS_MAPPING.copy()
        if custom_mapping:
            mapping.update(custom_mapping)

        return cls(mapping, gas_names)

    def validate(self) -> Dict[str, str]:
        """Validates and returns the final mapping."""
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
    """Container for dimension and variable mappings"""

    dim_mapping: Dict[str, str]
    var_mapping: Dict[str, str]

    def __post_init__(self):
        """Validate mappings upon initialization"""
        pass

    @classmethod
    def from_dict(cls, d: Dict) -> "DatasetMapping":
        """Create mapping from dictionary"""
        return cls(dim_mapping=d["dim_mapping"], var_mapping=d["var_mapping"])


@xr.register_dataset_accessor("mapping")
class DatasetMappingAccessor:
    """
    An accessor for xarray datasets that provides information about variable mappings.
    The mapping is stored in the dataset's attributes.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def set_mapping(self, mapping: DatasetMapping) -> None:
        """Set the mapping in dataset attributes"""
        # Validate that mapped variables exist in dataset
        missing_dims = set(mapping.dim_mapping.values()) - set(self._obj.dims)
        if missing_dims:
            raise ValueError(f"Dataset missing required dimensions: {missing_dims}")

        # Store mapping in attributes
        self._obj.attrs["dataset_mapping"] = asdict(mapping)

    @property
    def mapping(self) -> Optional[DatasetMapping]:
        """Get the mapping from dataset attributes"""
        if "dataset_mapping" not in self._obj.attrs:
            return None
        return DatasetMapping.from_dict(self._obj.attrs["dataset_mapping"])

    def get_var(self, standard_name: str) -> Optional[str]:
        """Get the actual variable name in the dataset for a standard name"""
        mapping = self.mapping
        if mapping is None:
            return None
        return mapping.var_mapping.get(standard_name)

    def get_dim(self, standard_name: str) -> Optional[str]:
        """Get the actual dimension name in the dataset for a standard name"""
        mapping = self.mapping
        if mapping is None:
            return None
        return mapping.dim_mapping.get(standard_name)


@dataclass
class AtmosphericMapping(DatasetMapping):
    """Specific mapping for atmospheric data"""

    def __post_init__(self):
        """Validate atmospheric-specific mappings"""
        required_dims = {"site", "layer", "level"}
        missing_dims = required_dims - set(self.dim_mapping.keys())
        if missing_dims:
            raise ValueError(f"Missing required dimensions in mapping: {missing_dims}")

        required_vars = {"pres_layer", "pres_level", "temp_layer", "temp_level"}
        missing_vars = required_vars - set(self.var_mapping.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables in mapping: {missing_vars}")


def create_default_mapping() -> AtmosphericMapping:
    """Create a default mapping configuration"""
    return AtmosphericMapping(
        dim_mapping={"site": "site", "layer": "layer", "level": "level"},
        var_mapping={
            "pres_layer": "pres_layer",
            "pres_level": "pres_level",
            "temp_layer": "temp_layer",
            "temp_level": "temp_level",
            "surface_temperature": "surface_temperature",
            "solar_zenith_angle": "solar_zenith_angle",
            "surface_albedo": "surface_albedo",
            "surface_albedo_dir": "surface_albedo_dir",
            "surface_albedo_dif": "surface_albedo_dif",
            "surface_emissivity": "surface_emissivity",
            "surface_emissivity_jacobian": "surface_emissivity_jacobian",
        },
    )
