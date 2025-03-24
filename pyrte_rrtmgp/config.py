"""Default mappings for gas names, dimensions and variables used in RRTMGP.

This module contains dictionaries that map standard names to dataset-specific names
for gases, dimensions and variables used in radiative transfer calculations.
"""

from typing import Dict, Final

# Mapping of standard gas names to RRTMGP-specific names
DEFAULT_GAS_MAPPING: Final[Dict[str, list[str]]] = {
    "h2o": ["h2o", "water_vapor", "water"],
    "co2": ["co2", "carbon_dioxide"],
    "o3": ["o3", "ozone"],
    "n2o": ["n2o", "nitrous_oxide"],
    "co": ["co", "carbon_monoxide"],
    "ch4": ["ch4", "methane"],
    "o2": ["o2", "oxygen"],
    "n2": ["n2", "nitrogen"],
    "ccl4": ["ccl4", "carbon_tetrachloride"],
    "cfc11": ["cfc11", "cfc-11", "freon-11"],
    "cfc12": ["cfc12", "cfc-12", "freon-12"],
    "cfc22": ["cfc22", "cfc-22", "freon-22"],
    "hfc143a": ["hfc143a", "hfc-143a"],
    "hfc125": ["hfc125", "hfc-125"],
    "hfc23": ["hfc23", "hfc-23"],
    "hfc32": ["hfc32", "hfc-32"],
    "hfc134a": ["hfc134a", "hfc-134a"],
    "cf4": ["cf4", "cf-4", "carbon_tetrafluoride"],
    "no2": ["no2", "nitrogen_dioxide"],
}
"""
A mapping of standard gas names to their corresponding RRTMGP-specific names.

Each key is a standard gas name (e.g., "h2o" for water vapor), and the value is
a list of alternative names used in the RRTMGP model for the same gas.
"""

# Mapping of standard dimension names to dataset-specific names
DEFAULT_DIM_MAPPING: Final[Dict[str, str]] = {
    "site": "site",
    "layer": "layer",
    "level": "level",
}
"""
A mapping of standard dimension names to their corresponding dataset-specific names.

Each key is a standard dimension name (e.g., "site"), and the value is the corresponding
name used in the dataset for that dimension.
"""

# Mapping of standard variable names to dataset-specific names
DEFAULT_VAR_MAPPING: Final[Dict[str, str]] = {
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
}
"""
A mapping of standard variable names to their corresponding dataset-specific names.

Each key is a standard variable name (e.g., "pres_layer" for pressure at layer
midpoints), and the value is the corresponding name used in the dataset for that
variable.
"""
