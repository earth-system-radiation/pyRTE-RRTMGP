"""Default mappings for gas names, dimensions and variables used in RRTMGP.

This module contains dictionaries that map standard names to dataset-specific names
for gases, dimensions and variables used in radiative transfer calculations.
"""

from typing import Dict, Final

# Mapping of standard gas names to RRTMGP-specific names
DEFAULT_GAS_MAPPING: Final[Dict[str, list[str]]] = {
    "h2o": ["h2o", "water_vapor"],
    "co2": ["co2", "carbon_dioxide_GM"],
    "o3": ["o3", "ozone"],
    "n2o": ["n2o", "nitrous_oxide_GM"],
    "co": ["co", "carbon_monoxide_GM"],
    "ch4": ["ch4", "methane_GM"],
    "o2": ["o2", "oxygen_GM"],
    "n2": ["n2", "nitrogen_GM"],
    "ccl4": ["ccl4", "carbon_tetrachloride_GM"],
    "cfc11": ["cfc11", "cfc11_GM"],
    "cfc12": ["cfc12", "cfc12_GM"],
    "cfc22": ["cfc22", "hcfc22_GM"],
    "hfc143a": ["hfc143a", "hfc143a_GM"],
    "hfc125": ["hfc125", "hfc125_GM"],
    "hfc23": ["hfc23", "hfc23_GM"],
    "hfc32": ["hfc32", "hfc32_GM"],
    "hfc134a": ["hfc134a", "hfc134a_GM"],
    "cf4": ["cf4", "cf4_GM"],
    "no2": ["no2", "no2"],
}

# Mapping of standard dimension names to dataset-specific names
DEFAULT_DIM_MAPPING: Final[Dict[str, str]] = {
    "site": "site",
    "layer": "layer",
    "level": "level",
}

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
