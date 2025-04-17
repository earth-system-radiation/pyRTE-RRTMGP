"""Utility functions for pyRTE-RRTMGP."""

import os
from typing import Any

import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import AllSkyExampleFiles, RFMIPExampleFiles
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data


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


def create_gas_dataset(
    gas_values: dict[str, float], dims: dict[str, int]
) -> xr.Dataset:
    """Create a dataset with gas values and dimensions.

    Args:
        gas_values: Dictionary of gas values
        dims: Dictionary of dimensions

    Returns:
        xr.Dataset: Dataset with gas values and dimensions
    """
    ds = xr.Dataset()

    dim_names = list(dims.keys())
    coords = {k: np.arange(v) for k, v in dims.items()}

    # Convert each gas value to 2D array and add as variable
    for gas_name, value in gas_values.items():
        ds[gas_name] = xr.DataArray(value, dims=dim_names, coords=coords)

    return ds


def compute_profiles(sst: float, ncol: int, nlay: int) -> xr.Dataset:
    """Construct atmospheric profiles following the RCEMIP protocol.

    Computes vertical profiles of pressure, temperature, humidity, and ozone
    for radiative transfer calculations. Based on the RCEMIP:
    (Radiative-Convective Equilibrium Model Intercomparison Project) protocol.

    Args:
        sst: Surface temperature [K]
        ncol: Number of columns
        nlay: Number of vertical layers

    Returns:
        xr.Dataset: Dataset containing the following variables:
            - pres_layer: Pressure at layer centers [Pa]
            - temp_layer: Temperature at layer centers [K]
            - pres_level: Pressure at layer interfaces [Pa]
            - temp_level: Temperature at layer interfaces [K]
            - water_vapor: Water vapor mass mixing ratio [kg/kg]
            - ozone: Ozone mass mixing ratio [kg/kg]
            - surface_temperature: Temperature at surface [K]

    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if not isinstance(nlay, int) or nlay < 2:
        raise ValueError("nlay must be an integer >= 2")
    if not isinstance(ncol, int) or ncol < 1:
        raise ValueError("ncol must be a positive integer")
    if not isinstance(sst, (int, float)) or sst < 0:
        raise ValueError("sst must be a positive number")

    # Physical constants
    PHYS_CONSTANTS = {
        "z_trop": 15000.0,  # Tropopause height [m]
        "z_top": 70.0e3,  # Model top height [m]
        "g": 9.79764,  # Gravitational acceleration [m/s^2]
        "Rd": 287.04,  # Gas constant for dry air [J/kg/K]
        "p0": 101480.0,  # Surface pressure [Pa]
        "gamma": 6.7e-3,  # Lapse rate [K/m]
    }

    # Humidity parameters
    HUMIDITY_PARAMS = {
        "z_q1": 4.0e3,  # Scale height for water vapor [m]
        "z_q2": 7.5e3,  # Secondary scale height for water vapor [m]
        "q_t": 1.0e-8,  # Minimum specific humidity [kg/kg]
        "q_0": 0.01864,  # Surface specific humidity for 300K SST [kg/kg]
    }

    # Ozone parameters
    O3_PARAMS = {
        "g1": 3.6478,
        "g2": 0.83209,
        "g3": 11.3515,
        "o3_min": 1e-13,  # Minimum ozone mixing ratio [kg/kg]
    }

    # Constants
    z_trop = PHYS_CONSTANTS["z_trop"]
    z_top = PHYS_CONSTANTS["z_top"]
    g1 = O3_PARAMS["g1"]
    g2 = O3_PARAMS["g2"]
    g3 = O3_PARAMS["g3"]
    o3_min = O3_PARAMS["o3_min"]
    g = PHYS_CONSTANTS["g"]
    Rd = PHYS_CONSTANTS["Rd"]
    p0 = PHYS_CONSTANTS["p0"]
    z_q1 = HUMIDITY_PARAMS["z_q1"]
    z_q2 = HUMIDITY_PARAMS["z_q2"]
    q_t = HUMIDITY_PARAMS["q_t"]
    gamma = PHYS_CONSTANTS["gamma"]
    q_0 = HUMIDITY_PARAMS["q_0"]

    # Initial calculations
    Tv0 = (1.0 + 0.608 * q_0) * sst

    # Split resolution above and below RCE tropopause (15 km or about 125 hPa)
    z_lev = np.zeros(nlay + 1)
    z_lev[0] = 0.0
    z_lev[1 : nlay // 2 + 1] = 2.0 * z_trop / nlay * np.arange(1, nlay // 2 + 1)
    z_lev[nlay // 2 + 1 :] = z_trop + 2.0 * (z_top - z_trop) / nlay * np.arange(
        1, nlay // 2 + 1
    )
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])

    # Layer calculations with broadcasting
    z_lay_bc = z_lay[np.newaxis, :]
    z_lev_bc = z_lev[np.newaxis, :]

    q_lay = np.where(
        z_lay_bc > z_trop,
        q_t,
        q_0 * np.exp(-z_lay_bc / z_q1) * np.exp(-((z_lay_bc / z_q2) ** 2)),
    )
    t_lay = np.where(
        z_lay_bc > z_trop,
        sst - gamma * z_trop / (1.0 + 0.608 * q_0),
        sst - gamma * z_lay_bc / (1.0 + 0.608 * q_lay),
    )
    Tv_lay = (1.0 + 0.608 * q_lay) * t_lay
    p_lay = np.where(
        z_lay_bc > z_trop,
        p0
        * (Tv_lay / Tv0) ** (g / (Rd * gamma))
        * np.exp(-((g * (z_lay_bc - z_trop)) / (Rd * Tv_lay))),
        p0 * (Tv_lay / Tv0) ** (g / (Rd * gamma)),
    )

    p_hpa = p_lay / 100.0
    o3 = np.maximum(o3_min, g1 * p_hpa**g2 * np.exp(-p_hpa / g3) * 1.0e-6)

    # Level calculations with broadcasting
    q_lev = np.where(
        z_lev_bc > z_trop,
        q_t,
        q_0 * np.exp(-z_lev_bc / z_q1) * np.exp(-((z_lev_bc / z_q2) ** 2)),
    )
    t_lev = np.where(
        z_lev_bc > z_trop,
        sst - gamma * z_trop / (1.0 + 0.608 * q_0),
        sst - gamma * z_lev_bc / (1.0 + 0.608 * q_lev),
    )
    Tv_lev = (1.0 + 0.608 * q_lev) * t_lev
    p_lev = np.where(
        z_lev_bc > z_trop,
        p0
        * (Tv_lev / Tv0) ** (g / (Rd * gamma))
        * np.exp(-((g * (z_lev_bc - z_trop)) / (Rd * Tv_lev))),
        p0 * (Tv_lev / Tv0) ** (g / (Rd * gamma)),
    )

    t_sfc = np.repeat(sst, ncol)

    # Repeat profiles for each column
    p_lay = np.repeat(p_lay, ncol, axis=0)
    t_lay = np.repeat(t_lay, ncol, axis=0)
    q_lay = np.repeat(q_lay, ncol, axis=0)
    o3 = np.repeat(o3, ncol, axis=0)
    p_lev = np.repeat(p_lev, ncol, axis=0)
    t_lev = np.repeat(t_lev, ncol, axis=0)

    return xr.Dataset(
        data_vars={
            "pres_layer": (["site", "layer"], p_lay),
            "temp_layer": (["site", "layer"], t_lay),
            "pres_level": (["site", "level"], p_lev),
            "temp_level": (["site", "level"], t_lev),
            "surface_temperature": (["site"], t_sfc),
            "h2o": (["site", "layer"], q_lay),
            "o3": (["site", "layer"], o3),
        },
        attrs={
            "description": "Atmospheric profiles following RCEMIP protocol",
            "sst": sst,
            "ncol": ncol,
            "nlay": nlay,
        },
    )


def compute_clouds(
    cloud_optics: xr.Dataset, p_lay: xr.DataArray, t_lay: xr.DataArray
) -> xr.Dataset:
    """
    Compute cloud properties required for radiative transfer calculations.

    Using cloud optics data, atmospheric pressure and temperature layers,
    This function calculates cloud properties including:
      - liquid water path (lwp)
      - ice water path (iwp)
      - effective liquid water radius (rel)
      - effective ice radius (rei)

    The effective radii are computed as the average of the corresponding lower
    and upper bounds provided in the cloud_optics dataset.

    Args:
        cloud_optics : xr.Dataset
            Dataset containing cloud optical properties.
            It must include the following keys:
            - 'radliq_lwr': Lower bound for liquid water effective radius.
            - 'radliq_upr': Upper bound for liquid water effective radius.
            - 'diamice_lwr': Lower bound for ice effective radius.
            - 'diamice_upr': Upper bound for ice effective radius.
        p_lay : xr.DataArray
            Pressure levels of the atmospheric layers,
            where the dimension 'site' corresponds to spatial columns.
        t_lay : xr.DataArray
            Temperature values associated with the pressure layers.

    Returns:
        xr.Dataset
            A dataset containing the computed cloud properties
            with the following variables:
            - lwp: Liquid Water Path, assigned a value of 10.0
              in cloud regions where the temperature exceeds 263 K.
            - iwp: Ice Water Path, assigned a value of 10.0
              in cloud regions where the temperature is below 273 K.
            - rel: Effective liquid water radius, computed as the average of
              'radliq_lwr' and 'radliq_upr' in cloud regions.
            - rei: Effective ice radius, computed as the average
              of 'diamice_lwr' and 'diamice_upr' in cloud regions.
    """
    # Get dimensions from atmosphere
    ncol = p_lay.sizes["site"]

    # Get reference radii values
    rel_val = 0.5 * (cloud_optics["radliq_lwr"] + cloud_optics["radliq_upr"])
    rei_val = 0.5 * (cloud_optics["diamice_lwr"] + cloud_optics["diamice_upr"])

    # Create cloud mask - clouds between 100-900 hPa and in 2/3 of columns
    cloud_mask = (
        (p_lay > 100 * 100)
        & (p_lay < 900 * 100)
        & ((np.arange(ncol) + 1) % 3 != 0).reshape(-1, 1)
    )

    # Initialize arrays with zeros
    lwp = xr.zeros_like(p_lay)
    iwp = xr.zeros_like(p_lay)
    rel = xr.zeros_like(p_lay)
    rei = xr.zeros_like(p_lay)

    # Set values where clouds exist
    lwp = lwp.where(~(cloud_mask & (t_lay > 263)), 10.0)
    rel = rel.where(~(cloud_mask & (t_lay > 263)), rel_val)

    iwp = iwp.where(~(cloud_mask & (t_lay < 273)), 10.0)
    rei = rei.where(~(cloud_mask & (t_lay < 273)), rei_val)

    return xr.Dataset(
        {
            "lwp": lwp,
            "iwp": iwp,
            "rel": rel,
            "rei": rei,
        }
    )


def load_rrtmgp_file(file: AllSkyExampleFiles | RFMIPExampleFiles) -> xr.Dataset:
    """Load an RRTMGP file.

    Args:
        file: The file to load

    Returns:
        xr.Dataset: The loaded dataset
    """
    rte_rrtmgp_dir = download_rrtmgp_data()
    ref_path = os.path.join(rte_rrtmgp_dir, file.value)
    return xr.load_dataset(ref_path, decode_cf=False)


def safer_divide(a: Any, b: Any) -> Any:
    """Safer np.divide util func."""
    return np.divide(a, b, out=np.zeros_like(a), where=b > np.finfo(float).eps)
