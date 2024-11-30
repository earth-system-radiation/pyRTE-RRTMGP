import os
import sys
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt
import xarray as xr

from pyrte_rrtmgp.constants import (
    AVOGAD,
    HELMERT1,
    HELMERT2,
    M_DRY,
    M_H2O,
    SOLAR_CONSTANTS,
)
from pyrte_rrtmgp.data_types import GasOpticsFiles, ProblemTypes
from pyrte_rrtmgp.kernels.rrtmgp import (
    compute_planck_source,
    compute_tau_absorption,
    compute_tau_rayleigh,
    interpolation,
)
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data
from pyrte_rrtmgp import atmospheric_data
from pyrte_rrtmgp.atmospheric_data import AtmosphericMapping, create_default_mapping


def load_gas_optics(
    file_path: str | None = None,
    gas_optics_file: GasOpticsFiles | None = None,
    selected_gases=None,
) -> xr.Dataset:
    """Load gas optics data from a file or predefined gas optics file.

    Args:
        file_path: Path to custom gas optics netCDF file
        gas_optics_file: Predefined gas optics file enum
        selected_gases: List of gases to include

    Returns:
        xarray Dataset containing the gas optics data
    """
    if file_path is not None:
        dataset = xr.load_dataset(file_path)
    elif gas_optics_file is not None:
        rte_rrtmgp_dir = download_rrtmgp_data()
        dataset = xr.load_dataset(os.path.join(rte_rrtmgp_dir, gas_optics_file.value))
    else:
        raise ValueError("Either file_path or gas_optics_file must be provided")

    dataset.attrs["selected_gases"] = selected_gases
    return dataset


@xr.register_dataset_accessor("gas_optics")
class GasOpticsAccessor:
    """Factory class that returns appropriate GasOptics implementation"""

    def __new__(cls, xarray_obj, selected_gases=None):
        # Check if source is internal by looking at required variables
        is_internal = (
            "totplnk" in xarray_obj.data_vars
            and "plank_fraction" in xarray_obj.data_vars
        )

        if is_internal:
            return LWGasOpticsAccessor(xarray_obj, is_internal, selected_gases)
        else:
            return SWGasOpticsAccessor(xarray_obj, is_internal, selected_gases)


class BaseGasOpticsAccessor:

    def __init__(self, xarray_obj, is_internal, selected_gases: list[str] | None = None, gas_name_map: dict[str, str] | None = None):
        self._dataset = xarray_obj

        self.is_internal = is_internal

        # Get the gas names from the dataset
        self._gas_names = self.extract_names(self._dataset["gas_names"].values)

        # Default gas name map
        default_map = {
            "h2o": "water_vapor",
            "co2": "carbon_dioxide_GM", 
            "o3": "ozone",
            "n2o": "nitrous_oxide_GM",
            "co": "carbon_monoxide_GM",
            "ch4": "methane_GM",
            "o2": "oxygen_GM",
            "n2": "nitrogen_GM",
            "ccl4": "carbon_tetrachloride_GM",
            "cfc11": "cfc11_GM",
            "cfc12": "cfc12_GM",
            "cfc22": "hcfc22_GM",
            "hfc143a": "hfc143a_GM",
            "hfc125": "hfc125_GM",
            "hfc23": "hfc23_GM",
            "hfc32": "hfc32_GM",
            "hfc134a": "hfc134a_GM",
            "cf4": "cf4_GM",
            "no2": "no2",
        }

        # Get the gas map to use
        if gas_name_map is None:
            gas_map = default_map
            if selected_gases is not None:
                gas_map = {g: default_map[g] for g in selected_gases if g in default_map}
        else:
            if selected_gases is not None:
                raise ValueError("When providing a gas_name_map, the selected gases are the keys of the gas_name_map")
            gas_map = gas_name_map

        # Validate the gas map
        invalid_gases = [g for g in gas_map.keys() if g not in self._gas_names]
        if invalid_gases:
            raise ValueError(f"Invalid gases in gas_name_map: {invalid_gases}. Valid gases are: {self._gas_names}")

        if "h2o" not in gas_map:
            raise ValueError("'h2o' must be included in gas mapping as it is required to compute Dry air")
        
        self.gas_name_map = gas_map

        # Set the gas names as coordinate in the dataset
        self._dataset.coords["absorber_ext"] = np.array(("dry_air",) + self._gas_names)

    def _initialize_pressure_levels(self, atmospheric_conditions, inplace=True):
        """Initialize pressure levels with minimum pressure adjustment"""
        pres_level_var = atmospheric_conditions.mapping.get_var("pres_level")

        min_index = np.argmin(atmospheric_conditions[pres_level_var].data)
        min_press = self._dataset["press_ref"].min().item() + sys.float_info.epsilon
        atmospheric_conditions[pres_level_var][:, min_index] = min_press

        if not inplace:
            return atmospheric_conditions

    @property
    def _selected_gas_names(self):
        return list(self.gas_name_map.keys())
    
    @property
    def _selected_gas_names_ext(self):
        return ["dry_air"] + self._selected_gas_names

    def get_gases_columns(self, atm_data):
        gas_values = []

        pres_level_var = atm_data.mapping.get_var("pres_level")

        for gas_map in self.gas_name_map.values():
            if gas_map in atm_data.data_vars.keys():
                if hasattr(atm_data[gas_map], 'units'):
                    values = atm_data[gas_map].values * float(atm_data[gas_map].units)
                else:
                    values = atm_data[gas_map].values
                if values.ndim == 0:
                    values = np.full((len(atm_data.site), len(atm_data.layer)), values)
            else:
                values = np.zeros((len(atm_data.site), len(atm_data.layer)))
            gas_values.append(values)

        gas_values = np.stack(gas_values, axis=-1)
        
        vmr_h2o = gas_values[..., self._selected_gas_names.index("h2o")]
        col_dry = self.get_col_dry(
            vmr_h2o, atm_data[pres_level_var].data, latitude=None
        )
        
        gas_values = gas_values * col_dry[..., np.newaxis]
        gas_values = np.concatenate([col_dry[..., np.newaxis], gas_values], axis=-1)
        
        return xr.DataArray(
            gas_values,
            dims=['site', 'layer', 'gas'],
            coords={
                'site': atm_data.site,
                'layer': atm_data.layer,
                'gas': self._selected_gas_names_ext
            },
            name='gases_columns'
        )

    def compute_problem(self, atmospheric_conditions, gas_interpolation_data):
        raise NotImplementedError()
    
    def compute_sources(self, atmospheric_conditions, gas_interpolation_data):
        raise NotImplementedError()
    
    def compute_boundary_conditions(self, atmospheric_conditions):
        raise NotImplementedError()

    def interpolate(self, atmospheric_conditions) -> xr.Dataset:
        # Get the gas columns from atmospheric conditions
        gas_order = self._selected_gas_names_ext
        gases_columns = self.get_gases_columns(atmospheric_conditions).sel(gas=gas_order)

        (jtemp, fmajor, fminor, col_mix, tropo, jeta, jpress) = interpolation(
            neta=len(self._dataset["mixing_fraction"]),
            flavor=self.flavors_sets.values.T,
            press_ref=self._dataset["press_ref"].values,
            temp_ref=self._dataset["temp_ref"].values,
            press_ref_trop=self._dataset["press_ref_trop"].values,
            vmr_ref=self._dataset["vmr_ref"].sel(absorber_ext=gas_order).values.T,
            play=atmospheric_conditions["pres_layer"].values,
            tlay=atmospheric_conditions["temp_layer"].values,
            col_gas=gases_columns.values,
        )

        # Create and return the dataset
        return xr.Dataset(
            data_vars={
                'temperature_index': (['site', 'layer'], jtemp),
                'pressure_index': (['site', 'layer'], jpress), 
                'tropopause_mask': (['site', 'layer'], tropo),
                'eta_index': (['gas_pair', 'column', 'layer', 'flavor'], jeta),
                'column_mix': (['temp_interp', 'column', 'layer', 'flavor'], col_mix),
                'fmajor': (['eta_interp', 'pressure_interp', 'temp_interp', 'column', 'layer', 'flavor'], fmajor),
                'fminor': (['eta_interp', 'temp_interp', 'column', 'layer', 'flavor'], fminor),
                'gases_columns': (['site', 'layer', 'gas'], gases_columns.values)
            },
            coords={
                'site': atmospheric_conditions.site,
                'layer': atmospheric_conditions.layer,
                'flavor': np.arange(fmajor.shape[-1]),
                'eta_interp': ['lower', 'upper'],
                'pressure_interp': ['lower', 'upper'],
                'temp_interp': ['lower', 'upper'],
                'gas_pair': ['first', 'second'],
                'gas': gases_columns.gas
            },
        )

    def tau_absorption(self, atmospheric_conditions, gas_interpolation_data):
        minor_gases_lower = self.extract_names(self._dataset["minor_gases_lower"].data)
        minor_gases_upper = self.extract_names(self._dataset["minor_gases_upper"].data)
        # check if the index is correct
        idx_minor_lower = self.get_idx_minor(self._gas_names, minor_gases_lower)
        idx_minor_upper = self.get_idx_minor(self._gas_names, minor_gases_upper)

        scaling_gas_lower = self.extract_names(self._dataset["scaling_gas_lower"].data)
        scaling_gas_upper = self.extract_names(self._dataset["scaling_gas_upper"].data)

        idx_minor_scaling_lower = self.get_idx_minor(self._gas_names, scaling_gas_lower)
        idx_minor_scaling_upper = self.get_idx_minor(self._gas_names, scaling_gas_upper)

        tau_absorption = compute_tau_absorption(
            self.idx_h2o,
            self.gpoint_flavor,
            self._dataset["bnd_limits_gpt"].values.T,
            self._dataset["kmajor"].values,
            self._dataset["kminor_lower"].values,
            self._dataset["kminor_upper"].values,
            self._dataset["minor_limits_gpt_lower"].values.T,
            self._dataset["minor_limits_gpt_upper"].values.T,
            self._dataset["minor_scales_with_density_lower"].values.astype(bool),
            self._dataset["minor_scales_with_density_upper"].values.astype(bool),
            self._dataset["scale_by_complement_lower"].values.astype(bool),
            self._dataset["scale_by_complement_upper"].values.astype(bool),
            idx_minor_lower,
            idx_minor_upper,
            idx_minor_scaling_lower,
            idx_minor_scaling_upper,
            self._dataset["kminor_start_lower"].values,
            self._dataset["kminor_start_upper"].values,
            gas_interpolation_data["tropopause_mask"].values,
            gas_interpolation_data["column_mix"].values,
            gas_interpolation_data["fmajor"].values,
            gas_interpolation_data["fminor"].values,
            atmospheric_conditions["pres_layer"].values,
            atmospheric_conditions["temp_layer"].values,
            gas_interpolation_data["gases_columns"].values,
            gas_interpolation_data["eta_index"].values,
            gas_interpolation_data["temperature_index"].values,
            gas_interpolation_data["pressure_index"].values,
        )

        # Create xarray Dataset with the computed values
        return xr.Dataset(
            {
                "tau": (["site", "layer", "gpt"], tau_absorption),
            },
            coords={
                "site": atmospheric_conditions.site,
                "layer": atmospheric_conditions.layer,
                "gpt": self._dataset.gpt,
            },
        )

    @property
    def idx_h2o(self):
        return list(self._gas_names).index("h2o") + 1

    @property
    def gpoint_flavor(self) -> npt.NDArray:
        """Get the g-point flavors from the k-distribution file.

        Each g-point is associated with a flavor, which is a pair of key species.

        Returns:
            np.ndarray: G-point flavors.
        """
        key_species = self._dataset["key_species"].values

        band_ranges = [
            [i] * (r.values[1] - r.values[0] + 1)
            for i, r in enumerate(self._dataset["bnd_limits_gpt"], 1)
        ]
        gpoint_bands = np.concatenate(band_ranges)

        key_species_rep = key_species.copy()
        key_species_rep[np.all(key_species_rep == [0, 0], axis=2)] = [2, 2]

        # unique flavors
        flist = self.flavors_sets.values.tolist()

        def key_species_pair2flavor(key_species_pair):
            return flist.index(key_species_pair.tolist()) + 1

        flavors_bands = np.apply_along_axis(
            key_species_pair2flavor, 2, key_species_rep
        ).tolist()
        gpoint_flavor = np.array([flavors_bands[gp - 1] for gp in gpoint_bands]).T

        return gpoint_flavor

    @property
    def flavors_sets(self) -> npt.NDArray:
        """Get the unique flavors from the k-distribution file.

        Returns:
            np.ndarray: Unique flavors.
        """
        key_species = self._dataset["key_species"].values
        tot_flav = len(self._dataset["bnd"]) * len(self._dataset["atmos_layer"])
        npairs = len(self._dataset["pair"])
        all_flav = np.reshape(key_species, (tot_flav, npairs))
        # (0,0) becomes (2,2) because absorption coefficients for these g-points will be 0.
        all_flav[np.all(all_flav == [0, 0], axis=1)] = [2, 2]
        # we do that instead of unique to preserv the order
        _, idx = np.unique(all_flav, axis=0, return_index=True)
        # return all_flav[np.sort(idx)].T
        flavors = all_flav[np.sort(idx)]
        return xr.DataArray(
            flavors,
            dims=['flavor', 'pairs'],
            coords={
                'flavor': np.arange(flavors.shape[0]),
                'pairs': np.arange(flavors.shape[1])
            }
        )

    @staticmethod
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
        return np.array(idx_minor_gas, dtype=np.int32)

    @staticmethod
    def extract_names(names):
        """Extract names from arrays, decoding and removing the suffix

        Args:
            names (np.ndarray): Names

        Returns:
            tuple: tuple of names
        """
        output = tuple(gas.tobytes().decode().strip().split("_")[0] for gas in names)
        return output

    @staticmethod
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

        # TODO: use numpy instead of loops
        for ilev in range(nlev - 1):
            for icol in range(ncol):
                delta_plev = abs(plev[icol, ilev] - plev[icol, ilev + 1])
                fact = 1.0 / (1.0 + vmr_h2o[icol, ilev])
                m_air = (M_DRY + M_H2O * vmr_h2o[icol, ilev]) * fact
                col_dry[icol, ilev] = (
                    10.0
                    * delta_plev
                    * AVOGAD
                    * fact
                    / (1000.0 * m_air * 100.0 * g0[icol])
                )
        return col_dry


    def compute(
        self,
        atmospheric_conditions: xr.Dataset,
        problem_type: str,
        mapping: AtmosphericMapping | None = None,
        add_to_input: bool = True,
    ):
        if mapping is None:
            mapping = create_default_mapping()

        # Set mapping in accessor
        atmospheric_conditions.mapping.set_mapping(mapping)

        # Get actual variable names in dataset
        pres_var = atmospheric_conditions.mapping.get_var("pres_layer")
        layer_dim = atmospheric_conditions.mapping.get_dim("layer")

        # Modify pressure levels to avoid division by zero, runs inplace
        self._initialize_pressure_levels(atmospheric_conditions)

        gas_interpolation_data = self.interpolate(atmospheric_conditions)
        problem = self.compute_problem(atmospheric_conditions, gas_interpolation_data)
        sources = self.compute_sources(atmospheric_conditions, gas_interpolation_data)
        boundary_conditions = self.compute_boundary_conditions(atmospheric_conditions)

        gas_optics = xr.merge([sources, problem, boundary_conditions])

        # Add problem type to dataset attributes
        if problem_type == "absorption" and self.is_internal:
            problem_type = ProblemTypes.LW_ABSORPTION.value
        elif problem_type == "two-stream" and self.is_internal:
            problem_type = ProblemTypes.LW_2STREAM.value
        elif problem_type == "absorption" and not self.is_internal:
            problem_type = ProblemTypes.SW_ABSORPTION.value
        elif problem_type == "two-stream" and not self.is_internal:
            problem_type = ProblemTypes.SW_2STREAM.value
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")

        if add_to_input:
            atmospheric_conditions.update(gas_optics)
            atmospheric_conditions.attrs["problem_type"] = problem_type
        else:
            output_ds = gas_optics
            output_ds.attrs["problem_type"] = problem_type
            return output_ds

class LWGasOpticsAccessor(BaseGasOpticsAccessor):
    """Accessor for internal radiation sources"""
    
    def compute_problem(self, atmospheric_conditions, gas_interpolation_data):
        return self.tau_absorption(atmospheric_conditions, gas_interpolation_data)
    
    def compute_sources(self, atmospheric_conditions, gas_interpolation_data):
        return self.compute_planck(atmospheric_conditions, gas_interpolation_data)
    
    def compute_boundary_conditions(self, atmospheric_conditions):
        if "surface_emissivity" not in atmospheric_conditions.data_vars:
            # Add surface emissivity directly to atmospheric conditions
            return xr.DataArray(
                np.ones(
                    (
                        atmospheric_conditions.sizes["site"],
                        atmospheric_conditions.sizes["gpt"],
                    )
                ),
                dims=["site", "gpt"],
                coords={
                    "site": atmospheric_conditions.site,
                    "gpt": atmospheric_conditions.gpt,
                },
            )
        else:
            return atmospheric_conditions["surface_emissivity"]


    def compute_planck(self, atmospheric_conditions, gas_interpolation_data):

        top_at_1 = atmospheric_conditions["layer"][0] < atmospheric_conditions["layer"][-1]

        (
            sfc_src,
            lay_source,
            lev_source,
            sfc_src_jac,
        ) = compute_planck_source(
            atmospheric_conditions["temp_layer"].values,
            atmospheric_conditions["temp_level"].values,
            atmospheric_conditions["surface_temperature"].values,
            top_at_1,
            gas_interpolation_data["fmajor"].values,
            gas_interpolation_data["eta_index"].values,
            gas_interpolation_data["tropopause_mask"].values,
            gas_interpolation_data["temperature_index"].values,
            gas_interpolation_data["pressure_index"].values,
            self._dataset["bnd_limits_gpt"].values.T,
            self._dataset["plank_fraction"].values.transpose(0, 2, 1, 3),
            self._dataset["temp_ref"].values.min(),
            self._dataset["temp_ref"].values.max(),
            self._dataset["totplnk"].values.T,
            self.gpoint_flavor,
        )

        # Create xarray Dataset with the computed values
        return xr.Dataset(
            {
                "surface_source": (["site", "gpt"], sfc_src),
                "layer_source": (["site", "layer", "gpt"], lay_source),
                "level_source": (["site", "level", "gpt"], lev_source),
                "surface_source_jacobian": (["site", "gpt"], sfc_src_jac),
            },
            coords={
                "site": atmospheric_conditions.site,
                "layer": atmospheric_conditions.layer,
                "level": atmospheric_conditions.level,
                "gpt": self._dataset.gpt,
            },
        )

class SWGasOpticsAccessor(BaseGasOpticsAccessor):
    """Accessor for external radiation sources"""

    def compute_problem(self, atmospheric_conditions, gas_interpolation_data):
        # Calculate absorption optical depth
        tau_abs = self.tau_absorption(atmospheric_conditions, gas_interpolation_data)
        
        # Calculate Rayleigh scattering optical depth  
        tau_rayleigh = self.tau_rayleigh(gas_interpolation_data)
        tau = tau_abs + tau_rayleigh
        ssa = xr.where(
            tau["tau"] > 2.0 * np.finfo(float).tiny,
            tau_rayleigh["tau"] / tau["tau"],
            0.0,
        ).rename("ssa")
        g = xr.zeros_like(tau["tau"]).rename("g")
        return xr.merge([tau, ssa, g])

    def compute_sources(self, atmospheric_conditions, *args, **kwargs):
        """Implementation for external source computation"""
        a_offset = SOLAR_CONSTANTS["A_OFFSET"]
        b_offset = SOLAR_CONSTANTS["B_OFFSET"]

        solar_source_quiet = self._dataset["solar_source_quiet"]
        solar_source_facular = self._dataset["solar_source_facular"]
        solar_source_sunspot = self._dataset["solar_source_sunspot"]

        mg_index = self._dataset["mg_default"]
        sb_index = self._dataset["sb_default"]

        solar_source = (
            solar_source_quiet
            + (mg_index - a_offset) * solar_source_facular
            + (sb_index - b_offset) * solar_source_sunspot
        )

        total_solar_irradiance = atmospheric_conditions["total_solar_irradiance"]

        toa_flux = solar_source.broadcast_like(total_solar_irradiance)
        def_tsi = toa_flux.sum(dim='gpt')
        return (toa_flux * (total_solar_irradiance / def_tsi)).rename("toa_source")

    def compute_boundary_conditions(self, atmospheric_conditions):

        usecol_values = atmospheric_conditions["solar_zenith_angle"] < (90.0 - 2.0 * np.spacing(90.0))
        usecol_values = usecol_values.rename("solar_angle_mask")
        mu0 = xr.where(usecol_values, np.cos(np.radians(atmospheric_conditions["solar_zenith_angle"])), 1.0)
        solar_zenith_angle = mu0.broadcast_like(atmospheric_conditions.layer).rename("solar_zenith_angle")

        if "surface_albedo_dir" not in atmospheric_conditions.data_vars:
            surface_albedo_direct = atmospheric_conditions["surface_albedo"]
            surface_albedo_direct = surface_albedo_direct.rename("surface_albedo_direct")
            surface_albedo_diffuse = atmospheric_conditions["surface_albedo"]
            surface_albedo_diffuse = surface_albedo_diffuse.rename("surface_albedo_diffuse")
        else:
            surface_albedo_direct = atmospheric_conditions["surface_albedo_dir"]
            surface_albedo_direct = surface_albedo_direct.rename("surface_albedo_direct")
            surface_albedo_diffuse = atmospheric_conditions["surface_albedo_dif"]
            surface_albedo_diffuse = surface_albedo_diffuse.rename("surface_albedo_diffuse")
        
        return xr.merge([solar_zenith_angle, surface_albedo_direct, surface_albedo_diffuse, usecol_values])


    def tau_rayleigh(self, gas_interpolation_data):
        krayl = np.stack(
            [self._dataset["rayl_lower"].values, self._dataset["rayl_upper"].values],
            axis=-1,
        )
        tau_rayleigh = compute_tau_rayleigh(
            self.gpoint_flavor,
            self._dataset["bnd_limits_gpt"].values.T,
            krayl,
            self.idx_h2o,
            gas_interpolation_data["gases_columns"].values[:, :, 0],
            gas_interpolation_data["gases_columns"].values,
            gas_interpolation_data["fminor"].values,
            gas_interpolation_data["eta_index"].values,
            gas_interpolation_data["tropopause_mask"].values,
            gas_interpolation_data["temperature_index"].values,
        )
    
        return xr.Dataset(
            {
                "tau": (["site", "layer", "gpt"], tau_rayleigh),
            },
            coords={
                "site": gas_interpolation_data.site,
                "layer": gas_interpolation_data.layer,
                "gpt": self._dataset.gpt,
            },
        )

