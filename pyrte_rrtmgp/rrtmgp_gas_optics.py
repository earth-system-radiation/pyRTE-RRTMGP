import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


import numpy.typing as npt
import xarray as xr

from pyrte_rrtmgp.constants import AVOGAD, HELMERT1, HELMERT2, M_DRY, M_H2O, SOLAR_CONSTANTS
from pyrte_rrtmgp.exceptions import (
    MissingAtmosphericConditionsError,
)
from pyrte_rrtmgp.kernels.rrtmgp import (
    compute_planck_source,
    compute_tau_absorption,
    compute_tau_rayleigh,
    interpolation,
)

from pyrte_rrtmgp.rte_problems import LWProblem, SWProblem

from functools import cached_property


@dataclass
class InterpolatedAtmosphereGases:
    """Stores interpolated atmosphere gas data with type hints and validation.
    
    All fields are optional numpy arrays of type float64.
    """
    jtemp: Optional[npt.NDArray[np.float64]] = None
    fmajor: Optional[npt.NDArray[np.float64]] = None 
    fminor: Optional[npt.NDArray[np.float64]] = None
    col_mix: Optional[npt.NDArray[np.float64]] = None
    tropo: Optional[npt.NDArray[np.float64]] = None
    jeta: Optional[npt.NDArray[np.float64]] = None
    jpress: Optional[npt.NDArray[np.float64]] = None


@xr.register_dataset_accessor("gas_optics")
class GasOpticsAccessor:
    """Factory class that returns appropriate GasOptics implementation"""
    def __new__(cls, xarray_obj, selected_gases=None):
        # Check if source is internal by looking at required variables
        is_internal = "totplnk" in xarray_obj.data_vars and "plank_fraction" in xarray_obj.data_vars
        
        if is_internal:
            return LWGasOpticsAccessor(xarray_obj, selected_gases)
        else:
            return SWGasOpticsAccessor(xarray_obj, selected_gases)


class BaseGasOpticsAccessor:
    def __init__(self, xarray_obj, selected_gases=None):
        self._dataset = xarray_obj
        self._selected_gases = selected_gases
        self._gas_names = None
        self._gas_mappings = None
        self._top_at_1 = None
        self._vmr_ref = None
        self.column_gases = None

        self._interpolated = InterpolatedAtmosphereGases()
        self._atmospheric_conditions = None

    @property
    def gas_names(self):
        """Gas names"""
        if self._gas_names is None:
            names = self._dataset["gas_names"].values
            self._gas_names = self.extract_names(names)
        return self._gas_names

    @property
    def gas_optics(self):
        """Return the appropriate problem instance - to be implemented by subclasses"""
        raise NotImplementedError()

    def load_atmospheric_conditions(self, atmospheric_conditions: xr.Dataset):
        """Load atmospheric conditions"""
        if not isinstance(atmospheric_conditions, xr.Dataset):
            raise TypeError("atmospheric_conditions must be an xarray Dataset")

        # Validate required dimensions
        required_dims = {'site', 'layer', 'level'}
        missing_dims = required_dims - set(atmospheric_conditions.dims)
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")

        # Validate required variables
        required_vars = {'pres_level', 'temp_layer', 'pres_layer', 'temp_level'}
        missing_vars = required_vars - set(atmospheric_conditions.data_vars)
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        self._atmospheric_conditions = atmospheric_conditions
        self._initialize_pressure_levels()
        self.get_col_gas()
        self.interpolate()
        self.compute_source()
        return self.gas_optics

    def _initialize_pressure_levels(self):
        """Initialize pressure levels with minimum pressure adjustment"""
        min_index = np.argmin(self._atmospheric_conditions["pres_level"].data)
        min_press = self._dataset["press_ref"].min().item() + sys.float_info.epsilon
        self._atmospheric_conditions["pres_level"][:, min_index] = min_press

    def get_col_gas(self):
        if self._atmospheric_conditions is None:
            raise MissingAtmosphericConditionsError()

        ncol = len(self._atmospheric_conditions["site"])
        nlay = len(self._atmospheric_conditions["layer"])
        col_gas = []
        for gas_name in self.gas_mappings.values():
            # if gas_name is not available, fill it with zeros
            if gas_name not in self._atmospheric_conditions.data_vars.keys():
                gas_values = np.zeros((ncol, nlay))
            else:
                try:
                    scale = float(self._atmospheric_conditions[gas_name].units)
                except AttributeError:
                    scale = 1.0
                gas_values = self._atmospheric_conditions[gas_name].values * scale

            if gas_values.ndim == 0:
                gas_values = np.full((ncol, nlay), gas_values)
            col_gas.append(gas_values)

        vmr_h2o = col_gas[self.gas_names.index("h2o")]
        col_dry = self.get_col_dry(
            vmr_h2o, self._atmospheric_conditions["pres_level"].data, latitude=None
        )
        col_gas = [col_dry] + col_gas

        col_gas = np.stack(col_gas, axis=-1).astype(np.float64)
        col_gas[:, :, 1:] = col_gas[:, :, 1:] * col_gas[:, :, :1]

        self.column_gases = col_gas

    def compute_gas_taus(self):
        raise NotImplementedError()

    @property
    def gas_mappings(self):
        """Gas mappings"""

        if self._atmospheric_conditions is None:
            raise MissingAtmosphericConditionsError()

        if self._gas_mappings is None:
            gas_name_map = {
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

            if self._selected_gases is not None:
                gas_name_map = {
                    g: gas_name_map[g]
                    for g in self._selected_gases
                    if g in gas_name_map
                }

            gas_name_map = {
                g: gas_name_map[g] for g in self.gas_names if g in gas_name_map
            }
            self._gas_mappings = gas_name_map
        return self._gas_mappings

    @property
    def top_at_1(self):
        if self._top_at_1 is None:
            if self._atmospheric_conditions is None:
                raise MissingAtmosphericConditionsError()

            pres_layers = self._atmospheric_conditions["pres_layer"]["layer"]
            self._top_at_1 = pres_layers[0] < pres_layers[-1]
        return self._top_at_1.item()

    @property
    def vmr_ref(self):
        if self._vmr_ref is None:
            if self._atmospheric_conditions is None:
                raise MissingAtmosphericConditionsError()
            sel_gases = self.gas_mappings.keys()
            vmr_idx = [i for i, g in enumerate(self._gas_names, 1) if g in sel_gases]
            vmr_idx = [0] + vmr_idx
            self._vmr_ref = self._dataset["vmr_ref"].sel(absorber_ext=vmr_idx).values.T
        return self._vmr_ref

    def interpolate(self):
        (
            self._interpolated.jtemp,
            self._interpolated.fmajor,
            self._interpolated.fminor,
            self._interpolated.col_mix,
            self._interpolated.tropo,
            self._interpolated.jeta,
            self._interpolated.jpress,
        ) = interpolation(
            neta=len(self._dataset["mixing_fraction"]),
            flavor=self.flavors_sets,
            press_ref=self._dataset["press_ref"].values,
            temp_ref=self._dataset["temp_ref"].values,
            press_ref_trop=self._dataset["press_ref_trop"].values.item(),
            vmr_ref=self.vmr_ref,
            play=self._atmospheric_conditions["pres_layer"].values,
            tlay=self._atmospheric_conditions["temp_layer"].values,
            col_gas=self.column_gases,
        )

    @cached_property
    def tau_absorption(self):
        minor_gases_lower = self.extract_names(self._dataset["minor_gases_lower"].data)
        minor_gases_upper = self.extract_names(self._dataset["minor_gases_upper"].data)
        # check if the index is correct
        idx_minor_lower = self.get_idx_minor(self.gas_names, minor_gases_lower)
        idx_minor_upper = self.get_idx_minor(self.gas_names, minor_gases_upper)

        scaling_gas_lower = self.extract_names(self._dataset["scaling_gas_lower"].data)
        scaling_gas_upper = self.extract_names(self._dataset["scaling_gas_upper"].data)

        idx_minor_scaling_lower = self.get_idx_minor(self.gas_names, scaling_gas_lower)
        idx_minor_scaling_upper = self.get_idx_minor(self.gas_names, scaling_gas_upper)

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
            self._interpolated.tropo,
            self._interpolated.col_mix,
            self._interpolated.fmajor,
            self._interpolated.fminor,
            self._atmospheric_conditions["pres_layer"].values,
            self._atmospheric_conditions["temp_layer"].values,
            self.column_gases,
            self._interpolated.jeta,
            self._interpolated.jtemp,
            self._interpolated.jpress,
        )

        return tau_absorption


    @property
    def idx_h2o(self):
        return list(self.gas_names).index("h2o") + 1

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
        flist = self.flavors_sets.T.tolist()

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
        return all_flav[np.sort(idx)].T

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


class LWGasOpticsAccessor(BaseGasOpticsAccessor):
    """Accessor for internal radiation sources"""

    def __init__(self, xarray_obj, selected_gases=None):
        super().__init__(xarray_obj, selected_gases)
        self.lay_source = None
        self.lev_source = None
        self.sfc_src = None
        self.sfc_src_jac = None

    @property
    def gas_optics(self):
        return LWProblem(
            tau=self.tau_absorption,
            lay_source=self.lay_source,
            lev_source=self.lev_source,
            sfc_src=self.sfc_src,
            sfc_src_jac=self.sfc_src_jac
        )

    def compute_source(self):
        """Implementation for internal source computation"""
        self.compute_planck()

    def compute_planck(self):
        (
            self.sfc_src,
            self.lay_source,
            self.lev_source,
            self.sfc_src_jac,
        ) = compute_planck_source(
            self._atmospheric_conditions["temp_layer"].values,
            self._atmospheric_conditions["temp_level"].values,
            self._atmospheric_conditions["surface_temperature"].values,
            self.top_at_1,
            self._interpolated.fmajor,
            self._interpolated.jeta,
            self._interpolated.tropo,
            self._interpolated.jtemp,
            self._interpolated.jpress,
            self._dataset["bnd_limits_gpt"].values.T,
            self._dataset["plank_fraction"].values.transpose(0, 2, 1, 3),
            self._dataset["temp_ref"].values.min(),
            self._dataset["temp_ref"].values.max(),
            self._dataset["totplnk"].values.T,
            self.gpoint_flavor,
        )


class SWGasOpticsAccessor(BaseGasOpticsAccessor):
    """Accessor for external radiation sources"""
    
    def __init__(self, xarray_obj, selected_gases=None):
        super().__init__(xarray_obj, selected_gases)
        self._solar_source = None
        self._total_solar_irradiance = None
        self._solar_zenith_angle = None
        self._sfc_alb_dir = None
        self._sfc_alb_dif = None

    @property
    def gas_optics(self):
        return SWProblem(
            tau=self.tau,
            ssa=self.ssa,
            g=self.g,
            solar_zenith_angle=self._solar_zenith_angle,
            sfc_alb_dir=self._sfc_alb_dir,
            sfc_alb_dif=self._sfc_alb_dif,
            total_solar_irradiance=self._total_solar_irradiance,
            solar_source=self._solar_source,
            compute_mu0_fn=self.compute_mu0,
            compute_toa_flux_fn=self.compute_toa_flux
        )

    def compute_source(self):
        """Implementation for external source computation"""
        a_offset = SOLAR_CONSTANTS['A_OFFSET']
        b_offset = SOLAR_CONSTANTS['B_OFFSET']

        solar_source_quiet = self._dataset["solar_source_quiet"]
        solar_source_facular = self._dataset["solar_source_facular"]
        solar_source_sunspot = self._dataset["solar_source_sunspot"]

        mg_index = self._dataset["mg_default"]
        sb_index = self._dataset["sb_default"]

        self._solar_source = (
            solar_source_quiet
            + (mg_index - a_offset) * solar_source_facular
            + (sb_index - b_offset) * solar_source_sunspot
        ).data

    @cached_property
    def tau_rayleigh(self):
        krayl = np.stack(
            [self._dataset["rayl_lower"].values, self._dataset["rayl_upper"].values],
            axis=-1,
        )
        return compute_tau_rayleigh(
            self.gpoint_flavor,
            self._dataset["bnd_limits_gpt"].values.T,
            krayl,
            self.idx_h2o,
            self.column_gases[:, :, 0],
            self.column_gases,
            self._interpolated.fminor,
            self._interpolated.jeta,
            self._interpolated.tropo,
            self._interpolated.jtemp,
        )
    
    @property
    def tau(self):
        return self.tau_absorption + self.tau_rayleigh

    @property
    def ssa(self):
        return np.where(
            self.tau > 2.0 * np.finfo(float).tiny,
            self.tau_rayleigh / self.tau,
            0.0,
        )
    
    @property
    def g(self):
        return np.zeros(self.tau.shape)

    @staticmethod
    def compute_mu0(solar_zenith_angle, nlayer=None):
        """Calculate the cosine of the solar zenith angle

        Args:
            solar_zenith_angle (np.ndarray): Solar zenith angle in degrees
            nlayer (int, optional): Number of layers. Defaults to None.
        """
        usecol_values = solar_zenith_angle < 90.0 - 2.0 * np.spacing(90.0)
        mu0 = np.where(usecol_values, np.cos(np.radians(solar_zenith_angle)), 1.0)
        if nlayer is not None:
            mu0 = np.stack([mu0] * nlayer).T
        return mu0

    @staticmethod
    def compute_toa_flux(total_solar_irradiance, solar_source):
        """Compute the top of atmosphere flux

        Args:
            total_solar_irradiance (np.ndarray): Total solar irradiance
            solar_source (np.ndarray): Solar source

        Returns:
            np.ndarray: Top of atmosphere flux
        """
        ncol = total_solar_irradiance.shape[0]
        toa_flux = np.stack([solar_source] * ncol)
        def_tsi = toa_flux.sum(axis=1)
        return (toa_flux.T * (total_solar_irradiance / def_tsi)).T