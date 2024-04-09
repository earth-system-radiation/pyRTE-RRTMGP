import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import xarray as xr

from pyrte_rrtmgp.constants import AVOGAD, HELMERT1, HELMERT2, M_DRY, M_H2O
from pyrte_rrtmgp.exceptions import (
    MissingAtmosfericConditionsError,
    NotExternalSourceError,
    NotInternalSourceError,
)
from pyrte_rrtmgp.kernels.rrtmgp import (
    compute_planck_source,
    compute_tau_absorption,
    compute_tau_rayleigh,
    interpolation,
)


@dataclass
class GasOptics:
    tau: Optional[np.ndarray] = None
    tau_rayleigh: Optional[np.ndarray] = None
    tau_absorption: Optional[np.ndarray] = None
    g: Optional[np.ndarray] = None
    ssa: Optional[np.ndarray] = None
    lay_src: Optional[np.ndarray] = None
    lev_src: Optional[np.ndarray] = None
    sfc_src: Optional[np.ndarray] = None
    sfc_src_jac: Optional[np.ndarray] = None
    solar_source: Optional[np.ndarray] = None


@dataclass
class InterpolatedAtmosfereGases:
    jtemp: Optional[np.ndarray] = None
    fmajor: Optional[np.ndarray] = None
    fminor: Optional[np.ndarray] = None
    col_mix: Optional[np.ndarray] = None
    tropo: Optional[np.ndarray] = None
    jeta: Optional[np.ndarray] = None
    jpress: Optional[np.ndarray] = None


@xr.register_dataset_accessor("gas_optics")
class GasOpticsAccessor:
    def __init__(self, xarray_obj, selected_gases=None):
        self._obj = xarray_obj
        self._selected_gases = selected_gases
        self._gas_names = None
        self._is_internal = None
        self._gas_mappings = None
        self._top_at_1 = None
        self._vmr_ref = None
        self.col_gas = None

        self._interpolated = InterpolatedAtmosfereGases()
        self.gas_optics = GasOptics()

    @property
    def gas_names(self):
        """Gas names"""
        if self._gas_names is None:
            names = self._obj["gas_names"].values
            self._gas_names = self.extract_names(names)
        return self._gas_names

    @property
    def source_is_internal(self):
        """Check if the source is internal"""
        if self._is_internal is None:
            variables = self._obj.data_vars
            self._is_internal = "totplnk" in variables and "plank_fraction" in variables
        return self._is_internal

    def solar_source(self):
        """Calculate the solar variability

        Returns:
            np.ndarray: Solar source
        """

        if self.source_is_internal:
            raise NotExternalSourceError(
                "Solar source is not available for internal sources."
            )

        if self.gas_optics.solar_source is None:
            a_offset = 0.1495954
            b_offset = 0.00066696

            solar_source_quiet = self._obj["solar_source_quiet"]
            solar_source_facular = self._obj["solar_source_facular"]
            solar_source_sunspot = self._obj["solar_source_sunspot"]

            mg_index = self._obj["mg_default"]
            sb_index = self._obj["sb_default"]

            self.gas_optics.solar_source = (
                solar_source_quiet
                + (mg_index - a_offset) * solar_source_facular
                + (sb_index - b_offset) * solar_source_sunspot
            ).data

    def load_atmosferic_conditions(self, atmosferic_conditions: xr.Dataset):
        """Load atmospheric conditions"""
        self._atm_cond = atmosferic_conditions

        # RRTMGP won't run with pressure less than its minimum.
        # So we add a small value to the minimum pressure
        min_index = np.argmin(self._atm_cond["pres_level"].data)
        min_press = self._obj["press_ref"].min().item() + sys.float_info.epsilon
        self._atm_cond["pres_level"][:, min_index] = min_press

        self.get_col_gas()

        self.interpolate()
        self.compute_gas_taus()
        if self.source_is_internal:
            self.compute_planck()
        else:
            self.solar_source()

        return self.gas_optics

    def get_col_gas(self):
        if self._atm_cond is None:
            raise MissingAtmosfericConditionsError()

        ncol = len(self._atm_cond["site"])
        nlay = len(self._atm_cond["layer"])
        col_gas = []
        for gas_name in self.gas_mappings.values():
            # if gas_name is not available, fill it with zeros
            if gas_name not in self._atm_cond.data_vars.keys():
                gas_values = np.zeros((ncol, nlay))
            else:
                try:
                    scale = float(self._atm_cond[gas_name].units)
                except AttributeError:
                    scale = 1.0
                gas_values = self._atm_cond[gas_name].values * scale

            if gas_values.ndim == 0:
                gas_values = np.full((ncol, nlay), gas_values)
            col_gas.append(gas_values)

        vmr_h2o = col_gas[self.gas_names.index("h2o")]
        col_dry = self.get_col_dry(
            vmr_h2o, self._atm_cond["pres_level"].data, latitude=None
        )
        col_gas = [col_dry] + col_gas

        col_gas = np.stack(col_gas, axis=-1).astype(np.float64)
        col_gas[:, :, 1:] = col_gas[:, :, 1:] * col_gas[:, :, :1]

        self.col_gas = col_gas

    @property
    def gas_mappings(self):
        """Gas mappings"""

        if self._atm_cond is None:
            raise MissingAtmosfericConditionsError()

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
            if self._atm_cond is None:
                raise MissingAtmosfericConditionsError()

            pres_layers = self._atm_cond["pres_layer"]["layer"]
            self._top_at_1 = pres_layers[0] < pres_layers[-1]
        return self._top_at_1.item()

    @property
    def vmr_ref(self):
        if self._vmr_ref is None:
            if self._atm_cond is None:
                raise MissingAtmosfericConditionsError()
            sel_gases = self.gas_mappings.keys()
            vmr_idx = [i for i, g in enumerate(self._gas_names, 1) if g in sel_gases]
            vmr_idx = [0] + vmr_idx
            self._vmr_ref = self._obj["vmr_ref"].sel(absorber_ext=vmr_idx).values.T
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
            neta=len(self._obj["mixing_fraction"]),
            flavor=self.flavors_sets,
            press_ref=self._obj["press_ref"].values,
            temp_ref=self._obj["temp_ref"].values,
            press_ref_trop=self._obj["press_ref_trop"].values.item(),
            vmr_ref=self.vmr_ref,
            play=self._atm_cond["pres_layer"].values,
            tlay=self._atm_cond["temp_layer"].values,
            col_gas=self.col_gas,
        )

    def compute_planck(self):
        (
            self.gas_optics.sfc_src,
            self.gas_optics.lay_src,
            self.gas_optics.lev_src,
            self.gas_optics.sfc_src_jac,
        ) = compute_planck_source(
            self._atm_cond["temp_layer"].values,
            self._atm_cond["temp_level"].values,
            self._atm_cond["surface_temperature"].values,
            self.top_at_1,
            self._interpolated.fmajor,
            self._interpolated.jeta,
            self._interpolated.tropo,
            self._interpolated.jtemp,
            self._interpolated.jpress,
            self._obj["bnd_limits_gpt"].values.T,
            self._obj["plank_fraction"].values.transpose(0, 2, 1, 3),
            self._obj["temp_ref"].values.min(),
            self._obj["temp_ref"].values.max(),
            self._obj["totplnk"].values.T,
            self.gpoint_flavor,
        )

    def compute_gas_taus(self):
        minor_gases_lower = self.extract_names(self._obj["minor_gases_lower"].data)
        minor_gases_upper = self.extract_names(self._obj["minor_gases_upper"].data)
        # check if the index is correct
        idx_minor_lower = self.get_idx_minor(self.gas_names, minor_gases_lower)
        idx_minor_upper = self.get_idx_minor(self.gas_names, minor_gases_upper)

        scaling_gas_lower = self.extract_names(self._obj["scaling_gas_lower"].data)
        scaling_gas_upper = self.extract_names(self._obj["scaling_gas_upper"].data)

        idx_minor_scaling_lower = self.get_idx_minor(self.gas_names, scaling_gas_lower)
        idx_minor_scaling_upper = self.get_idx_minor(self.gas_names, scaling_gas_upper)

        tau_absorption = compute_tau_absorption(
            self.idx_h2o,
            self.gpoint_flavor,
            self._obj["bnd_limits_gpt"].values.T,
            self._obj["kmajor"].values,
            self._obj["kminor_lower"].values,
            self._obj["kminor_upper"].values,
            self._obj["minor_limits_gpt_lower"].values.T,
            self._obj["minor_limits_gpt_upper"].values.T,
            self._obj["minor_scales_with_density_lower"].values.astype(bool),
            self._obj["minor_scales_with_density_upper"].values.astype(bool),
            self._obj["scale_by_complement_lower"].values.astype(bool),
            self._obj["scale_by_complement_upper"].values.astype(bool),
            idx_minor_lower,
            idx_minor_upper,
            idx_minor_scaling_lower,
            idx_minor_scaling_upper,
            self._obj["kminor_start_lower"].values,
            self._obj["kminor_start_upper"].values,
            self._interpolated.tropo,
            self._interpolated.col_mix,
            self._interpolated.fmajor,
            self._interpolated.fminor,
            self._atm_cond["pres_layer"].values,
            self._atm_cond["temp_layer"].values,
            self.col_gas,
            self._interpolated.jeta,
            self._interpolated.jtemp,
            self._interpolated.jpress,
        )

        self.gas_optics.tau_absorption = tau_absorption
        if self.source_is_internal:
            self.gas_optics.tau = tau_absorption
            self.gas_optics.ssa = np.full_like(tau_absorption, np.nan)
            self.gas_optics.g = np.full_like(tau_absorption, np.nan)
        else:
            krayl = np.stack(
                [self._obj["rayl_lower"].values, self._obj["rayl_upper"].values],
                axis=-1,
            )
            tau_rayleigh = compute_tau_rayleigh(
                self.gpoint_flavor,
                self._obj["bnd_limits_gpt"].values.T,
                krayl,
                self.idx_h2o,
                self.col_gas[:, :, 0],
                self.col_gas,
                self._interpolated.fminor,
                self._interpolated.jeta,
                self._interpolated.tropo,
                self._interpolated.jtemp,
            )

            self.gas_optics.tau_rayleigh = tau_rayleigh
            self.gas_optics.tau = tau_absorption + tau_rayleigh
            self.gas_optics.ssa = np.where(
                self.gas_optics.tau > 2.0 * np.finfo(float).tiny,
                tau_rayleigh / self.gas_optics.tau,
                0.0,
            )
            self.gas_optics.g = np.zeros(self.gas_optics.tau.shape)

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
        key_species = self._obj["key_species"].values

        band_ranges = [
            [i] * (r.values[1] - r.values[0] + 1)
            for i, r in enumerate(self._obj["bnd_limits_gpt"], 1)
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
        key_species = self._obj["key_species"].values
        tot_flav = len(self._obj["bnd"]) * len(self._obj["atmos_layer"])
        npairs = len(self._obj["pair"])
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
