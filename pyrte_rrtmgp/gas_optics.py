import numpy as np

from pyrte_rrtmgp.rrtmgp import (
    compute_planck_source,
    compute_tau_absorption,
    compute_tau_rayleigh,
    interpolation,
)
from pyrte_rrtmgp.utils import (
    combine_abs_and_rayleigh,
    extract_gas_names,
    flavors_from_kdist,
    get_idx_minor,
    gpoint_flavor_from_kdist,
    krayl_from_kdist,
    rfmip_2_col_gas,
)


class GasOptics:

    def __init__(self, kdist, rfmip, gases_to_use=None):
        self.kdist = kdist
        self.rfmip = rfmip

        kdist_gas_names = extract_gas_names(kdist["gas_names"].values)
        self.kdist_gas_names = kdist_gas_names
        rfmip_vars = list(rfmip.keys())

        gas_names = {n: n + "_GM" for n in kdist_gas_names if n + "_GM" in rfmip_vars}

        # Create a dict that maps the gas names in the kdist gas names to the gas names in the rfmip dataset
        gas_names.update(
            {
                "co": "carbon_monoxide_GM",
                "ch4": "methane_GM",
                "o2": "oxygen_GM",
                "n2o": "nitrous_oxide_GM",
                "n2": "nitrogen_GM",
                "co2": "carbon_dioxide_GM",
                "ccl4": "carbon_tetrachloride_GM",
                "cfc22": "hcfc22_GM",
                "h2o": "water_vapor",
                "o3": "ozone",
                "no2": "no2",
            }
        )

        # sort gas names based on kdist
        gas_names = {g: gas_names[g] for g in kdist_gas_names if g in gas_names}

        if gases_to_use is not None:
            gas_names = {g: gas_names[g] for g in gases_to_use}

        self.gas_names = gas_names

        self.tlay = rfmip["temp_layer"].values
        self.play = rfmip["pres_layer"].values
        self.col_gas = rfmip_2_col_gas(rfmip, list(gas_names.values()), dry_air=True)

    @property
    def source_is_internal(self):
        variables = self.kdist.data_vars
        return "totplnk" in variables and "plank_fraction" in variables

    def gas_optics(self):
        if self.source_is_internal:
            self.interpolate()
            self.compute_planck()
            self.compute_gas_taus()
            return (
                self.tau,
                self.g,
                self.ssa,
                self.lay_src,
                self.lev_src,
                self.sfc_src,
                self.sfc_src_jac,
            )
        else:
            self.interpolate()
            self.compute_gas_taus()
            self.compute_solar_variability()
            return self.tau, self.g, self.ssa, self.solar_source

    def interpolate(self):
        neta = len(self.kdist["mixing_fraction"])
        press_ref = self.kdist["press_ref"].values
        temp_ref = self.kdist["temp_ref"].values

        press_ref_trop = self.kdist["press_ref_trop"].values.item()

        # dry air is zero
        vmr_idx = [
            i for i, g in enumerate(self.kdist_gas_names, 1) if g in self.gas_names
        ]
        vmr_idx = [0] + vmr_idx
        vmr_ref = self.kdist["vmr_ref"].sel(absorber_ext=vmr_idx).values.T

        # just the unique sets of gases
        flavor = flavors_from_kdist(self.kdist)

        (
            self.jtemp,
            self.fmajor,
            self.fminor,
            self.col_mix,
            self.tropo,
            self.jeta,
            self.jpress,
        ) = interpolation(
            neta=neta,
            flavor=flavor,
            press_ref=press_ref,
            temp_ref=temp_ref,
            press_ref_trop=press_ref_trop,
            vmr_ref=vmr_ref,
            play=self.play,
            tlay=self.tlay,
            col_gas=self.col_gas,
        )

    def compute_planck(self):

        tlay = self.rfmip["temp_layer"].values
        tlev = self.rfmip["temp_level"].values
        tsfc = self.rfmip["surface_temperature"].values
        pres_layers = self.rfmip["pres_layer"]["layer"]
        top_at_1 = pres_layers[0] < pres_layers[-1]
        band_lims_gpt = self.kdist["bnd_limits_gpt"].values.T
        pfracin = self.kdist["plank_fraction"].values.transpose(0, 2, 1, 3)
        temp_ref_min = self.kdist["temp_ref"].values.min()
        temp_ref_max = self.kdist["temp_ref"].values.max()
        totplnk = self.kdist["totplnk"].values.T

        gpoint_flavor = gpoint_flavor_from_kdist(self.kdist)

        self.sfc_src, self.lay_src, self.lev_src, self.sfc_src_jac = (
            compute_planck_source(
                tlay,
                tlev,
                tsfc,
                top_at_1,
                self.fmajor,
                self.jeta,
                self.tropo,
                self.jtemp,
                self.jpress,
                band_lims_gpt,
                pfracin,
                temp_ref_min,
                temp_ref_max,
                totplnk,
                gpoint_flavor,
            )
        )

    def compute_gas_taus(self):

        idx_h2o = list(self.gas_names).index("h2o") + 1

        gpoint_flavor = gpoint_flavor_from_kdist(self.kdist)

        kmajor = self.kdist["kmajor"].values
        kminor_lower = self.kdist["kminor_lower"].values
        kminor_upper = self.kdist["kminor_upper"].values
        minor_limits_gpt_lower = self.kdist["minor_limits_gpt_lower"].values.T
        minor_limits_gpt_upper = self.kdist["minor_limits_gpt_upper"].values.T

        minor_scales_with_density_lower = self.kdist[
            "minor_scales_with_density_lower"
        ].values.astype(bool)
        minor_scales_with_density_upper = self.kdist[
            "minor_scales_with_density_upper"
        ].values.astype(bool)
        scale_by_complement_lower = self.kdist[
            "scale_by_complement_lower"
        ].values.astype(bool)
        scale_by_complement_upper = self.kdist[
            "scale_by_complement_upper"
        ].values.astype(bool)

        gas_name_list = list(self.gas_names.keys())

        band_lims_gpt = self.kdist["bnd_limits_gpt"].values.T

        minor_gases_lower = extract_gas_names(self.kdist["minor_gases_lower"].values)
        minor_gases_upper = extract_gas_names(self.kdist["minor_gases_upper"].values)
        # check if the index is correct
        idx_minor_lower = get_idx_minor(gas_name_list, minor_gases_lower)
        idx_minor_upper = get_idx_minor(gas_name_list, minor_gases_upper)

        minor_scaling_gas_lower = extract_gas_names(
            self.kdist["scaling_gas_lower"].values
        )
        minor_scaling_gas_upper = extract_gas_names(
            self.kdist["scaling_gas_upper"].values
        )

        idx_minor_scaling_lower = get_idx_minor(gas_name_list, minor_scaling_gas_lower)
        idx_minor_scaling_upper = get_idx_minor(gas_name_list, minor_scaling_gas_upper)

        kminor_start_lower = self.kdist["kminor_start_lower"].values
        kminor_start_upper = self.kdist["kminor_start_upper"].values

        tau_absorption = compute_tau_absorption(
            idx_h2o,
            gpoint_flavor,
            band_lims_gpt,
            kmajor,
            kminor_lower,
            kminor_upper,
            minor_limits_gpt_lower,
            minor_limits_gpt_upper,
            minor_scales_with_density_lower,
            minor_scales_with_density_upper,
            scale_by_complement_lower,
            scale_by_complement_upper,
            idx_minor_lower,
            idx_minor_upper,
            idx_minor_scaling_lower,
            idx_minor_scaling_upper,
            kminor_start_lower,
            kminor_start_upper,
            self.tropo,
            self.col_mix,
            self.fmajor,
            self.fminor,
            self.play,
            self.tlay,
            self.col_gas,
            self.jeta,
            self.jtemp,
            self.jpress,
        )

        if self.source_is_internal:
            self.tau = tau_absorption
            self.ssa = np.full_like(tau_absorption, np.nan)
            self.g = np.full_like(tau_absorption, np.nan)
        else:
            krayl = krayl_from_kdist(self.kdist)
            tau_rayleigh = compute_tau_rayleigh(
                gpoint_flavor,
                band_lims_gpt,
                krayl,
                idx_h2o,
                self.col_gas[:, :, 0],
                self.col_gas,
                self.fminor,
                self.jeta,
                self.tropo,
                self.jtemp,
            )
            self.tau, self.ssa, self.g = combine_abs_and_rayleigh(
                tau_absorption, tau_rayleigh
            )

    def compute_solar_variability(self):
        """Calculate the solar variability

        Returns:
            np.ndarray: Solar source
        """

        a_offset = 0.1495954
        b_offset = 0.00066696

        solar_source_quiet = self.kdist["solar_source_quiet"]
        solar_source_facular = self.kdist["solar_source_facular"]
        solar_source_sunspot = self.kdist["solar_source_sunspot"]

        mg_index = self.kdist["mg_default"]
        sb_index = self.kdist["sb_default"]

        self.solar_source = (
            solar_source_quiet
            + (mg_index - a_offset) * solar_source_facular
            + (sb_index - b_offset) * solar_source_sunspot
        )
