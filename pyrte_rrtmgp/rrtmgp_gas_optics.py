import os
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pyrte_rrtmgp import data_validation
from pyrte_rrtmgp.data_validation import AtmosphericMapping, GasMapping, create_default_mapping
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
from pyrte_rrtmgp.utils import logger


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
    def __init__(
        self,
        xarray_obj,
        is_internal,
        selected_gases: list[str] | None = None,
    ):
        self._dataset = xarray_obj

        self.is_internal = is_internal

        # Get the gas names from the dataset
        self._gas_names = self.extract_names(self._dataset["gas_names"].values)

        if selected_gases is not None:
            # Filter gas names to only include those that exist in the dataset
            available_gases = tuple(g for g in selected_gases if g in self._gas_names)
            
            # Log warning for any gases that weren't found
            missing_gases = set(selected_gases) - set(available_gases)
            for gas in missing_gases:
                logger.warning(f"Gas {gas} not found in gas optics file")

            self._gas_names = available_gases


        if "h2o" not in self._gas_names:
            raise ValueError(
                "'h2o' must be included in gas mapping as it is required to compute Dry air"
            )

        # Set the gas names as coordinate in the dataset
        self._dataset.coords["absorber_ext"] = np.array(("dry_air",) + self._gas_names)

    def _initialize_pressure_levels(self, atmosphere, inplace=True):
        """Initialize pressure levels with minimum pressure adjustment"""
        pres_level_var = atmosphere.mapping.get_var("pres_level")

        min_index = np.argmin(atmosphere[pres_level_var].data)
        min_press = self._dataset["press_ref"].min().item() + sys.float_info.epsilon
        atmosphere[pres_level_var][:, min_index] = min_press

        if not inplace:
            return atmosphere

    @property
    def _selected_gas_names(self):
        return list(self._gas_names)

    @property
    def _selected_gas_names_ext(self):
        return ["dry_air"] + self._selected_gas_names

    def get_gases_columns(self, atmosphere, gas_name_map):
        pres_level_var = atmosphere.mapping.get_var("pres_level")

        gas_values = []
        for gas_map in gas_name_map.values():
            if gas_map in atmosphere.data_vars:
                values = atmosphere[gas_map]
                if hasattr(values, "units"):
                    values = values * float(values.units)
                if values.ndim == 0:
                    values = xr.full_like(
                        atmosphere[pres_level_var].isel(level=0), values
                    )
            else:
                values = xr.zeros_like(atmosphere[pres_level_var].isel(level=0))
            gas_values.append(values)

        gas_values = xr.concat(
            gas_values, dim=pd.Index(gas_name_map.keys(), name="gas")
        )

        col_dry = self.get_col_dry(
            gas_values.sel(gas="h2o"), atmosphere[pres_level_var], latitude=None
        )

        gas_values = gas_values * col_dry
        gas_values = xr.concat(
            [col_dry.expand_dims(gas=["dry_air"]), gas_values], dim="gas"
        )

        return gas_values

    def compute_problem(self, atmosphere, gas_interpolation_data):
        raise NotImplementedError()

    def compute_sources(self, atmosphere, gas_interpolation_data):
        raise NotImplementedError()

    def compute_boundary_conditions(self, atmosphere):
        raise NotImplementedError()

    def interpolate(self, atmosphere, gas_name_map) -> xr.Dataset:
        # Get the gas columns from atmospheric conditions
        gas_order = self._selected_gas_names_ext
        gases_columns = self.get_gases_columns(atmosphere, gas_name_map).sel(
            gas=gas_order
        )

        site_dim = atmosphere.mapping.get_dim("site")
        layer_dim = atmosphere.mapping.get_dim("layer")

        (jtemp, fmajor, fminor, col_mix, tropo, jeta, jpress) = interpolation(
            neta=self._dataset["mixing_fraction"].size,
            flavor=self.flavors_sets.transpose("pair", "flavor"),
            press_ref=self._dataset["press_ref"],
            temp_ref=self._dataset["temp_ref"],
            press_ref_trop=self._dataset["press_ref_trop"],
            vmr_ref=self._dataset["vmr_ref"]
            .sel(absorber_ext=gas_order)
            .transpose("atmos_layer", "absorber_ext", "temperature"),
            play=atmosphere["pres_layer"].transpose(site_dim, layer_dim),
            tlay=atmosphere["temp_layer"].transpose(site_dim, layer_dim),
            col_gas=gases_columns.sel(gas=gas_order).transpose(site_dim, layer_dim, "gas"),
        )

        # Create and return the dataset
        return xr.Dataset(
            data_vars={
                "temperature_index": (["site", "layer"], jtemp),
                "pressure_index": (["site", "layer"], jpress),
                "tropopause_mask": (["site", "layer"], tropo),
                "eta_index": (["pair", "site", "layer", "flavor"], jeta),
                "column_mix": (["temp_interp", "site", "layer", "flavor"], col_mix),
                "fmajor": (
                    [
                        "eta_interp",
                        "pressure_interp",
                        "temp_interp",
                        "site",
                        "layer",
                        "flavor",
                    ],
                    fmajor,
                ),
                "fminor": (
                    ["eta_interp", "temp_interp", "site", "layer", "flavor"],
                    fminor,
                ),
                "gases_columns": (["gas", "site", "layer"], gases_columns.values),
            },
            coords={
                "site": atmosphere[site_dim],
                "layer": atmosphere[layer_dim],
                "flavor": np.arange(fmajor.shape[-1]),
                "eta_interp": ["lower", "upper"],
                "pressure_interp": ["lower", "upper"],
                "temp_interp": ["lower", "upper"],
                "pair": [0, 1],
                "gas": gases_columns.gas,
            },
        )

    def tau_absorption(self, atmosphere, gas_interpolation_data):
        minor_gases_lower = self.extract_names(self._dataset["minor_gases_lower"].data)
        minor_gases_upper = self.extract_names(self._dataset["minor_gases_upper"].data)
        # check if the index is correct
        idx_minor_lower = self.get_idx_minor(minor_gases_lower)
        idx_minor_upper = self.get_idx_minor(minor_gases_upper)

        scaling_gas_lower = self.extract_names(self._dataset["scaling_gas_lower"].data)
        scaling_gas_upper = self.extract_names(self._dataset["scaling_gas_upper"].data)

        idx_minor_scaling_lower = self.get_idx_minor(scaling_gas_lower)
        idx_minor_scaling_upper = self.get_idx_minor(scaling_gas_upper)

        site_dim = atmosphere.mapping.get_dim("site")
        layer_dim = atmosphere.mapping.get_dim("layer")
        pres_layer_var = atmosphere.mapping.get_var("pres_layer")
        temp_layer_var = atmosphere.mapping.get_var("temp_layer")


        tau_absorption = compute_tau_absorption(
            self._selected_gas_names_ext.index("h2o"),
            self.gpoint_flavor.transpose("atmos_layer", "gpt"),
            self._dataset["bnd_limits_gpt"].transpose("pair", "bnd"),
            self._dataset["kmajor"].transpose(
                "temperature", "mixing_fraction", "pressure_interp", "gpt"
            ),
            self._dataset["kminor_lower"].transpose(
                "temperature", "mixing_fraction", "contributors_lower"
            ),
            self._dataset["kminor_upper"].transpose(
                "temperature", "mixing_fraction", "contributors_upper"
            ),
            self._dataset["minor_limits_gpt_lower"].transpose(
                "pair", "minor_absorber_intervals_lower"
            ),
            self._dataset["minor_limits_gpt_upper"].transpose(
                "pair", "minor_absorber_intervals_upper"
            ),
            self._dataset["minor_scales_with_density_lower"],
            self._dataset["minor_scales_with_density_upper"],
            self._dataset["scale_by_complement_lower"],
            self._dataset["scale_by_complement_upper"],
            idx_minor_lower,
            idx_minor_upper,
            idx_minor_scaling_lower,
            idx_minor_scaling_upper,
            self._dataset["kminor_start_lower"],
            self._dataset["kminor_start_upper"],
            gas_interpolation_data["tropopause_mask"].transpose("site", "layer"),
            gas_interpolation_data["column_mix"].transpose(
                "temp_interp", "site", "layer", "flavor"
            ),
            gas_interpolation_data["fmajor"].transpose(
                "eta_interp",
                "pressure_interp",
                "temp_interp",
                "site",
                "layer",
                "flavor",
            ),
            gas_interpolation_data["fminor"].transpose(
                "eta_interp", "temp_interp", "site", "layer", "flavor"
            ),
            atmosphere[pres_layer_var].transpose(site_dim, layer_dim),
            atmosphere[temp_layer_var].transpose(site_dim, layer_dim),
            gas_interpolation_data["gases_columns"].transpose("site", "layer", "gas"),
            gas_interpolation_data["eta_index"].transpose(
                "pair", "site", "layer", "flavor"
            ),
            gas_interpolation_data["temperature_index"].transpose("site", "layer"),
            gas_interpolation_data["pressure_index"].transpose("site", "layer"),
        )

        # Create xarray Dataset with the computed values
        return xr.Dataset(
            {
                "tau": (["site", "layer", "gpt"], tau_absorption),
            },
            coords={
                "site": atmosphere[site_dim],
                "layer": atmosphere[layer_dim],
                "gpt": self._dataset.gpt,
            },
        )

    @property
    def gpoint_flavor(self) -> xr.DataArray:
        """Get the g-point flavors from the k-distribution file.

        Each g-point is associated with a flavor, which is a pair of key species.

        Returns:
            np.ndarray: G-point flavors.
        """
        band_sizes = (
            self._dataset["bnd_limits_gpt"].values[:, 1]
            - self._dataset["bnd_limits_gpt"].values[:, 0]
            + 1
        )
        gpoint_bands = xr.DataArray(
            np.repeat(np.arange(1, len(band_sizes) + 1), band_sizes),
            dims=["gpt"],
            coords={"gpt": self._dataset.gpt},
        )

        # key_species = self._dataset["key_species"]
        key_species_rep = xr.where(
            (self._dataset["key_species"] == 0).all("pair"),
            np.array([2, 2]),
            self._dataset["key_species"],
        )

        matches = (self.flavors_sets == key_species_rep).all(dim="pair")
        match_indices = (
            matches.argmax(dim="flavor") + 1
        )  # +1 because flavors are 1-indexed
        # Create a mapping from band number to flavor index
        band_to_flavor = match_indices.sel(bnd=np.arange(len(band_sizes)))
        # Map each g-point to its corresponding flavor using the band number
        return band_to_flavor.sel(bnd=gpoint_bands - 1)

    @property
    def flavors_sets(self) -> npt.NDArray:
        """Get the unique flavors from the k-distribution file.

        Returns:
            np.ndarray: Unique flavors.
        """
        # Calculate total number of flavors and pairs
        n_bands = self._dataset["bnd"].size
        n_layers = self._dataset["atmos_layer"].size
        n_pairs = self._dataset["pair"].size
        tot_flavors = n_bands * n_layers

        # Flatten key species array
        all_flavors = np.reshape(
            self._dataset["key_species"].data, (tot_flavors, n_pairs)
        )

        # Replace (0,0) pairs with (2,2) since these g-points have zero absorption
        zero_mask = np.all(all_flavors == [0, 0], axis=1)
        all_flavors[zero_mask] = [2, 2]

        # Get unique flavors while preserving original order
        _, unique_indices = np.unique(all_flavors, axis=0, return_index=True)
        unique_flavors = all_flavors[np.sort(unique_indices)]

        # Create xarray DataArray with flavor data
        return xr.DataArray(
            unique_flavors,
            dims=["flavor", "pair"],
            coords={
                "pair": np.arange(unique_flavors.shape[1]),
                "flavor": np.arange(1, unique_flavors.shape[0] + 1),
            },
        )

    def get_idx_minor(self, minor_gases):
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
                gas_idx = self._selected_gas_names.index(gas) + 1
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
        # Convert latitude to g0 DataArray
        if latitude is not None:
            g0 = xr.DataArray(
                HELMERT1 - HELMERT2 * np.cos(2.0 * np.pi * latitude / 180.0),
                dims=["site"],
                coords={"site": plev.site},
            )
        else:
            g0 = xr.full_like(plev.isel(level=0), HELMERT1)

        # Calculate pressure difference between layers
        delta_plev = np.abs(plev.diff(dim="level")).rename({"level": "layer"})

        # Calculate factors using xarray operations
        fact = 1.0 / (1.0 + vmr_h2o)
        m_air = (M_DRY + M_H2O * vmr_h2o) * fact

        # Calculate col_dry using xarray operations
        col_dry = 10.0 * delta_plev * AVOGAD * fact / (1000.0 * m_air * 100.0 * g0)

        return col_dry.rename("dry_air")

    def compute(
        self,
        atmosphere: xr.Dataset,
        problem_type: str,
        gas_name_map: dict[str, str] | None = None,
        variable_mapping: AtmosphericMapping | None = None,
        add_to_input: bool = True,
    ):
        
        # Create and validate gas mapping
        gas_mapping = GasMapping.create(self._gas_names, gas_name_map).validate()

        if variable_mapping is None:
            variable_mapping = create_default_mapping()
        # Set mapping in accessor
        atmosphere.mapping.set_mapping(variable_mapping)

        # Get actual variable names in dataset
        pres_var = atmosphere.mapping.get_var("pres_layer")
        layer_dim = atmosphere.mapping.get_dim("layer")

        # Modify pressure levels to avoid division by zero, runs inplace
        self._initialize_pressure_levels(atmosphere)

        gas_interpolation_data = self.interpolate(atmosphere, gas_mapping)
        problem = self.compute_problem(atmosphere, gas_interpolation_data)
        sources = self.compute_sources(atmosphere, gas_interpolation_data)
        boundary_conditions = self.compute_boundary_conditions(atmosphere)

        gas_optics = xr.merge([sources, problem, boundary_conditions])

        # Add problem type to dataset attributes
        if problem_type == "absorption" and self.is_internal:
            problem_type = ProblemTypes.LW_ABSORPTION.value
        elif problem_type == "two-stream" and self.is_internal:
            problem_type = ProblemTypes.LW_2STREAM.value
        elif problem_type == "direct" and not self.is_internal:
            problem_type = ProblemTypes.SW_DIRECT.value
        elif problem_type == "two-stream" and not self.is_internal:
            problem_type = ProblemTypes.SW_2STREAM.value
        else:
            raise ValueError(f"Invalid problem type: {problem_type} for {'LW' if self.is_internal else 'SW'} radiation")

        if add_to_input:
            atmosphere.update(gas_optics)
            atmosphere.attrs["problem_type"] = problem_type
        else:
            output_ds = gas_optics
            output_ds.attrs["problem_type"] = problem_type
            return output_ds


class LWGasOpticsAccessor(BaseGasOpticsAccessor):
    """Accessor for internal radiation sources"""

    def compute_problem(self, atmosphere, gas_interpolation_data):
        return self.tau_absorption(atmosphere, gas_interpolation_data)

    def compute_sources(self, atmosphere, gas_interpolation_data):
        return self.compute_planck(atmosphere, gas_interpolation_data)

    def compute_boundary_conditions(self, atmosphere):
        if "surface_emissivity" not in atmosphere.data_vars:
            # Add surface emissivity directly to atmospheric conditions
            return xr.DataArray(
                np.ones(
                    (
                        atmosphere.sizes["site"],
                        atmosphere.sizes["gpt"],
                    )
                ),
                dims=["site", "gpt"],
                coords={
                    "site": atmosphere.site,
                    "gpt": atmosphere.gpt,
                },
            )
        else:
            return atmosphere["surface_emissivity"]

    def compute_planck(self, atmosphere, gas_interpolation_data):
        site_dim = atmosphere.mapping.get_dim("site")
        layer_dim = atmosphere.mapping.get_dim("layer")
        level_dim = atmosphere.mapping.get_dim("level")

        temp_layer_var = atmosphere.mapping.get_var("temp_layer")
        temp_level_var = atmosphere.mapping.get_var("temp_level")
        surface_temperature_var = atmosphere.mapping.get_var("surface_temperature")

        # Check if the top layer is at the first level
        top_at_1 = (
            atmosphere[layer_dim][0] < atmosphere[layer_dim][-1]
        )

        (
            sfc_src,
            lay_source,
            lev_source,
            sfc_src_jac,
        ) = compute_planck_source(
            tlay=atmosphere[temp_layer_var].transpose(site_dim, layer_dim),
            tlev=atmosphere[temp_level_var].transpose(site_dim, level_dim),
            tsfc=atmosphere[surface_temperature_var].transpose(site_dim),
            top_at_1=top_at_1,
            fmajor=gas_interpolation_data["fmajor"].transpose(
                "eta_interp",
                "pressure_interp",
                "temp_interp",
                "site",
                "layer",
                "flavor",
            ),
            jeta=gas_interpolation_data["eta_index"].transpose(
                "pair", "site", "layer", "flavor"
            ),
            tropo=gas_interpolation_data["tropopause_mask"].transpose("site", "layer"),
            jtemp=gas_interpolation_data["temperature_index"].transpose(
                "site", "layer"
            ),
            jpress=gas_interpolation_data["pressure_index"].transpose("site", "layer"),
            band_lims_gpt=self._dataset["bnd_limits_gpt"].transpose("pair", "bnd"),
            pfracin=self._dataset["plank_fraction"].transpose(
                "temperature", "mixing_fraction", "pressure_interp", "gpt"
            ),
            temp_ref_min=self._dataset["temp_ref"].min(),
            temp_ref_max=self._dataset["temp_ref"].max(),
            totplnk=self._dataset["totplnk"].transpose("temperature_Planck", "bnd"),
            gpoint_flavor=self.gpoint_flavor.transpose("atmos_layer", "gpt"),
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
                "site": atmosphere[site_dim],
                "layer": atmosphere[layer_dim],
                "level": atmosphere[level_dim],
                "gpt": self._dataset.gpt,
            },
        )


class SWGasOpticsAccessor(BaseGasOpticsAccessor):
    """Accessor for external radiation sources"""

    def compute_problem(self, atmosphere, gas_interpolation_data):
        # Calculate absorption optical depth
        tau_abs = self.tau_absorption(atmosphere, gas_interpolation_data)

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

    def compute_sources(self, atmosphere, *args, **kwargs):
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

        total_solar_irradiance = atmosphere["total_solar_irradiance"]

        toa_flux = solar_source.broadcast_like(total_solar_irradiance)
        def_tsi = toa_flux.sum(dim="gpt")
        return (toa_flux * (total_solar_irradiance / def_tsi)).rename("toa_source")

    def compute_boundary_conditions(self, atmosphere):
        solar_zenith_angle_var = atmosphere.mapping.get_var("solar_zenith_angle")
        surface_albedo_var = atmosphere.mapping.get_var("surface_albedo")
        surface_albedo_dir_var = atmosphere.mapping.get_var("surface_albedo_dir")
        surface_albedo_dif_var = atmosphere.mapping.get_var("surface_albedo_dif")

        usecol_values = atmosphere[solar_zenith_angle_var] < (
            90.0 - 2.0 * np.spacing(90.0)
        )
        usecol_values = usecol_values.rename("solar_angle_mask")
        mu0 = xr.where(
            usecol_values,
            np.cos(np.radians(atmosphere[solar_zenith_angle_var])),
            1.0,
        )
        solar_zenith_angle = mu0.broadcast_like(atmosphere.layer).rename(
            "solar_zenith_angle"
        )

        if surface_albedo_dir_var not in atmosphere.data_vars:
            surface_albedo_direct = atmosphere[surface_albedo_var]
            surface_albedo_direct = surface_albedo_direct.rename(
                "surface_albedo_direct"
            )
            surface_albedo_diffuse = atmosphere[surface_albedo_var]
            surface_albedo_diffuse = surface_albedo_diffuse.rename(
                "surface_albedo_diffuse"
            )
        else:
            surface_albedo_direct = atmosphere[surface_albedo_dir_var]
            surface_albedo_direct = surface_albedo_direct.rename(
                "surface_albedo_direct"
            )
            surface_albedo_diffuse = atmosphere[surface_albedo_dif_var]
            surface_albedo_diffuse = surface_albedo_diffuse.rename(
                "surface_albedo_diffuse"
            )

        return xr.merge(
            [
                solar_zenith_angle,
                surface_albedo_direct,
                surface_albedo_diffuse,
                usecol_values,
            ]
        )

    def tau_rayleigh(self, gas_interpolation_data):
        krayl = xr.concat(
            [self._dataset["rayl_lower"], self._dataset["rayl_upper"]],
            dim=pd.Index(["lower", "upper"], name="rayl_bound"),
        )

        tau_rayleigh = compute_tau_rayleigh(
            self.gpoint_flavor.transpose("atmos_layer", "gpt"),
            self._dataset["bnd_limits_gpt"].transpose("pair", "bnd"),
            krayl.transpose("temperature", "mixing_fraction", "gpt", "rayl_bound"),
            self._selected_gas_names_ext.index("h2o"),
            gas_interpolation_data["gases_columns"].sel(gas="dry_air"),
            gas_interpolation_data["gases_columns"]
            .sel(gas=self._selected_gas_names_ext)
            .transpose("site", "layer", "gas"),
            gas_interpolation_data["fminor"].transpose(
                "eta_interp", "temp_interp", "site", "layer", "flavor"
            ),
            gas_interpolation_data["eta_index"].transpose(
                "pair", "site", "layer", "flavor"
            ),
            gas_interpolation_data["tropopause_mask"].transpose("site", "layer"),
            gas_interpolation_data["temperature_index"].transpose("site", "layer"),
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
