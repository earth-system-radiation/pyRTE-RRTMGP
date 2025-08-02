"""Gas optics utilities for pyRTE-RRTMGP."""

import logging
import os
import sys
from typing import Dict, Final, Iterable, cast

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pyrte_rrtmgp.config import DEFAULT_GAS_MAPPING
from pyrte_rrtmgp.data_types import ProblemTypes
from pyrte_rrtmgp.input_mapping import (
    AtmosphericMapping,
    create_default_mapping,
)
from pyrte_rrtmgp.kernels.rrtmgp import (
    compute_cld_from_table,
    compute_planck_source,
    compute_tau_absorption,
    compute_tau_rayleigh,
    interpolation,
)
from pyrte_rrtmgp.rrtmgp_data_files import (
    CloudOpticsFiles,
    GasOpticsFiles,
    download_rrtmgp_data,
)
from pyrte_rrtmgp.utils import safer_divide

# Gravitational parameters from Helmert's equation (m/s^2)

HELMERT1: Final[float] = 9.80665
"""Standard gravity at sea level"""

HELMERT2: Final[float] = 0.02586
"""Gravity variation with latitude"""

# Molecular masses (kg/mol)

M_DRY: Final[float] = 0.028964
"""Dry air (molecular mass in kg/mol)"""

M_H2O: Final[float] = 0.018016
"""Water vapor (molecular mass in kg/mol)"""

# Avogadro's number (molecules/mol)

AVOGAD: Final[float] = 6.02214076e23
"""Avogadro's number (molecules/mol)"""

# Solar constants for orbit calculations

SOLAR_CONSTANTS: Final[Dict[str, float]] = {
    "A_OFFSET": 0.1495954,  # Semi-major axis offset (AU)
    "B_OFFSET": 0.00066696,  # Orbital eccentricity factor
}
"""Solar constants for orbit calculations. Contains the following keys:

- ``A_OFFSET``: Semi-major axis offset (AU)
- ``B_OFFSET``: Orbital eccentricity factor
"""

logger = logging.getLogger(__name__)


class BaseGasOpticsAccessor:
    """Base class for gas optics calculations.

    This class provides common functionality for both longwave and shortwave gas optics
    calculations, including gas interpolation, optical depth computation, and handling
    of atmospheric conditions.

    Args:
        xarray_obj (xr.Dataset): Dataset containing gas optics data
        is_internal (bool): Whether this is for internal (longwave) radiation
        selected_gases (list[str] | None): List of gases to include in calculations

    Raises:
        ValueError: If missing a required gas in the gas mapping (e.g. 'co', or 'h2o').
    """

    def __init__(
        self,
        xarray_obj: xr.Dataset,
        is_internal: bool,
        selected_gases: list[str] | None = None,
    ) -> None:
        """Initialize the BaseGasOpticsAccessor.

        Args:
            xarray_obj: Dataset containing gas optics data.
            is_internal: Whether this is for internal (longwave) radiation.
            selected_gases: List of gases to include in calculations.
                If None, all gases in the file will be used.

        Raises:
            ValueError: If missing required gas in gas mapping (e.g. 'co', or 'h2o').
        """
        self._dataset = xarray_obj
        self.is_internal = is_internal

        # Get the gas names from the dataset
        self._gas_names: tuple[str, ...] = tuple(
            self.extract_names(self._dataset["gas_names"].values)
        )

        if selected_gases is not None:

            # Filter gas names to only include those that exist in the dataset
            available_gases = tuple(g for g in selected_gases if g in self._gas_names)

            # Log warning for any gases that weren't found
            missing_gases = set(selected_gases) - set(available_gases)
            for gas in missing_gases:
                logger.warning(f"Gas {gas} not found in gas optics file")

            self._gas_names = available_gases

            if "h2o" not in self._gas_names:
                raise ValueError("Dry air calc requires 'h2o' to be in gas mapping")

        # Set the gas names as coordinate in the dataset
        self._dataset.coords["absorber_ext"] = np.array(("dry_air",) + self._gas_names)

    # The following four properties return scalars as per
    #   https://stackoverflow.com/questions/78697726/systematically-turn-numpy-1d-array-of-size-1-to-scalar
    @property
    def press_min(self) -> np.float64:
        """Minimum layer pressure for which gas optics data is valid."""
        return self._dataset["press_ref"].min().values.squeeze()[()]

    @property
    def press_max(self) -> np.float64:
        """Minimum layer pressure for which gas optics data is valid."""
        return self._dataset["press_ref"].max().values.squeeze()[()]

    @property
    def temp_min(self) -> np.float64:
        """Minimum layer temperature for which gas optics data is valid."""
        return self._dataset.temp_ref.min().values.squeeze()[()]

    @property
    def temp_max(self) -> np.float64:
        """Minimum layer temperature for which gas optics data is valid."""
        return self._dataset.temp_ref.max().values.squeeze()[()]

    @property
    def required_gases(self) -> set[str]:
        """Gases for which the concentration must be specified."""
        uniq_key_species = np.unique(self._dataset.key_species.values)
        required_gases = self._dataset.gas_names.values[uniq_key_species]
        return set([g.decode().strip() for g in required_gases])

    @property
    def available_gases(self) -> set[str]:
        """Gases whose concentrations influence optical depth."""
        return set([g.decode().strip() for g in self._dataset.gas_names.values])

    @property
    def _selected_gas_names(self) -> list[str]:
        """List of selected gas names."""
        return list(self._gas_names)

    @property
    def _selected_gas_names_ext(self) -> list[str]:
        """List of selected gas names including dry air."""
        return ["dry_air"] + self._selected_gas_names

    def get_gases_columns(
        self, atmosphere: xr.Dataset, gas_name_map: dict[str, str]
    ) -> xr.DataArray:
        """Get gas columns from atmospheric conditions.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            gas_name_map: Mapping between gas names and variable names

        Returns:
            DataArray containing gas columns including dry air
        """
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

        gas_da: xr.DataArray = xr.concat(
            gas_values, dim=pd.Index(gas_name_map.keys(), name="gas"), coords="minimal"
        )

        col_dry = self.get_col_dry(gas_da.sel(gas="h2o"), atmosphere, latitude=None)

        gas_da = gas_da * col_dry
        gas_da = xr.concat(
            [col_dry.expand_dims(gas=["dry_air"]), gas_da],
            dim="gas",
        )

        return gas_da.compute()  # some chunks are not computed

    def compute_problem(
        self, atmosphere: xr.Dataset, gas_interpolation_data: xr.Dataset
    ) -> xr.Dataset:
        """Compute optical properties for radiative transfer problem.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            gas_interpolation_data: Dataset containing interpolated gas data

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    def compute_sources(
        self, atmosphere: xr.Dataset, gas_interpolation_data: xr.Dataset | None = None
    ) -> xr.Dataset:
        """Compute radiation sources.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            gas_interpolation_data: Dataset containing interpolated gas data

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    def interpolate(
        self, atmosphere: xr.Dataset, gas_name_map: dict[str, str]
    ) -> xr.Dataset:
        """Interpolate gas optics data to atmospheric conditions.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            gas_name_map: Mapping between gas names and variable names

        Returns:
            Dataset containing interpolated gas optics data
        """
        # Get the gas columns from atmospheric conditions
        gas_order = self._selected_gas_names_ext
        gases_columns = self.get_gases_columns(atmosphere, gas_name_map).sel(
            gas=gas_order
        )

        layer_dim = atmosphere.mapping.get_dim("layer")

        npres = self._dataset.sizes["pressure"]
        ntemp = self._dataset.sizes["temperature"]
        ngas = gases_columns["gas"].size
        nlay = atmosphere[layer_dim].size
        nflav = self.flavors_sets["flavor"].size
        neta = self._dataset["mixing_fraction"].size

        play = atmosphere[atmosphere.mapping.get_var("pres_layer")]
        tlay = atmosphere[atmosphere.mapping.get_var("temp_layer")]
        col_gas = gases_columns.sel(gas=gas_order)

        # Stack all non-core dimensions
        stack_dims = [d for d in tlay.dims if d not in [layer_dim]]

        # Ensure play has the same dimensions as tlay by expanding it if needed
        missing_dims = [d for d in tlay.dims if d not in play.dims]
        if missing_dims:
            # Expand play along missing dimensions
            for dim in missing_dims:
                play = play.expand_dims({dim: atmosphere[dim].size})

        play_stacked = play.compute().stack({"stacked_cols": stack_dims})
        tlay_stacked = tlay.compute().stack({"stacked_cols": stack_dims})
        col_gas_stacked = col_gas.compute().stack({"stacked_cols": stack_dims})

        jtemp, fmajor, fminor, col_mix, tropo, jeta, jpress = xr.apply_ufunc(
            interpolation,
            nlay,
            ngas,
            nflav,
            neta,
            npres,
            ntemp,
            self.flavors_sets,
            self._dataset["press_ref"],
            self._dataset["temp_ref"],
            self._dataset["press_ref_trop"],
            self._dataset["vmr_ref"].sel(absorber_ext=gas_order),
            play_stacked,
            tlay_stacked,
            col_gas_stacked,
            input_core_dims=[
                [],  # nlay
                [],  # ngas
                [],  # nflav
                [],  # neta
                [],  # npres
                [],  # ntemp
                ["pair", "flavor"],  # flavor
                ["pressure"],  # press_ref
                ["temperature"],  # temp_ref
                [],  # press_ref_trop
                ["atmos_layer", "absorber_ext", "temperature"],  # vmr_ref
                [layer_dim],  # play
                [layer_dim],  # tlay
                [layer_dim, "gas"],  # col_gas
            ],
            output_core_dims=[
                [layer_dim],  # jtemp
                [
                    "eta_interp",
                    "press_interp",
                    "temp_interp",
                    layer_dim,
                    "flavor",
                ],  # fmajor
                ["eta_interp", "temp_interp", layer_dim, "flavor"],  # fminor
                ["temp_interp", layer_dim, "flavor"],  # col_mix
                [layer_dim],  # tropo
                ["pair", layer_dim, "flavor"],  # jeta
                [layer_dim],  # jpress
            ],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "eta_interp": neta,
                    "temp_interp": ntemp,
                    "press_interp": npres,
                    "pair": 2,
                }
            },
            output_dtypes=[
                np.int32,
                np.float64,
                np.float64,
                np.float64,
                np.bool_,
                np.int32,
                np.int32,
            ],
            dask="parallelized",
        )

        interpolation_results = xr.Dataset(
            {
                "temperature_index": jtemp.unstack("stacked_cols"),
                "fmajor": fmajor.unstack("stacked_cols"),
                "fminor": fminor.unstack("stacked_cols"),
                "column_mix": col_mix.unstack("stacked_cols"),
                "tropopause_mask": tropo.unstack("stacked_cols"),
                "eta_index": jeta.unstack("stacked_cols"),
                "pressure_index": jpress.unstack("stacked_cols"),
                "gases_columns": gases_columns,
            }
        )

        interpolation_results.attrs["dataset_mapping"] = atmosphere.attrs[
            "dataset_mapping"
        ]

        return interpolation_results

    def tau_absorption(
        self, atmosphere: xr.Dataset, gas_interpolation_data: xr.Dataset
    ) -> xr.Dataset:
        """Compute absorption optical depth.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            gas_interpolation_data: Dataset containing interpolated gas data

        Returns:
            Dataset containing absorption optical depth
        """
        layer_dim = atmosphere.mapping.get_dim("layer")
        level_dim = atmosphere.mapping.get_dim("level")
        nlay = atmosphere[layer_dim].size
        ntemp = self._dataset["temperature"].size
        neta = self._dataset["mixing_fraction"].size
        npres = self._dataset["press_ref"].size
        nbnd = self._dataset["bnd"].size
        ngpt = self._dataset["gpt"].size
        ngas = gas_interpolation_data["gas"].size
        nflav = self.flavors_sets["flavor"].size

        minor_gases_lower = self.extract_names(self._dataset["minor_gases_lower"].data)
        minor_gases_upper = self.extract_names(self._dataset["minor_gases_upper"].data)

        lower_gases_mask = np.isin(minor_gases_lower, self._gas_names)
        upper_gases_mask = np.isin(minor_gases_upper, self._gas_names)

        lower_gpt_sizes = (
            (self._dataset["minor_limits_gpt_lower"].diff(dim="pair") + 1)
            .transpose()
            .values[0]
        )
        upper_gpt_sizes = (
            (self._dataset["minor_limits_gpt_upper"].diff(dim="pair") + 1)
            .transpose()
            .values[0]
        )

        upper_gases_mask_expanded = np.repeat(upper_gases_mask, upper_gpt_sizes)
        lower_gases_mask_expanded = np.repeat(lower_gases_mask, lower_gpt_sizes)

        reduced_dataset = self._dataset.isel(
            contributors_lower=lower_gases_mask_expanded
        )
        reduced_dataset = reduced_dataset.isel(
            contributors_upper=upper_gases_mask_expanded
        )
        reduced_dataset = reduced_dataset.isel(
            minor_absorber_intervals_lower=lower_gases_mask
        )
        reduced_dataset = reduced_dataset.isel(
            minor_absorber_intervals_upper=upper_gases_mask
        )

        minor_gases_lower_reduced = minor_gases_lower[lower_gases_mask]
        minor_gases_upper_reduced = minor_gases_upper[upper_gases_mask]

        nminorlower = reduced_dataset.sizes["minor_absorber_intervals_lower"]
        nminorupper = reduced_dataset.sizes["minor_absorber_intervals_upper"]
        nminorklower = reduced_dataset.sizes["contributors_lower"]
        nminorkupper = reduced_dataset.sizes["contributors_upper"]

        # check if the index is correct
        idx_minor_lower = self.get_idx_minor(minor_gases_lower_reduced)
        idx_minor_upper = self.get_idx_minor(minor_gases_upper_reduced)

        scaling_gas_lower = self.extract_names(
            reduced_dataset["scaling_gas_lower"].data
        )
        scaling_gas_upper = self.extract_names(
            reduced_dataset["scaling_gas_upper"].data
        )

        idx_minor_scaling_lower = self.get_idx_minor(scaling_gas_lower)
        idx_minor_scaling_upper = self.get_idx_minor(scaling_gas_upper)

        kminor_start_lower = self._dataset["kminor_start_lower"].isel(
            minor_absorber_intervals_lower=slice(nminorlower)
        )
        kminor_start_upper = self._dataset["kminor_start_upper"].isel(
            minor_absorber_intervals_upper=slice(nminorupper)
        )

        # Stack all non-core dimensions
        non_default_dims = [
            d for d in atmosphere.dims if d not in [layer_dim, level_dim, "gpt", "bnd"]
        ]

        atmosphere = atmosphere.stack({"stacked_cols": non_default_dims})
        play = atmosphere[atmosphere.mapping.get_var("pres_layer")]
        tlay = atmosphere[atmosphere.mapping.get_var("temp_layer")]

        gas_interpolation_data = gas_interpolation_data.stack(
            {"stacked_cols": non_default_dims}
        )

        tau_absorption = xr.apply_ufunc(
            compute_tau_absorption,
            nlay,
            nbnd,
            ngpt,
            ngas,
            nflav,
            neta,
            npres,
            ntemp,
            nminorlower,
            nminorklower,
            nminorupper,
            nminorkupper,
            self._selected_gas_names_ext.index("h2o"),
            self.gpoint_flavor,
            reduced_dataset["bnd_limits_gpt"],
            reduced_dataset["kmajor"],
            reduced_dataset["kminor_lower"],
            reduced_dataset["kminor_upper"],
            reduced_dataset["minor_limits_gpt_lower"],
            reduced_dataset["minor_limits_gpt_upper"],
            reduced_dataset["minor_scales_with_density_lower"],
            reduced_dataset["minor_scales_with_density_upper"],
            reduced_dataset["scale_by_complement_lower"],
            reduced_dataset["scale_by_complement_upper"],
            idx_minor_lower,
            idx_minor_upper,
            idx_minor_scaling_lower,
            idx_minor_scaling_upper,
            kminor_start_lower,
            kminor_start_upper,
            gas_interpolation_data["tropopause_mask"],
            gas_interpolation_data["column_mix"],
            gas_interpolation_data["fmajor"],
            gas_interpolation_data["fminor"],
            play,
            tlay,
            gas_interpolation_data["gases_columns"],
            gas_interpolation_data["eta_index"],
            gas_interpolation_data["temperature_index"],
            gas_interpolation_data["pressure_index"],
            input_core_dims=[
                [],  # nlay
                [],  # nbnd
                [],  # ngpt
                [],  # ngas
                [],  # nflav
                [],  # neta
                [],  # npres
                [],  # ntemp
                [],  # nminorlower
                [],  # nminorklower
                [],  # nminorupper
                [],  # nminorkupper
                [],  # idx_h2o
                ["atmos_layer", "gpt"],  # gpoint_flavor
                ["pair", "bnd"],  # bnd_limits_gpt
                ["temperature", "mixing_fraction", "pressure_interp", "gpt"],  # kmajor
                [
                    "temperature",
                    "mixing_fraction",
                    "contributors_lower",
                ],  # kminor_lower
                [
                    "temperature",
                    "mixing_fraction",
                    "contributors_upper",
                ],  # kminor_upper
                ["pair", "minor_absorber_intervals_lower"],  # minor_limits_gpt_lower
                ["pair", "minor_absorber_intervals_upper"],  # minor_limits_gpt_upper
                ["minor_absorber_intervals_lower"],  # minor_scales_with_density_lower
                ["minor_absorber_intervals_upper"],  # minor_scales_with_density_upper
                ["minor_absorber_intervals_lower"],  # scale_by_complement_lower
                ["minor_absorber_intervals_upper"],  # scale_by_complement_upper
                ["minor_absorber_intervals_lower"],  # idx_minor_lower
                ["minor_absorber_intervals_upper"],  # idx_minor_upper
                ["minor_absorber_intervals_lower"],  # idx_minor_scaling_lower
                ["minor_absorber_intervals_upper"],  # idx_minor_scaling_upper
                ["minor_absorber_intervals_lower"],  # kminor_start_lower
                ["minor_absorber_intervals_upper"],  # kminor_start_upper
                [layer_dim],  # tropopause_mask
                ["temp_interp", layer_dim, "flavor"],  # column_mix
                [
                    "eta_interp",
                    "press_interp",
                    "temp_interp",
                    layer_dim,
                    "flavor",
                ],  # fmajor
                ["eta_interp", "temp_interp", layer_dim, "flavor"],  # fminor
                [layer_dim],  # pres_layer
                [layer_dim],  # temp_layer
                [layer_dim, "gas"],  # gases_columns
                ["pair", layer_dim, "flavor"],  # eta_index
                [layer_dim],  # temperature_index
                [layer_dim],  # pressure_index
            ],
            output_core_dims=[[layer_dim, "gpt"]],
            output_dtypes=[np.float64],
            dask="parallelized",
        )

        tau_absorption = tau_absorption.unstack("stacked_cols")
        for var in non_default_dims + ["gpt"]:
            tau_absorption = tau_absorption.drop_vars(var)

        return tau_absorption.rename("tau").to_dataset()

    @property
    def gpoint_flavor(self) -> xr.DataArray:
        """Get the g-point flavors from the k-distribution file.

        Each g-point is associated with a flavor, which is a pair of key species.

        Returns:
            DataArray containing g-point flavors
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
    def flavors_sets(self) -> xr.DataArray:
        """Get the unique flavors from the k-distribution file.

        Returns:
            DataArray containing unique flavors
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

        # NOTE: This is a workaround for the fact that dask array
        # doesn't support np.unique with axis=0

        if isinstance(all_flavors, da.Array):
            all_flavors = all_flavors.compute()

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

    def get_idx_minor(self, minor_gases: Iterable[str]) -> npt.NDArray[np.int32]:
        """Get index of each minor gas in col_gas.

        Args:
            minor_gases: List of minor gases

        Returns:
            Array containing indices of minor gases
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
    def extract_names(names: npt.NDArray) -> tuple[str, ...]:
        """Extract names from arrays, decoding and removing the suffix.

        Args:
            names: Array of encoded names

        Returns:
            Tuple of decoded and cleaned names
        """
        if isinstance(names, da.Array):
            names = names.compute()

        output = np.array(
            [gas.tobytes().decode().strip().split("_")[0] for gas in names]
        )
        return output

    @staticmethod
    def get_col_dry(
        vmr_h2o: xr.DataArray,
        atmosphere: xr.Dataset,
        latitude: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Calculate the dry column of the atmosphere.

        Args:
            vmr_h2o: Water vapor volume mixing ratio
            atmosphere: Dataset containing atmospheric conditions
            latitude: Latitude of the location

        Returns:
            DataArray containing dry column of the atmosphere
        """
        level_dim = atmosphere.mapping.get_dim("level")
        layer_dim = atmosphere.mapping.get_dim("layer")

        non_default_dims = [
            d for d in atmosphere.dims if d not in [level_dim, layer_dim]
        ]

        plev = atmosphere[atmosphere.mapping.get_var("pres_level")]

        # Convert latitude to g0 DataArray
        if latitude is not None:
            g0 = xr.DataArray(
                HELMERT1 - HELMERT2 * np.cos(2.0 * np.pi * latitude / 180.0),
                dims=non_default_dims,
                coords={d: atmosphere[d] for d in non_default_dims},
            )
        else:
            g0 = xr.full_like(plev.isel(level=0), HELMERT1)

        # Calculate pressure difference between layers
        delta_plev = np.abs(plev.diff(dim=level_dim)).rename({level_dim: layer_dim})

        # Calculate factors using xarray operations
        fact = 1.0 / (1.0 + vmr_h2o)
        m_air = (M_DRY + M_H2O * vmr_h2o) * fact

        # Calculate col_dry using xarray operations
        col_dry = 10.0 * delta_plev * AVOGAD * fact / (1000.0 * m_air * 100.0 * g0)

        return col_dry.rename("dry_air")

    def validate_input_data(
        self,
        atmosphere: xr.Dataset,
        gas_mapping: dict,
    ) -> None:
        """Validate input data: required information is present, values are valid.

        Args:
            atmosphere: Dataset containing atmospheric conditions

        Raises:
            ValueError if data is missing or has out-of-bounds values
        """
        # layer and level dimensions should be present, nlay = nlev -1

        # Some gas concentrations are required. Are they present?
        gas_validation_set = self.required_gases - set(gas_mapping.keys())
        if len(gas_validation_set) > 0:
            raise ValueError(
                f"Missing required gases in gas mapping: {gas_validation_set}"
            )

        #  layer temperatures and pressures within temp_ref, press_ref
        #  level pressure differences > 0 (not implemented)
        pres_layer_var = atmosphere.mapping.get_var("pres_layer")
        if (
            atmosphere[pres_layer_var] < self.press_min + sys.float_info.epsilon
        ).any() or (
            atmosphere[pres_layer_var] > self.press_max - sys.float_info.epsilon
        ).any():
            raise ValueError("Layer pressures outside valid range")

        temp_layer_var = atmosphere.mapping.get_var("temp_layer")
        if (
            atmosphere[temp_layer_var] < self.temp_min + sys.float_info.epsilon
        ).any() or (
            atmosphere[temp_layer_var] > self.temp_max - sys.float_info.epsilon
        ).any():
            raise ValueError("Layer temperatures outside valid range")

        pres_level_var = atmosphere.mapping.get_var("pres_level")
        if (atmosphere[pres_level_var] < 0).any():
            raise ValueError("Level pressures less than 0")

        temp_level_var = atmosphere.mapping.get_var("temp_level")
        if (atmosphere[temp_level_var] < 0).any():
            raise ValueError("Level temperatures less than 0")

        return None

    def compute(
        self,
        atmosphere: xr.Dataset,
        problem_type: str,
        gas_name_map: dict[str, str] | None = None,
        variable_mapping: AtmosphericMapping | None = None,
        add_to_input: bool = True,
    ) -> xr.Dataset | None:
        """Compute gas optics for given atmospheric conditions.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            problem_type: Type of radiative transfer problem to solve
            gas_name_map: Optional mapping between gas names and variable names
            variable_mapping: Optional mapping for atmospheric variables
            add_to_input: Whether to add results to input dataset

        Returns:
            Dataset containing gas optics results if add_to_input=False,
            otherwise None

        Raises:
            ValueError: If problem_type is invalid
        """
        gas_mapping = {}
        if gas_name_map is None:
            for gas, valid_names in DEFAULT_GAS_MAPPING.items():
                for gas_data_name in list(atmosphere.data_vars):
                    if gas_data_name in valid_names:
                        gas_mapping[gas] = gas_data_name
        else:
            for gas in DEFAULT_GAS_MAPPING:
                if gas in list(gas_name_map.keys()):
                    gas_mapping[gas] = gas_name_map[gas]

        self._gas_names = tuple(
            k for k, v in gas_mapping.items() if v in list(atmosphere.data_vars)
        )

        if variable_mapping is None:
            variable_mapping = create_default_mapping()

        # Set mapping in accessor
        atmosphere.mapping.set_mapping(variable_mapping)

        self.validate_input_data(atmosphere, gas_mapping)

        # top_at_1 describes the ordering - is the first element in
        #   the layer dimension the top or bottom of the atmosphere?
        pres_layer_var = atmosphere.mapping.get_var("pres_layer")
        top_at_1 = (
            atmosphere[pres_layer_var].isel(layer=0)
            - atmosphere[pres_layer_var].isel(layer=-1)
        )[0] < 0

        gas_interpolation_data = self.interpolate(atmosphere, gas_mapping)
        problem = self.compute_problem(atmosphere, gas_interpolation_data)
        sources = self.compute_sources(atmosphere, gas_interpolation_data)
        spectrum = self._dataset["bnd_limits_gpt"].to_dataset()
        gas_optics = xr.merge([sources, problem, spectrum])

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
            raise ValueError(
                f"Invalid problem type: {problem_type} for "
                f"{'LW' if self.is_internal else 'SW'} radiation"
            )

        if add_to_input:
            atmosphere.update(gas_optics)
            atmosphere.attrs["problem_type"] = problem_type
            atmosphere.attrs["top_at_1"] = top_at_1
            return None
        else:
            output_ds = gas_optics
            output_ds.attrs["problem_type"] = problem_type
            output_ds.attrs["top_at_1"] = top_at_1
            output_ds.mapping.set_mapping(variable_mapping)
            return output_ds


class LWGasOpticsAccessor(BaseGasOpticsAccessor):
    """Accessor for internal (longwave) radiation sources.

    This class handles gas optics calculations specific to longwave radiation, including
    computing absorption optical depths, Planck sources, and boundary conditions.
    """

    def compute_problem(
        self, atmosphere: xr.Dataset, gas_interpolation_data: xr.Dataset
    ) -> xr.Dataset:
        """Compute absorption optical depths for longwave radiation.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            gas_interpolation_data: Dataset containing interpolated gas properties

        Returns:
            Dataset containing absorption optical depths
        """
        return self.tau_absorption(atmosphere, gas_interpolation_data)

    def compute_sources(
        self, atmosphere: xr.Dataset, gas_interpolation_data: xr.Dataset | None = None
    ) -> xr.Dataset:
        """Compute Planck source terms for longwave radiation.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            gas_interpolation_data: Dataset containing interpolated gas properties

        Returns:
            Dataset containing Planck source terms
        """
        return self.compute_planck(atmosphere, gas_interpolation_data)

    def compute_planck(
        self, atmosphere: xr.Dataset, gas_interpolation_data: xr.Dataset
    ) -> xr.Dataset:
        """Compute Planck source terms for longwave radiation.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            gas_interpolation_data: Dataset containing interpolated gas properties

        Returns:
            Dataset containing Planck source terms including surface, layer and level
              sources
        """
        layer_dim = atmosphere.mapping.get_dim("layer")
        level_dim = atmosphere.mapping.get_dim("level")

        temp_layer_var = atmosphere.mapping.get_var("temp_layer")
        temp_level_var = atmosphere.mapping.get_var("temp_level")
        surface_temperature_var = atmosphere.mapping.get_var("surface_temperature")

        # Check if the top layer is at the first level
        pres_layer_var = atmosphere.mapping.get_var("pres_layer")
        top_at_1 = (
            atmosphere[pres_layer_var].values[0, 0]
            < atmosphere[pres_layer_var].values[0, -1]
        )

        nlay = atmosphere.sizes[layer_dim]
        nbnd = self._dataset.sizes["bnd"]
        ngpt = self._dataset.sizes["gpt"]
        nflav = self.flavors_sets.sizes["flavor"]
        neta = self._dataset.sizes["mixing_fraction"]
        npres = self._dataset.sizes["pressure"]
        ntemp = self._dataset.sizes["temperature"]
        nPlanckTemp = self._dataset.sizes["temperature_Planck"]

        # stack non-default dims
        non_default_dims = [
            d for d in atmosphere.dims if d not in [layer_dim, level_dim]
        ]

        # stack gas interpolation data
        gas_interpolation_data = gas_interpolation_data.stack(
            {"stacked_cols": non_default_dims}
        )
        atmosphere = atmosphere.stack({"stacked_cols": non_default_dims})

        sfc_src, lay_source, lev_source, sfc_src_jac = xr.apply_ufunc(
            compute_planck_source,
            nlay,
            nbnd,
            ngpt,
            nflav,
            neta,
            npres,
            ntemp,
            nPlanckTemp,
            atmosphere[temp_layer_var],
            atmosphere[temp_level_var],
            atmosphere[surface_temperature_var],
            top_at_1,
            gas_interpolation_data["fmajor"],
            gas_interpolation_data["eta_index"],
            gas_interpolation_data["tropopause_mask"],
            gas_interpolation_data["temperature_index"],
            gas_interpolation_data["pressure_index"],
            self._dataset["bnd_limits_gpt"],
            self._dataset["plank_fraction"],
            self._dataset["temp_ref"].min(),
            self._dataset["temp_ref"].max(),
            self._dataset["totplnk"],
            self.gpoint_flavor,
            input_core_dims=[
                [],  # nlay
                [],  # nbnd
                [],  # ngpt
                [],  # nflav
                [],  # neta
                [],  # npres
                [],  # ntemp
                [],  # nPlanckTemp
                [layer_dim],  # tlay
                [level_dim],  # tlev
                [],  # tsfc
                [],  # top_at_1
                [
                    "eta_interp",
                    "press_interp",
                    "temp_interp",
                    layer_dim,
                    "flavor",
                ],  # fmajor
                ["pair", layer_dim, "flavor"],  # jeta
                [layer_dim],  # tropo
                [layer_dim],  # jtemp
                [layer_dim],  # jpress
                ["pair", "bnd"],  # band_lims_gpt
                ["temperature", "mixing_fraction", "pressure_interp", "gpt"],  # pfracin
                [],  # temp_ref_min
                [],  # temp_ref_max
                ["temperature_Planck", "bnd"],  # totplnk
                ["atmos_layer", "gpt"],  # gpoint_flavor
            ],
            output_core_dims=[
                ["gpt"],  # sfc_src
                [layer_dim, "gpt"],  # lay_source
                [level_dim, "gpt"],  # lev_source
                ["gpt"],  # sfc_src_jac
            ],
            output_dtypes=[np.float64, np.float64, np.float64, np.float64],
            dask="parallelized",
        )

        # TODO: should chunks be added / perserved here for surface source arrays?
        # TODO: should surface source arrays be dask arrays?
        return xr.Dataset(
            {
                "surface_source": sfc_src.unstack("stacked_cols"),
                "layer_source": lay_source.unstack("stacked_cols"),
                "level_source": lev_source.unstack("stacked_cols"),
                "surface_source_jacobian": sfc_src_jac.unstack("stacked_cols"),
            }
        )


class SWGasOpticsAccessor(BaseGasOpticsAccessor):
    """Accessor for external (shortwave) radiation sources.

    This class handles gas optics calculations specific to shortwave radiation,
    including computing absorption and Rayleigh scattering optical depths, solar
    sources, and boundary conditions.
    """

    def compute_problem(
        self, atmosphere: xr.Dataset, gas_interpolation_data: xr.Dataset
    ) -> xr.Dataset:
        """Compute optical properties for shortwave radiation.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            gas_interpolation_data: Dataset containing interpolated gas properties

        Returns:
            Dataset containing optical properties (tau, ssa, g)
        """
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

    def compute_sources(
        self,
        atmosphere: xr.Dataset,
        gas_interpolation_data: xr.Dataset | None = None,
    ) -> xr.DataArray:
        """Compute solar source terms.

        Args:
            atmosphere: Dataset containing atmospheric conditions
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            DataArray containing top-of-atmosphere solar source
        """
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

        # Check if total_solar_irradiance is available in atmosphere
        if "total_solar_irradiance" in atmosphere:
            total_solar_irradiance = atmosphere["total_solar_irradiance"]
            toa_flux = solar_source.broadcast_like(total_solar_irradiance)
            def_tsi = toa_flux.sum(dim="gpt")
            return (toa_flux * (total_solar_irradiance / def_tsi)).rename("toa_source")
        else:
            # Compute normalization factor
            norm = 1.0 / solar_source.sum(dim="gpt")
            default_tsi = self._dataset["tsi_default"]
            # Scale solar source to default TSI
            toa_source = (solar_source * default_tsi * norm).rename("toa_source")

            layer_dim = atmosphere.mapping.get_dim("layer")
            level_dim = atmosphere.mapping.get_dim("level")
            non_default_dims = [
                d for d in atmosphere.dims if d not in [layer_dim, level_dim, "gpt"]
            ]

            # Ensure play has the same dimensions as tlay by expanding it if needed
            if non_default_dims:
                for dim in non_default_dims:
                    toa_source = toa_source.expand_dims({dim: atmosphere[dim]})
            return toa_source

    def tau_rayleigh(self, gas_interpolation_data: xr.Dataset) -> xr.Dataset:
        """Compute Rayleigh scattering optical depth.

        Args:
            gas_interpolation_data: Dataset containing interpolated gas properties

        Returns:
            Dataset containing Rayleigh scattering optical depth
        """
        # Combine upper and lower Rayleigh coefficients
        krayl = xr.concat(
            [self._dataset["rayl_lower"], self._dataset["rayl_upper"]],
            dim=pd.Index(["lower", "upper"], name="rayl_bound"),
        )

        layer_dim = gas_interpolation_data.mapping.get_dim("layer")
        level_dim = gas_interpolation_data.mapping.get_dim("level")

        non_default_dims = [
            d
            for d in gas_interpolation_data.dims
            if d
            not in [
                level_dim,
                layer_dim,
                "eta_interp",
                "temp_interp",
                "flavor",
                "press_interp",
                "pair",
                "gas",
            ]
        ]

        gas_interpolation_data = gas_interpolation_data.stack(
            {"stacked_cols": non_default_dims}
        )

        tau_rayleigh = xr.apply_ufunc(
            compute_tau_rayleigh,
            gas_interpolation_data.sizes[layer_dim],
            self._dataset.sizes["bnd"],
            self._dataset.sizes["gpt"],
            gas_interpolation_data.sizes["gas"],
            self.flavors_sets.sizes["flavor"],
            self._dataset.sizes["mixing_fraction"],
            self._dataset.sizes["temperature"],
            self.gpoint_flavor,
            self._dataset["bnd_limits_gpt"],
            krayl,
            self._selected_gas_names_ext.index("h2o"),
            gas_interpolation_data["gases_columns"].sel(gas="dry_air"),
            gas_interpolation_data["gases_columns"].sel(
                gas=self._selected_gas_names_ext
            ),
            gas_interpolation_data["fminor"],
            gas_interpolation_data["eta_index"],
            gas_interpolation_data["tropopause_mask"],
            gas_interpolation_data["temperature_index"],
            input_core_dims=[
                [],  # nlay
                [],  # nbnd
                [],  # ngpt
                [],  # ngas
                [],  # nflav
                [],  # neta
                [],  # ntemp
                ["atmos_layer", "gpt"],  # gpoint_flavor
                ["pair", "bnd"],  # band_lims_gpt
                ["temperature", "mixing_fraction", "gpt", "rayl_bound"],  # krayl
                [],  # idx_h2o
                [layer_dim],  # col_dry
                [layer_dim, "gas"],  # col_gas
                ["eta_interp", "temp_interp", layer_dim, "flavor"],  # fminor
                ["pair", layer_dim, "flavor"],  # jeta
                [layer_dim],  # tropo
                [layer_dim],  # jtemp
            ],
            output_core_dims=[[layer_dim, "gpt"]],
            output_dtypes=[np.float64],
            dask="parallelized",
        )

        tau_rayleigh = tau_rayleigh.unstack("stacked_cols")
        for var in non_default_dims + ["gpt"]:
            tau_rayleigh = tau_rayleigh.drop_vars(var)

        return tau_rayleigh.rename("tau").to_dataset()


class GasOptics:
    """Factory class that returns appropriate GasOptics based on dataset contents.

    This class determines whether to return a longwave (LW) or shortwave (SW) gas optics
    accessor by checking for the presence of internal source variables in the dataset.

    Example usage:
    `dataset.compute_gas_optics(selected_gases=["gas_a", "gas_b"])`

    Args:
        xarray_obj (xr.Dataset): The xarray Dataset containing gas optics data
        selected_gases (list[str] | None): Optional list of gas names to include.
            If None, all gases in the dataset will be used.
    """

    def __new__(
        cls,
        file_path: str | None = None,
        gas_optics_file: GasOpticsFiles | None = None,
        selected_gases: list[str] | None = None,
    ) -> "GasOptics":
        """Initialze gas optics objectsfrom a file or predefined gas optics file.

        Load gas optics data either from a custom netCDF file or from
        a predefined gas optics file included in the RRTMGP data package. The data
        contains absorption coefficients and other optical properties needed for
        radiative transfer calculations.

        Args:
            file_path: Path to a custom gas optics netCDF file. If provided, this takes
                precedence over gas_optics_file.
            gas_optics_file: Enum specifying a predefined gas optics file from the
                RRTMGP data package. Only used if file_path is None.
            selected_gases: Optional list of gas names to include in calculations.
                If None, all gases in the file will be used.

        Returns:
            "GasOptics": Dataset containing the gas optics data with selected_gases
                stored in the attributes.

        Raises:
            ValueError: If neither file_path nor gas_optics_file is provided.
        """
        if file_path is not None:
            dataset = xr.load_dataset(file_path)
        elif gas_optics_file is not None:
            rte_rrtmgp_dir = download_rrtmgp_data()
            dataset = xr.load_dataset(
                os.path.join(rte_rrtmgp_dir, gas_optics_file.value)
            )
        else:
            raise ValueError("Either file_path or gas_optics_file must be provided")

        dataset.attrs["selected_gases"] = selected_gases
        """Return either the LW or SW accessor depending on dataset contents."""
        # Check if source is internal by looking for required LW variables
        is_internal: bool = (
            "totplnk" in dataset.data_vars and "plank_fraction" in dataset.data_vars
        )

        if is_internal:
            return cast(
                GasOptics,
                LWGasOpticsAccessor(dataset, is_internal, selected_gases),
            )
        else:
            return cast(
                GasOptics,
                SWGasOpticsAccessor(dataset, is_internal, selected_gases),
            )


class CloudOptics:
    """Accessor for computing cloud optical properties.

    This accessor allows you to compute cloud optical properties using the
    `compute_cloud_optics` method.

    Example usage:
        `dataset.compute_cloud_optics(cloud_properties)`

    Args:
        cloud_properties (xr.Dataset): Dataset containing cloud properties.
        problem_type (str): Type of problem to solve, either "two-stream" (default)
            or "absorption".
        add_to_input (bool): Whether to add the computed properties to the input
            dataset (default: False).

    Returns:
        xr.Dataset: Dataset containing optical properties for both liquid and ice
            phases.
    """

    def __init__(
        self,
        file_path: str | None = None,
        cloud_optics_file: CloudOpticsFiles | None = None,
    ) -> None:
        """Load cloud optics data from a netCDF file.

        Args:
            cloud_optics_file: Enum specifying a predefined cloud optics file from the
                RRTMGP data package. Only used if file_path is None.

        Returns:
            xr.Dataset: Dataset containing the cloud optics data.

        Raises:
            ValueError: If neither file_path nor cloud_optics_file is provided.
        """
        if file_path is not None:
            dataset = xr.load_dataset(file_path)
        elif cloud_optics_file is not None:
            rte_rrtmgp_dir = download_rrtmgp_data()
            dataset = xr.load_dataset(
                os.path.join(rte_rrtmgp_dir, cloud_optics_file.value)
            )
        else:
            raise ValueError("Either file_path or cloud_optics_file must be provided")

        self._ds = dataset

    @property
    def rel_min(self) -> np.float64:
        """Minimum liquid water effective radius."""
        return self._ds["radliq_lwr"]

    @property
    def rel_max(self) -> np.float64:
        """Maximum liquid water effective radius."""
        return self._ds["radliq_upr"]

    @property
    def dei_min(self) -> np.float64:
        """Minimum ice water effective diameter."""
        return self._ds["diamice_lwr"]

    @property
    def dei_max(self) -> np.float64:
        """Maximum ice water effective diameter."""
        return self._ds["diamice_upr"]

    def compute(
        self,
        cloud_properties: xr.Dataset,
        problem_type: str = "two-stream",
        add_to_input: bool = False,
        variable_mapping: AtmosphericMapping | None = None,
    ) -> xr.Dataset:
        """
        Compute cloud optical properties for liquid and ice clouds.

        Args:
            cloud_properties: Dataset containing cloud properties.
            lw: Whether to compute liquid water phase (True) or ice water phase
                (False).
            add_to_input (bool): Whether to add the computed properties to the
                input dataset (default: False).

        Returns:
            xr.Dataset: Dataset containing optical properties for both liquid
                and ice phases.
        """
        cloud_optics = self._ds

        if variable_mapping is None:
            variable_mapping = create_default_mapping()
        # Set mapping in accessor
        cloud_properties.mapping.set_mapping(variable_mapping)

        layer_dim = cloud_properties.mapping.get_dim("layer")

        # Get dimensions
        nlay = cloud_properties.sizes[layer_dim]

        non_default_dims = [
            d for d in cloud_properties.dims if d not in ["level", "layer", "gpt"]
        ]
        cloud_properties = cloud_properties.stack({"stacked_cols": non_default_dims})

        # Determine if we're using band-averaged or spectral properties
        gpt_dim = "nband" if "gpt" not in cloud_optics.sizes else "gpt"
        gpt_out_dim = "bnd" if gpt_dim == "nband" else "gpt"
        ngpt = cloud_optics.sizes["nband" if gpt_dim == "nband" else "gpt"]

        # Sequentially process each chunk
        # Create cloud masks
        liq_mask = cloud_properties.lwp > 0
        ice_mask = cloud_properties.iwp > 0

        # Compute optical properties using lookup tables
        # Liquid phase
        step_size = (cloud_optics.radliq_upr - cloud_optics.radliq_lwr) / (
            cloud_optics.sizes["nsize_liq"] - 1
        )

        ltau, ltaussa, ltaussag = xr.apply_ufunc(
            compute_cld_from_table,
            nlay,
            ngpt,
            liq_mask,
            cloud_properties.lwp,
            cloud_properties.rel,
            cloud_optics.sizes["nsize_liq"],
            step_size.values,
            cloud_optics.radliq_lwr.values,
            cloud_optics.extliq,
            cloud_optics.ssaliq,
            cloud_optics.asyliq,
            input_core_dims=[
                [],  # nlay
                [],  # ngpt
                [layer_dim],  # liq_mask
                [layer_dim],  # lwp
                [layer_dim],  # rel
                [],  # nsize_liq
                [],  # step_size
                [],  # radliq_lwr
                ["nsize_liq", gpt_dim],  # extliq
                ["nsize_liq", gpt_dim],  # ssaliq
                ["nsize_liq", gpt_dim],  # asyliq
            ],
            output_core_dims=[
                [layer_dim, gpt_out_dim],  # ltau
                [layer_dim, gpt_out_dim],  # ltaussa
                [layer_dim, gpt_out_dim],  # ltaussag
            ],
            output_dtypes=[np.float64, np.float64, np.float64],
            dask_gufunc_kwargs={
                "output_sizes": {
                    gpt_out_dim: ngpt,
                },
            },
            dask="parallelized",
        )

        # Ice phase
        step_size = (cloud_optics.diamice_upr - cloud_optics.diamice_lwr) / (
            cloud_optics.sizes["nsize_ice"] - 1
        )
        ice_roughness = 1

        itau, itaussa, itaussag = xr.apply_ufunc(
            compute_cld_from_table,
            nlay,
            ngpt,
            ice_mask,
            cloud_properties.iwp,
            cloud_properties.rei,
            cloud_optics.sizes["nsize_ice"],
            step_size.values,
            cloud_optics.diamice_lwr.values,
            cloud_optics.extice[ice_roughness, :, :],
            cloud_optics.ssaice[ice_roughness, :, :],
            cloud_optics.asyice[ice_roughness, :, :],
            input_core_dims=[
                [],  # nlay
                [],  # ngpt
                [layer_dim],  # ice_mask
                [layer_dim],  # iwp
                [layer_dim],  # rei
                [],  # nsize_ice
                [],  # step_size
                [],  # diamice_lwr
                ["nsize_ice", gpt_dim],  # extice
                ["nsize_ice", gpt_dim],  # ssaice
                ["nsize_ice", gpt_dim],  # asyice
            ],
            output_core_dims=[
                [layer_dim, gpt_out_dim],  # itau
                [layer_dim, gpt_out_dim],  # itaussa
                [layer_dim, gpt_out_dim],  # itaussag
            ],
            output_dtypes=[np.float64, np.float64, np.float64],
            dask_gufunc_kwargs={
                "output_sizes": {
                    gpt_out_dim: ngpt,
                },
            },
            dask="parallelized",
        )

        ltau = ltau.unstack("stacked_cols")
        ltaussa = ltaussa.unstack("stacked_cols")
        ltaussag = ltaussag.unstack("stacked_cols")
        itau = itau.unstack("stacked_cols")
        itaussa = itaussa.unstack("stacked_cols")
        itaussag = itaussag.unstack("stacked_cols")

        # Combine liquid and ice contributions
        if problem_type == "absorption":
            tau = (ltau - ltaussa) + (itau - itaussa)
            props = xr.Dataset({"tau": tau})
        else:
            tau = ltau + itau
            taussa = ltaussa + itaussa
            taussag = ltaussag + itaussag

            # Apply the function with dask awareness.
            ssa = xr.apply_ufunc(
                safer_divide,
                taussa,
                tau,
                dask="allowed",  # Allows lazy evaluation with dask arrays.
                output_dtypes=[
                    taussa.dtype
                ],  # Ensure the output data type is correctly specified.
            )

            g = xr.apply_ufunc(
                safer_divide,
                taussag,
                taussa,
                dask="allowed",  # Allows lazy evaluation with dask arrays.
                output_dtypes=[
                    taussa.dtype
                ],  # Ensure the output data type is correctly specified.
            )

            props = xr.Dataset({"tau": tau, "ssa": ssa, "g": g})

        if add_to_input:
            cloud_properties.update(props)
            return

        props.mapping.set_mapping(cloud_properties.mapping.mapping)
        return props
