"""
Longwave gas-optics calculator for the Simple Spectral Model.

    The class stores spectral absorption data, wavenumber grid information,
    and molecular weights, then computes optical depth and Planck source
    terms from atmospheric xarray inputs.
"""

from typing import Any, Final, Tuple

import numpy as np
import xarray as xr

from .. import utils
from .kernels import (
    compute_absorption_coeffs,
    compute_layer_mass,
    compute_planck_source,
    compute_tau,
)

MOL_WEIGHTS: Final[dict[str, float]] = {
    "h2o": utils.get_molmass("H2O"),
    "co2": utils.get_molmass("CO2"),
    "o3": utils.get_molmass("O3"),
}

#
# Simple spectral model from Czarnecki and Pincus 2026
#
SSM_CP26: Final[xr.Dataset] = xr.Dataset(
    coords={
        "tags": ["co2", "h2o-rot", "h2o-vr", "h2o-cont"],
        "params": ["nu0", "l", "kappa0"],
    },
    data_vars={
        "triangles": (
            ["tags", "params"],
            np.array(
                [
                    [667.5, 10.2, 500.0],
                    [150.0, 58.0, 165.0],
                    [1500.0, 60.0, 15.0],
                    [700.0, 275.0, 0.1],
                ]
            ),
        )
    },
).assign_attrs({"pref": 1000.22})


class GasOptics:
    """Gas optics class for the simple spectral model (LW only for now)."""

    def __init__(
        self,
        spectral_data: xr.Dataset,
        nus: xr.DataArray,
        dnus: xr.DataArray,
        pref: float,
    ) -> None:
        """
        Initialize gas-optics data for longwave calculations.

        Parameters
        ----------
        spectral_data:
            Dataset containing ``triangles`` with dimensions ``("tags", "params")``.
            Parameters are ``"nu0"``, ``"l"``, and ``"kappa0"``.

        nus:

        dnus:

        pref:

        """
        triangles = self._init_inputs(
            spectral_data=spectral_data,
            nus=nus,
            dnus=dnus,
            pref=pref,
        )

        self._validate_inputs(triangles)

        self.absorption_coeffs = compute_absorption_coeffs(
            triangles=triangles,
            nus=self.spectral_grid["nus"],
        ) 
        
        def _init_inputs(
        self,
        spectral_data: xr.Dataset,
        nus: xr.DataArray,
        dnus: xr.DataArray,
        pref: float,
    ) -> xr.DataArray:
        """
        Normalize and store constructor inputs.

        ``spectral_data`` is only needed during construction to validate the
        spectral triangle table and compute absorption coefficients. This
        method prepares that triangle table and returns it, but does not store
        it on the object.

        """
        triangles = spectral_data["triangles"].rename(
            {"tags": "tag", "params": "param"}
        )

        self.tags = tuple(
            str(tag).lower() for tag in triangles.coords["tag"].values
        )

        self.species_by_tag = xr.DataArray(
            [tag.split("-")[0] for tag in self.tags],
            dims=("tag",),
            coords={"tag": list(self.tags)},
            name="species",
        )

       self.species = tuple(
            dict.fromkeys(str(species) for species in self.species_by_tag.values)
        )

        triangles = triangles.assign_coords(
            tag=list(self.tags),
            species=("tag", self.species_by_tag.values),
        )

        nus = self._as_gpt_array(nus, "nus")
        nus.attrs.setdefault("units", "cm^-1")

        dnus = self._as_gpt_array(dnus, "dnus").assign_coords(gpt=nus["gpt"])
        dnus.attrs.setdefault("units", "cm^-1")

        self.spectral_grid = xr.Dataset(
            data_vars={
                "nus": nus,
                "dnus": dnus,
            }
        )

        self.pref = float(pref)

        self.mol_weights_by_tag = xr.DataArray(
            [MOL_WEIGHTS[species] for species in self.species_by_tag.values],
            dims=("tag",),
            coords={
                "tag": list(self.tags),
                "species": ("tag", self.species_by_tag.values),
            },
            name="mol_weights",
            attrs={"units": "kg mol^-1"},
        )

        return triangles

    def _as_gpt_array(self, values: Any, name: str) -> xr.DataArray:
        """
        Return a one-dimensional spectral DataArray on dimension ``gpt``.

        ``nus`` and ``dnus`` may already be xarray DataArrays or plain array-like
        inputs. This helper preserves existing xarray metadata when possible,
        renames the DataArray to ``name``, and standardizes its only dimension
        to ``gpt`` so spectral kernels can broadcast over the same coordinate.
        """
        if isinstance(values, xr.DataArray):
            array = values.rename(name)
            if array.ndim == 1 and "gpt" not in array.dims:
                array = array.rename({array.dims[0]: "gpt"})
            return array

        return xr.DataArray(values, dims=("gpt",), name=name)

    def _as_layer_array(self, values: xr.DataArray, layer_dim: str) -> xr.DataArray:
        """
        Ensure an atmospheric layer field uses the requested layer dimension.

        Gas concentrations and layer temperatures must align with the pressure
        layer coordinate used by ``pres_layer``. This helper leaves matching
        DataArrays unchanged and otherwise renames the final dimension to
        ``layer_dim``. It does not add or remove dimensions.
        """
        if values.dims[-1] == layer_dim:
            return values

        return values.rename({values.dims[-1]: layer_dim})

    def _extract_layer_inputs(self, layer: xr.Dataset) -> Tuple[
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
    ]:
        """
        Pull the standard GasOptics inputs from a single atmospheric Dataset.

        The expected dataset fields are ``plev``, ``play``, ``Tlev``, ``Tlay``,
        ``surface_temperature``, and one variable for each species such as
        ``h2o`` or ``co2``. The returned objects are passed directly to
        ``compute`` so callers can use ``compute(layer)`` instead of supplying
        every atmospheric field separately.
        """
        pres_level = layer["plev"]
        pres_layer = layer["play"]
        temp_level = layer["Tlev"]
        temp_layer = self._as_layer_array(layer["Tlay"], pres_layer.dims[-1])
        surface_temperature = layer["surface_temperature"]
        vmr = layer[list(self.species)]

        return (
            pres_level,
            pres_layer,
            temp_level,
            temp_layer,
            surface_temperature,
            vmr,
        )
    def compute(
        self,
        layer: xr.Dataset = None,
        pres_level: xr.DataArray = None,
        pres_layer: xr.DataArray = None,
        temp_level: xr.DataArray = None,
        temp_layer: xr.DataArray = None,
        surface_temperature: xr.DataArray = None,
        vmr: xr.Dataset = None,
    ) -> xr.Dataset:
        """
        Compute longwave optical depth and Planck source terms.

        Parameters
        ----------
        pres_level:
            Pressure at layer interfaces, with dimensions
            ``(column_dim, level_dim)``.

        pres_layer:
            Pressure at layer centers, with dimensions
            ``(column_dim, layer_dim)``.

        temp_level:
            Temperature at layer interfaces, with dimensions
            ``(column_dim, level_dim)``.

        temp_layer:
            Temperature at layer centers, with dimensions
            ``(column_dim, layer_dim)``.

        surface_temperature:
            Surface temperature, typically with dimension ``column_dim``.

        vmr:
            Dataset containing volume mixing ratio fields for each species.
            For example, tags ``"h2o-rot"`` and ``"h2o-cont"`` both use
            ``vmr["h2o"]``.

        Returns
        -------
        xr.Dataset
            Dataset containing ``tau``, ``lay_source``, ``lev_source``,
            ``sfc_source``, ``nus``, and ``dnus``. The dataset also has a
            ``top_at_1`` attribute. ``top_at_1`` is ``True`` when pressure
            increases with layer index, meaning the first layer is nearest the
            top of atmosphere. For a single-layer atmosphere, this ordering is
            inferred from ``pres_level`` instead of ``pres_layer``.
        """
        if layer is not None:
            (
                pres_level,
                pres_layer,
                temp_level,
                temp_layer,
                surface_temperature,
                vmr,
            ) = self._extract_layer_inputs(layer)
        elif any(
            value is None
            for value in (
                pres_level,
                pres_layer,
                temp_level,
                temp_layer,
                surface_temperature,
                vmr,
            )
        ):
            raise ValueError(
                "Either pass layer, or pass all explicit atmospheric inputs."
            )

        temp_layer = self._as_layer_array(temp_layer, pres_layer.dims[-1])

        layer_mass = compute_layer_mass(
            vmr=vmr,
            plev=pres_level,
            play=pres_layer,
            mol_weights=self.mol_weights_by_tag,
            tags=self.tags,
            species_by_tag=self.species_by_tag,
        )

        tau = compute_tau(
            absorption_coeffs=self.absorption_coeffs,
            play=pres_layer,
            pref=self.pref,
            layer_mass=layer_mass,
        )

        lay_source = compute_planck_source(
            temp_layer,
            self.spectral_grid["nus"],
            self.spectral_grid["dnus"],
        ).transpose(*temp_layer.dims, "gpt")
        lev_source = compute_planck_source(
            temp_level,
            self.spectral_grid["nus"],
            self.spectral_grid["dnus"],
        ).transpose(*temp_level.dims, "gpt")
        sfc_source = compute_planck_source(
            surface_temperature,
            self.spectral_grid["nus"],
            self.spectral_grid["dnus"],
        ).transpose(*surface_temperature.dims, "gpt")

        p = pres_layer if pres_layer.sizes[pres_layer.dims[-1]] > 1 else pres_level
        top_at_1 = bool(p.isel({p.dims[-1]: -1}) > p.isel({p.dims[-1]: 0}))

        return xr.Dataset(
            data_vars={
                "tau": tau,
                "lay_source": lay_source,
                "lev_source": lev_source,
                "sfc_source": sfc_source,
                "nus": self.spectral_grid["nus"],
                "dnus": self.spectral_grid["dnus"],
            }
        ).assign_attrs({"top_at_1": top_at_1})
    

    def _validate_inputs(self) -> None:
        """
        Validate initialized gas-optics inputs.

        Checks that tags are unique, species have known molecular weights,
        triangle parameters are complete and finite, spectral grids are one
        dimensional and aligned, and all physical quantities have valid ranges.
        """
        if len(self.tags) == 0:
            raise ValueError("optics_data must contain at least one tag")

        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags must be unique")

        for species in self.species:
            if species not in MOL_WEIGHTS:
                raise ValueError(f"Unknown species name: {species}")

        if self.triangles.ndim != 2:
            raise ValueError("triangles must be 2D with dimensions tag and param")

        if set(self.triangles.dims) != {"tag", "param"}:
            raise ValueError("triangles must have dimensions tag and param")

        required_params = {"nu0", "l", "kappa0"}
        params = set(str(p) for p in self.triangles.coords["param"].values)

        if params != required_params:
            raise ValueError("triangles params must be exactly nu0, l, and kappa0")

        if not bool(np.isfinite(self.triangles).all()):
            raise ValueError("triangles must be finite")

        kappa0 = self.triangles.sel(param="kappa0")
        ell = self.triangles.sel(param="l")

        if not bool((kappa0 >= 0).all()):
            raise ValueError("kappa0 must be >= 0")

        if not bool((ell > 0).all()):
            raise ValueError("triangle l must be > 0")

        if self.nus.ndim != 1:
            raise ValueError("nus must be 1D")

        if self.nus.sizes["gpt"] < 2:
            raise ValueError("nus must contain at least two points")

        if not bool(np.isfinite(self.nus).all()):
            raise ValueError("nus must be finite")

        if not np.all(np.diff(self.nus.values) > 0):
            raise ValueError("nus must be strictly increasing")

        if not np.isfinite(self.pref) or self.pref < 0:
            raise ValueError("pref must be finite and >= 0")
        if self.dnus.ndim != 1:
            raise ValueError("dnus must be 1D")

        if "gpt" not in self.dnus.dims:
            raise ValueError("dnus must have dimension gpt")

        if self.dnus.sizes["gpt"] != self.nus.sizes["gpt"]:
            raise ValueError("dnus must have the same length as nus")

        if not bool(np.isfinite(self.dnus).all()):
            raise ValueError("dnus must be finite")

        if not bool((self.dnus > 0).all()):
            raise ValueError("dnus must be positive")
