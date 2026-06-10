"""
Longwave gas-optics calculator for the Simple Spectral Model.

    The class stores spectral absorption data, wavenumber grid information,
    and molecular weights, then computes optical depth and Planck source
    terms from atmospheric xarray inputs.
"""

from typing import Tuple

import numpy as np
import xarray as xr
from defaults import MOL_WEIGHTS
from kernels import (
    compute_absorption_coeffs,
    compute_layer_mass,
    compute_planck_source,
    compute_tau,
)


class GasOptics:
    """Gas optics class for the simple spectral model (LW only for now)."""

    def __init__(
        self,
        atmos_data: xr.Dataset,
        nus: xr.DataArray,
        dnus: xr.DataArray,
        pref: float = 1.0e5,
    ) -> None:
        """
        Initialize gas-optics data for longwave calculations.

        Parameters
        ----------
        atmos_data:
            Dataset containing ``triangles`` with dimensions ``("tags", "params")``.
            Parameters are ``"nu0"``, ``"l"``, and ``"kappa0"``.

        nus:

        dnus:

        pref:

        """
        self._init_inputs(
            atmos_data=atmos_data,
            nus=nus,
            dnus=dnus,
            pref=pref,
        )

        self._validate_inputs()

        self.absorption_coeffs = compute_absorption_coeffs(
            triangles=self.triangles,
            nus=self.nus,
        )

    def _init_inputs(
        self, atmos_data: xr.Dataset, nus: xr.DataArray, dnus: xr.DataArray, pref: float
    ) -> None:
        """
        Normalize and store constructor inputs.

        Not sure if we really need this.
        If all the inputs are expected to be in form of xarray dataset,
           this is then useless.

        """
        self.atmos_data = atmos_data

        self.triangles = atmos_data["triangles"].rename(
            {"tags": "tag", "params": "param"}
        )

        self.tags = tuple(
            str(tag).lower() for tag in self.triangles.coords["tag"].values
        )

        self.gases_by_tag = xr.DataArray(
            [tag.split("-")[0] for tag in self.tags],
            dims=("tag",),
            coords={"tag": list(self.tags)},
            name="gas",
        )

        self.gases = tuple(dict.fromkeys(str(gas) for gas in self.gases_by_tag.values))

        self.triangles = self.triangles.assign_coords(
            tag=list(self.tags),
            gas=("tag", self.gases_by_tag.values),
        )

        self.nus = self._as_gpt_array(nus, "nus")
        self.nus.attrs.setdefault("units", "cm^-1")

        self.dnus = self._as_gpt_array(dnus, "dnus").assign_coords(gpt=self.nus["gpt"])
        self.dnus.attrs.setdefault("units", "cm^-1")

        self.pref = float(pref)

        self.mol_weights_by_tag = xr.DataArray(
            [MOL_WEIGHTS[gas] for gas in self.gases_by_tag.values],
            dims=("tag",),
            coords={
                "tag": list(self.tags),
                "gas": ("tag", self.gases_by_tag.values),
            },
            name="mol_weights",
            attrs={"units": "kg mol^-1"},
        )

    def _as_gpt_array(self, values: xr.DataArray | np.array, name: str) -> xr.DataArray:
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
        ``surface_temperature``, and one variable for each physical gas such as
        ``h2o`` or ``co2``. The returned objects are passed directly to
        ``compute`` so callers can use ``compute(layer)`` instead of supplying
        every atmospheric field separately.
        """
        pres_level = layer["plev"]
        pres_layer = layer["play"]
        temp_level = layer["Tlev"]
        temp_layer = self._as_layer_array(layer["Tlay"], pres_layer.dims[-1])
        surface_temperature = layer["surface_temperature"]
        vmr = layer[list(self.gases)]

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
            Dataset containing volume mixing ratio fields for each physical gas.
            For example, tags ``"h2o-rot"`` and ``"h2o-cont"`` both use
            ``vmr["h2o"]``.

        Returns
        -------
        xr.Dataset
            Dataset containing ``tau``, ``lay_source``, ``lev_source``,
            ``sfc_source``, ``nus``, and ``dnus``.
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

        vmr_by_tag = xr.concat(
            [
                self._as_layer_array(vmr[str(gas)], pres_layer.dims[-1])
                for gas in self.gases_by_tag.values
            ],
            dim=xr.IndexVariable("tag", list(self.tags)),
        )
        vmr_by_tag = vmr_by_tag.assign_coords(
            gas=("tag", self.gases_by_tag.values),
        )

        layer_mass = compute_layer_mass(
            vmr=vmr_by_tag,
            plev=pres_level,
            play=pres_layer,
            mol_weights=self.mol_weights_by_tag,
        )

        tau = compute_tau(
            absorption_coeffs=self.absorption_coeffs,
            play=pres_layer,
            pref=self.pref,
            layer_mass=layer_mass,
        )

        lay_source = compute_planck_source(
            temp_layer,
            self.nus,
            self.dnus,
        ).transpose(*temp_layer.dims, "gpt")
        lev_source = compute_planck_source(
            temp_level,
            self.nus,
            self.dnus,
        ).transpose(*temp_level.dims, "gpt")
        sfc_source = compute_planck_source(
            surface_temperature,
            self.nus,
            self.dnus,
        ).transpose(*surface_temperature.dims, "gpt")

        return xr.Dataset(
            data_vars={
                "tau": tau,
                "lay_source": lay_source,
                "lev_source": lev_source,
                "sfc_source": sfc_source,
                "nus": self.nus,
                "dnus": self.dnus,
            }
        )

    def _validate_inputs(self) -> None:
        """
        Validate initialized gas-optics inputs.

        Checks that tags are unique, gases have known molecular weights,
        triangle parameters are complete and finite, spectral grids are one
        dimensional and aligned, and all physical quantities have valid ranges.
        """
        if len(self.tags) == 0:
            raise ValueError("optics_data must contain at least one tag")

        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags must be unique")

        for gas in self.gases:
            if gas not in MOL_WEIGHTS:
                raise ValueError(f"Unknown gas name: {gas}")

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
