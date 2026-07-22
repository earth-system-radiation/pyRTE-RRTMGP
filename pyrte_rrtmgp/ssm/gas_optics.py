"""
Longwave gas-optics calculator for the Simple Spectral Model.

    The class stores spectral absorption data, wavenumber grid information,
    and molecular weights, then computes optical depth and Planck source
    terms from atmospheric xarray inputs.
"""

from typing import Any, Final

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

SSM_W26: Final[xr.Dataset] = xr.Dataset(
    coords={
        "tags": ["co2", "h2o-rot", "h2o-vr"],
        "params": ["nu0", "l", "kappa0"],
    },
    data_vars={
        "triangles": (
            ["tags", "params"],
            np.array([[667.0, 12.0, 110.0], [0.0, 64.0, 282.0], [1600.0, 52.0, 24.0]]),
        )
    },
).assign_attrs({"pref": 500.0})


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
            Wavenumber grid points [cm^-1].

        dnus:
            Spectral band widths [cm^-1].

        pref:
            Reference pressure [Pa].
        """
        triangles = spectral_data["triangles"].rename(
            {"tags": "tag", "params": "param"}
        )

        self.tags = tuple(str(tag) for tag in triangles.coords["tag"].values)

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
            coords={"tag": list(self.tags)},
            name="mol_weights",
            attrs={"units": "kg mol^-1"},
        )

        self._validate_inputs(triangles)

        self.absorption_coeffs = compute_absorption_coeffs(
            triangles=triangles,
            nus=self.spectral_grid["nus"],
        )

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

    def compute(
        self,
        layer: xr.Dataset,
        add_to_input: bool = False,
    ) -> xr.Dataset | None:
        """
        Compute longwave optical depth and Planck source terms.

        Parameters
        ----------
        layer:
            Atmospheric Dataset with fields ``pres_level``, ``pres_layer``,
            ``temp_level``, ``temp_layer``, ``surface_temperature``, and one
            volume mixing ratio variable per species (e.g. ``h2o``, ``co2``).
            These names match ``pyrte_rrtmgp.rrtmgp.GasOptics`` and the RTE
            driver examples.

        add_to_input:
            If ``True``, write the computed optics fields into ``layer`` in
            place and return ``None``; otherwise return them as a new Dataset.

        Returns
        -------
        xr.Dataset | None
            Dataset containing ``tau``, ``layer_source``, ``level_source``,
            ``surface_source``, ``surface_source_jacobian``, ``nus``, and
            ``dnus``, with a ``top_at_1`` attribute. ``top_at_1`` is ``True``
            when pressure increases with layer index, meaning the first layer
            is nearest the top of atmosphere. Returns ``None`` when
            ``add_to_input`` is ``True``.
        """
        pres_level = layer["pres_level"]
        pres_layer = layer["pres_layer"]
        temp_level = layer["temp_level"]
        temp_layer = layer["temp_layer"]
        surface_temperature = layer["surface_temperature"]
        vmr = layer[list(self.species)]

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

        layer_source = compute_planck_source(
            temp_layer,
            self.spectral_grid["nus"],
            self.spectral_grid["dnus"],
        ).transpose(*temp_layer.dims, "gpt")
        level_source = compute_planck_source(
            temp_level,
            self.spectral_grid["nus"],
            self.spectral_grid["dnus"],
        ).transpose(*temp_level.dims, "gpt")
        surface_source = compute_planck_source(
            surface_temperature,
            self.spectral_grid["nus"],
            self.spectral_grid["dnus"],
        ).transpose(*surface_temperature.dims, "gpt")
        # lw_solver requires a surface source Jacobian array, but only uses it
        # when do_Jacobians=True (not set by rte.RTEAccessor.solve()), so a
        # zero placeholder of the right shape is sufficient here.
        surface_source_jacobian = xr.zeros_like(surface_source)

        lay_dim = pres_layer.dims[-1]
        top_at_1 = bool(
            (pres_layer.isel({lay_dim: -1}) > pres_layer.isel({lay_dim: 0})).all()
        )

        output_ds = xr.Dataset(
            data_vars={
                "tau": tau,
                "layer_source": layer_source,
                "level_source": level_source,
                "surface_source": surface_source,
                "surface_source_jacobian": surface_source_jacobian,
                "nus": self.spectral_grid["nus"],
                "dnus": self.spectral_grid["dnus"],
            }
        ).assign_attrs({"top_at_1": top_at_1})

        if add_to_input:
            layer.update(output_ds)
            layer.attrs["top_at_1"] = top_at_1
            return None

        return output_ds

    def _validate_inputs(self, triangles: xr.DataArray) -> None:
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

        if triangles.ndim != 2:
            raise ValueError("triangles must be 2D with dimensions tag and param")

        if set(triangles.dims) != {"tag", "param"}:
            raise ValueError("triangles must have dimensions tag and param")

        required_params = {"nu0", "l", "kappa0"}
        params = set(str(p) for p in triangles.coords["param"].values)

        if params != required_params:
            raise ValueError("triangles params must be exactly nu0, l, and kappa0")

        if not bool(np.isfinite(triangles).all()):
            raise ValueError("triangles must be finite")

        kappa0 = triangles.sel(param="kappa0")
        ell = triangles.sel(param="l")

        if not bool((kappa0 >= 0).all()):
            raise ValueError("kappa0 must be >= 0")

        if not bool((ell > 0).all()):
            raise ValueError("triangle l must be > 0")

        nus = self.spectral_grid["nus"]
        dnus = self.spectral_grid["dnus"]

        if nus.ndim != 1:
            raise ValueError("nus must be 1D")

        if nus.sizes["gpt"] < 2:
            raise ValueError("nus must contain at least two points")

        if not bool(np.isfinite(nus).all()):
            raise ValueError("nus must be finite")

        if not np.all(np.diff(nus.values) > 0):
            raise ValueError("nus must be strictly increasing")

        if not np.isfinite(self.pref) or self.pref < 0:
            raise ValueError("pref must be finite and >= 0")
        if dnus.ndim != 1:
            raise ValueError("dnus must be 1D")

        if "gpt" not in dnus.dims:
            raise ValueError("dnus must have dimension gpt")

        if dnus.sizes["gpt"] != nus.sizes["gpt"]:
            raise ValueError("dnus must have the same length as nus")

        if not bool(np.isfinite(dnus).all()):
            raise ValueError("dnus must be finite")

        if not bool((dnus > 0).all()):
            raise ValueError("dnus must be positive")
