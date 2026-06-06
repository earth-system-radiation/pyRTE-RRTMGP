import numpy as np
import xarray as xr

from .defaults import MOL_WEIGHTS
from .kernels import (
    compute_absorption_coeffs,
    compute_layer_mass,
    compute_planck_source,
    compute_tau,
)
"""
    Longwave gas-optics calculator for the Simple Spectral Model.

    The class stores spectral absorption data, wavenumber grid information,
    and molecular weights, then computes optical depth and Planck source
    terms from atmospheric xarray inputs.
"""
class GasOptics:
    
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
    def __init__(
        self,
        atmos_data: xr.Dataset,
        nus: xr.DataArray,
        dnus: xr.DataArray,
        pref=1.0e5,
    ):
        
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
    """
    Normalize and store constructor inputs.

    Not sure if we really need this. 
    If all the inputs are expected to be in form of xarray dataset, this is then useless.

    """
    def _init_inputs(self, atmos_data, nus, dnus, pref):
    
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
            coords={"tag": self.tags},
            name="gas",
        )

        self.gases = tuple(
            dict.fromkeys(str(gas) for gas in self.gases_by_tag.values)
        )

        self.triangles = self.triangles.assign_coords(
            tag=self.tags,
            gas=("tag", self.gases_by_tag.values),
        )

        self.nus = xr.DataArray(
            nus,
            dims=("gpt",),
            name="nus",
            attrs={"units": "cm^-1"},
        )

        self.dnus = xr.DataArray(
            dnus,
            dims=("gpt",),
            coords={"gpt": self.nus["gpt"]},
            name="dnus",
            attrs={"units": "cm^-1"},
        )

        self.pref = float(pref)

        self.mol_weights_by_tag = xr.DataArray(
            [MOL_WEIGHTS[gas] for gas in self.gases_by_tag.values],
            dims=("tag",),
            coords={
                "tag": self.tags,
                "gas": ("tag", self.gases_by_tag.values),
            },
            name="mol_weights",
            attrs={"units": "kg mol^-1"},
        )
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

    def compute(
            self, 
            pres_level: xr.DataArray,
            pres_layer: xr.DataArray,
            temp_level: xr.DataArray,
            temp_layer: xr.DataArray,
            surface_temperature: xr.DataArray,
            vmr: xr.Dataset,
    ) -> xr.Dataset:
        
        

        vmr_by_tag = vmr_by_tag.assign_coords(
            gas=("tag", self.gases_by_tag.values),
        )

        vmr = vmr.assign_coords(
            gas=("tag", self.gases_by_tag.values),
        )

        layer_mass = compute_layer_mass(
            vmr=vmr,
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

        return xr.Dataset(
        data_vars={
            "tau": tau,
            "lay_source": compute_planck_source(
                temp_layer,
                self.nus,
                self.dnus,
            ),
            "lev_source": compute_planck_source(
                temp_level,
                self.nus,
                self.dnus,
            ),
            "sfc_source": compute_planck_source(
                surface_temperature,
                self.nus,
                self.dnus,
            ),
            "nus": self.nus,
            "dnus": self.dnus,
        }
        )

    """
    Validate initialized gas-optics inputs.

    Checks that tags are unique, gases have known molecular weights,
    triangle parameters are complete and finite, spectral grids are one
    dimensional and aligned, and all physical quantities have valid ranges.
    """

    def _validate_inputs(self):
        
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
        nu0 = self.triangles.sel(param="nu0")
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
