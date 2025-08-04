"""RTE solver for pyRTE-RRTMGP."""

from typing import Optional

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from pyrte_rrtmgp.kernels.rte import (
    delta_scale_2str,
    delta_scale_2str_f,
    inc_1scalar_by_1scalar_bybnd,
    inc_1scalar_by_2stream_bybnd,
    inc_2stream_by_1scalar_bybnd,
    inc_2stream_by_2stream_bybnd,
    increment_1scalar_by_1scalar,
    increment_1scalar_by_2stream,
    increment_2stream_by_1scalar,
    increment_2stream_by_2stream,
    lw_solver_noscat,
    sw_solver_2stream,
)
from pyrte_rrtmgp.utils import expand_variable_dims


def _compute_quadrature(
    problem_ds: xr.Dataset, nmus: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute quadrature weights and secants for radiative transfer calculations.

    Args:
        problem_ds: Dataset containing the problem specification
        nmus: Number of quadrature angles to use

    Returns:
        tuple containing:
            ds (xr.DataArray): Quadrature secants (directional cosines)
            with dimensions [gpt, n_quad_angs].
            weights (xr.DataArray): Quadrature weights with dimension [n_quad_angs].
    """
    # Gaussian quadrature constants for radiative transfer

    GAUSS_DS: NDArray[np.float64] = np.reciprocal(
        np.array(
            [
                [0.6096748751, np.inf, np.inf, np.inf],
                [0.2509907356, 0.7908473988, np.inf, np.inf],
                [0.1024922169, 0.4417960320, 0.8633751621, np.inf],
                [0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973],
            ]
        )
    )
    """Gaussian quadrature points for the RRTMGP radiation scheme."""

    GAUSS_WTS: NDArray[np.float64] = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2300253764, 0.7699746236, 0.0, 0.0],
            [0.0437820218, 0.3875796738, 0.5686383044, 0.0],
            [0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087],
        ]
    )

    n_quad_angs: int = nmus
    ngpt = problem_ds.sizes["gpt"]

    # Extract quadrature secants for the specified number of angles
    ds: xr.DataArray = xr.DataArray(
        GAUSS_DS[0:n_quad_angs, n_quad_angs - 1],
        dims=["n_quad_angs"],
        coords={"n_quad_angs": range(n_quad_angs)},
    )
    # Expand dimensions to match problem size
    ds = ds.expand_dims({"gpt": ngpt})

    # Extract quadrature weights for the specified number of angles
    weights: xr.DataArray = xr.DataArray(
        GAUSS_WTS[0:n_quad_angs, n_quad_angs - 1],
        dims=["n_quad_angs"],
        coords={"n_quad_angs": range(n_quad_angs)},
    )

    return ds, weights


@xr.register_dataset_accessor("rte")
class RTEAccessor:
    """Functions for manipulating and solving radiation transfer problems."""

    def __init__(self, ds: xr.Dataset):
        """Initialize the accessor."""
        self._ds = ds

    def _compute_lw_fluxes_absorption(
        self,
    ) -> xr.Dataset:
        """Compute longwave fluxes for absorption-only radiative transfer.

        Args:
            self._ds: Dataset containing the problem specification with required
            variables:
                - tau: Optical depth
                - layer_source: Layer source function
                - level_source: Level source function
                - surface_emissivity: Surface emissivity
                - surface_source: Surface source function
                - surface_source_jacobian: Surface source Jacobian
                Optional variables:
                - incident_flux: Incident flux at top of atmosphere
                - ssa: Single scattering albedo
                - g: Asymmetry parameter

        Returns:
            Dataset containing the computed fluxes:
                - lw_flux_up_broadband: Broadband upward flux
                - lw_flux_down_broadband: Broadband downward flux
                - lw_flux_up: Spectrally resolved upward flux
                - lw_flux_down: Spectrally resolved downward flux
        """
        problem_ds = self._ds
        layer_dim = problem_ds.mapping.get_dim("layer")
        level_dim = problem_ds.mapping.get_dim("level")

        surface_emissivity_var = problem_ds.mapping.get_var("surface_emissivity")

        nmus: int = 1
        top_at_1: bool = problem_ds.attrs["top_at_1"]

        if "incident_flux" not in problem_ds:
            incident_flux: xr.DataArray = xr.zeros_like(problem_ds["surface_source"])
        else:
            incident_flux = problem_ds["incident_flux"]

        non_default_dims = [
            d
            for d in problem_ds.dims
            if d not in [layer_dim, level_dim, "gpt", "bnd", "pair"]
        ]

        # Expand surface emissivity dimensions if needed
        needed_dims = non_default_dims + ["gpt"]
        problem_ds = expand_variable_dims(
            problem_ds, surface_emissivity_var, needed_dims
        )

        problem_ds = problem_ds.stack({"stacked_cols": non_default_dims})
        incident_flux = incident_flux.stack({"stacked_cols": non_default_dims})

        ds, weights = _compute_quadrature(problem_ds, nmus)
        ssa: xr.DataArray = (
            problem_ds["ssa"] if "ssa" in problem_ds else problem_ds["tau"].copy()
        )
        g: xr.DataArray = (
            problem_ds["g"] if "g" in problem_ds else problem_ds["tau"].copy()
        )

        (
            solver_flux_up_broadband,
            solver_flux_down_broadband,
            _,
            _,
        ) = xr.apply_ufunc(
            lw_solver_noscat,
            problem_ds.sizes[layer_dim],
            problem_ds.sizes["gpt"],
            ds,
            weights,
            problem_ds["tau"],
            ssa,
            g,
            problem_ds["layer_source"],
            problem_ds["level_source"],
            problem_ds[surface_emissivity_var],
            problem_ds["surface_source"],
            problem_ds["surface_source_jacobian"],
            incident_flux,
            kwargs={"do_broadband": True, "top_at_1": top_at_1},
            input_core_dims=[
                [],  # nlay
                [],  # ngpt
                ["gpt", "n_quad_angs"],  # ds
                ["n_quad_angs"],  # weights
                [layer_dim, "gpt"],  # tau
                [layer_dim, "gpt"],  # ssa
                [layer_dim, "gpt"],  # g
                [layer_dim, "gpt"],  # lay_source
                [level_dim, "gpt"],  # lev_source
                ["gpt"],  # sfc_emis
                ["gpt"],  # sfc_src
                ["gpt"],  # sfc_src_jac
                ["gpt"],  # inc_flux
            ],
            output_core_dims=[
                [level_dim],  # solver_flux_up_broadband
                [level_dim],  # solver_flux_down_broadband
                [level_dim, "gpt"],  # solver_flux_up
                [level_dim, "gpt"],  # solver_flux_down
            ],
            output_dtypes=[np.float64, np.float64, np.float64, np.float64],
            dask="parallelized",
        )

        fluxes = xr.Dataset(
            {
                "lw_flux_up": solver_flux_up_broadband.unstack("stacked_cols"),
                "lw_flux_down": solver_flux_down_broadband.unstack("stacked_cols"),
            }
        )

        transpose_order = non_default_dims + ["level"]
        return fluxes.transpose(*transpose_order)

    def _compute_sw_boundary_conditions(self) -> xr.Dataset:
        """Compute surface and solar boundary conditions.

        Args:
            self: Dataset in which self._ds contains a radiative transfer problem

        Returns:
            Dataset containing solar zenith angles, surface albedos and solar angle mask
        """
        atmosphere = self._ds
        layer_dim = atmosphere.mapping.get_dim("layer")
        solar_zenith_angle_var = atmosphere.mapping.get_var("solar_zenith_angle")
        surface_albedo_var = atmosphere.mapping.get_var("surface_albedo")
        surface_albedo_dir_var = atmosphere.mapping.get_var("surface_albedo_direct")
        surface_albedo_dif_var = atmosphere.mapping.get_var("surface_albedo_diffuse")

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

        data_vars = [
            surface_albedo_direct,
            surface_albedo_diffuse,
        ]

        if solar_zenith_angle_var in atmosphere.data_vars:
            usecol_values = atmosphere[solar_zenith_angle_var] < (
                90.0 - 2.0 * np.spacing(90.0)
            )
            if layer_dim in usecol_values.dims:
                usecol_values = usecol_values.rename("solar_angle_mask").isel(
                    {layer_dim: 0}
                )
            else:
                usecol_values = usecol_values.rename("solar_angle_mask")
            mu0 = xr.where(
                usecol_values,
                np.cos(np.radians(atmosphere[solar_zenith_angle_var])),
                1.0,
            )
            data_vars.append(mu0.rename("mu0"))
            data_vars.append(usecol_values)
        elif "mu0" in atmosphere.data_vars:
            data_vars.append(atmosphere["mu0"])
            data_vars.append(xr.DataArray(True).rename("solar_angle_mask"))

        return xr.merge(data_vars)

    def _compute_sw_fluxes(self) -> xr.Dataset:
        """Compute shortwave fluxes using two-stream solver.

        Args:
            problem_ds: Dataset containing problem definition including optical
                properties, surface properties and boundary conditions.

        Returns:
            Dataset containing computed shortwave fluxes:
                - sw_flux_up_broadband: Upward broadband flux
                - sw_flux_down_broadband: Downward broadband flux
                - sw_flux_dir_broadband: Direct broadband flux
                - sw_flux_up: Upward spectral flux
                - sw_flux_down: Downward spectral flux
                - sw_flux_dir: Direct spectral flux
        """
        problem_ds = self._ds
        layer_dim = problem_ds.mapping.get_dim("layer")
        level_dim = problem_ds.mapping.get_dim("level")

        non_default_dims = [
            d
            for d in problem_ds.dims
            if d not in [layer_dim, level_dim, "gpt", "bnd", "pair"]
        ]

        boundary_conditions = self._compute_sw_boundary_conditions()
        problem_ds = xr.merge([problem_ds, boundary_conditions])

        needed_dims = non_default_dims + ["gpt"]
        if "surface_albedo_direct" in problem_ds.data_vars:
            problem_ds = expand_variable_dims(
                problem_ds, "surface_albedo_direct", needed_dims
            )
        if "surface_albedo_diffuse" in problem_ds.data_vars:
            problem_ds = expand_variable_dims(
                problem_ds, "surface_albedo_diffuse", needed_dims
            )

        # Expand mu0 dimensions if needed
        needed_dims = non_default_dims + [layer_dim]
        if "mu0" in problem_ds.data_vars:
            problem_ds = expand_variable_dims(problem_ds, "mu0", needed_dims)

        needed_dims = non_default_dims + [level_dim]
        problem_ds = expand_variable_dims(problem_ds, "solar_angle_mask", needed_dims)

        # Set diffuse incident flux
        needed_dims = non_default_dims + ["gpt"]
        if "incident_flux_dif" not in problem_ds:
            problem_ds["incident_flux_dif"] = 0
        problem_ds = expand_variable_dims(problem_ds, "incident_flux_dif", needed_dims)
        problem_ds = expand_variable_dims(problem_ds, "toa_source", needed_dims)

        # Determine vertical orientation
        top_at_1: bool = problem_ds.attrs["top_at_1"]

        problem_ds = problem_ds.stack({"stacked_cols": non_default_dims})

        # Call solver
        (
            _,
            _,
            _,
            solver_flux_up_broadband,
            solver_flux_down_broadband,
            solver_flux_dir_broadband,
        ) = xr.apply_ufunc(
            sw_solver_2stream,
            problem_ds.sizes[layer_dim],
            problem_ds.sizes["gpt"],
            problem_ds["tau"],
            problem_ds["ssa"],
            problem_ds["g"],
            problem_ds["mu0"],
            problem_ds["surface_albedo_direct"],
            problem_ds["surface_albedo_diffuse"],
            problem_ds["toa_source"],
            problem_ds["incident_flux_dif"],
            kwargs={"top_at_1": top_at_1, "do_broadband": True},
            input_core_dims=[
                [],  # nlay
                [],  # ngpt
                [layer_dim, "gpt"],  # tau
                [layer_dim, "gpt"],  # ssa
                [layer_dim, "gpt"],  # g
                [layer_dim],  # mu0
                ["gpt"],  # sfc_alb_dir
                ["gpt"],  # sfc_alb_dif
                ["gpt"],  # inc_flux_dir
                ["gpt"],  # inc_flux_dif
            ],
            output_core_dims=[
                [level_dim, "gpt"],  # solver_flux_up
                [level_dim, "gpt"],  # solver_flux_down
                [level_dim, "gpt"],  # solver_flux_dir
                [level_dim],  # solver_flux_up_broadband
                [level_dim],  # solver_flux_down_broadband
                [level_dim],  # solver_flux_dir_broadband
            ],
            output_dtypes=[
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
            ],
            dask_gufunc_kwargs={
                "output_sizes": {level_dim: problem_ds.sizes[layer_dim] + 1}
            },
            dask="parallelized",
        )

        # Construct output dataset
        fluxes = xr.Dataset(
            {
                "sw_flux_up": solver_flux_up_broadband.unstack("stacked_cols"),
                "sw_flux_down": solver_flux_down_broadband.unstack("stacked_cols"),
                "sw_flux_dir": solver_flux_dir_broadband.unstack("stacked_cols"),
            }
        )

        transpose_order = non_default_dims + ["level"]
        fluxes = fluxes.transpose(*transpose_order)

        return fluxes * problem_ds["solar_angle_mask"].unstack("stacked_cols")

    def solve(
        self: xr.Dataset,
        add_to_input: bool = True,
    ) -> Optional[xr.Dataset]:
        """Solve radiative transfer problem based on problem type.

        Args:
            problem_ds: Dataset containing problem definition and inputs
            add_to_input: If True, add computed fluxes to input dataset. If False,
                return fluxes separately

        Returns:
            Dataset containing computed fluxes if add_to_input is False, None otherwise
        """
        do_longwave = "layer_source" in self._ds.data_vars
        do_shortwave = "toa_source" in self._ds.data_vars
        if do_longwave and not do_shortwave:
            fluxes = self._compute_lw_fluxes_absorption()
        elif do_shortwave and not do_longwave:
            fluxes = self._compute_sw_fluxes()
        else:
            raise ValueError("Can't determine problem type from source information")

        fluxes = fluxes.compute()

        if add_to_input:
            self._ds.update(fluxes)
            return None
        return fluxes

    def add_to(
        self: xr.Dataset, other: xr.Dataset, delta_scale: bool = False
    ) -> xr.Dataset:
        """
        Combine two sets of optical properties.

        This function need not commute: it's possible to add low
        spectral resolution optical properties to those at a higher
        spectral resolution but not the other way around.

        Args:
            other: Second set of optical properties to add.

        Returns:
            xr.Dataset: Combined optical properties.
        """
        op1 = self._ds
        op2 = other

        layer_dim = op2.mapping.get_dim("layer")
        level_dim = op2.mapping.get_dim("level")
        gpt_dim = "gpt"
        bnd_dim = "bnd"

        non_default_dims = [
            d
            for d in op2.dims
            if d not in [layer_dim, level_dim, gpt_dim, bnd_dim, "pair"]
        ]
        op1 = op1.stack({"stacked_cols": non_default_dims})
        op2 = op2.stack({"stacked_cols": non_default_dims})

        for var in ["tau", "ssa", "g"]:
            if var in op1.data_vars:
                gpt_dim = "gpt" if "gpt" in op1[var].sizes else "bnd"
                transposed_data = op1[var].transpose("stacked_cols", "layer", gpt_dim)
                op1[var] = (["stacked_cols", "layer", gpt_dim], transposed_data.values)
                op1[var].values = np.asfortranarray(op1[var].values)
            if var in op2.data_vars:
                gpt_dim = "gpt" if "gpt" in op2[var].sizes else "bnd"
                transposed_data = op2[var].transpose("stacked_cols", "layer", gpt_dim)
                op2[var] = (["stacked_cols", "layer", gpt_dim], transposed_data.values)
                op2[var].values = np.asfortranarray(op2[var].values)

        if delta_scale:
            self.delta_scale_optical_props(op1)

        nlay = op2.sizes[layer_dim]
        ngpt = op2.sizes[gpt_dim]

        # Check if input has only tau (1-stream) or tau, ssa, g (2-stream)
        is_1stream_1 = "tau" in list(op1.data_vars) and "ssa" not in list(op1.data_vars)
        is_1stream_2 = "tau" in list(op2.data_vars) and "ssa" not in list(op2.data_vars)

        if "gpt" in op1["tau"].sizes:
            if is_1stream_1:
                if is_1stream_2:
                    # 1-stream by 1-stream
                    xr.apply_ufunc(
                        increment_1scalar_by_1scalar,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op1["tau"],
                        input_core_dims=[
                            [],  # nlay
                            [],  # ngpt
                            ["layer", "gpt"],  # tau_inout
                            ["layer", "gpt"],  # tau_in
                        ],
                        output_dtypes=[np.float64],
                        dask="parallelized",
                    )
                else:
                    # 1-stream by 2-stream
                    xr.apply_ufunc(
                        increment_1scalar_by_2stream,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op1["tau"],
                        op1["ssa"],
                        input_core_dims=[
                            [],  # nlay
                            [],  # ngpt
                            ["layer", "gpt"],  # tau_inout
                            ["layer", "gpt"],  # tau_in
                            ["layer", "gpt"],  # ssa_in
                        ],
                        output_dtypes=[np.float64],
                        dask="parallelized",
                    )
            else:  # 2-stream output
                if is_1stream_2:
                    # 2-stream by 1-stream
                    xr.apply_ufunc(
                        increment_2stream_by_1scalar,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op2["ssa"],
                        op1["tau"],
                        input_core_dims=[
                            [],  # nlay
                            [],  # ngpt
                            ["layer", "gpt"],  # tau_inout
                            ["layer", "gpt"],  # ssa_inout
                            ["layer", "gpt"],  # tau_in
                        ],
                        output_dtypes=[np.float64],
                        dask="parallelized",
                    )
                else:
                    # 2-stream by 2-stream
                    xr.apply_ufunc(
                        increment_2stream_by_2stream,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op2["ssa"],
                        op2["g"],
                        op1["tau"],
                        op1["ssa"],
                        op1["g"],
                        input_core_dims=[
                            [],  # nlay
                            [],  # ngpt
                            ["layer", "gpt"],  # tau_inout
                            ["layer", "gpt"],  # ssa_inout
                            ["layer", "gpt"],  # g_inout
                            ["layer", "gpt"],  # tau_in
                            ["layer", "gpt"],  # ssa_in
                            ["layer", "gpt"],  # g_in
                        ],
                        output_dtypes=[np.float64],
                        dask="parallelized",
                    )
        else:
            # By-band increment (when op2's ngpt equals op1's nband)
            if op2.sizes["bnd"] != op1.sizes["bnd"]:
                raise ValueError(
                    "Incompatible g-point structures for by-band increment"
                )

            if is_1stream_1:
                if is_1stream_2:
                    # 1-stream by 1-stream by band
                    xr.apply_ufunc(
                        inc_1scalar_by_1scalar_bybnd,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op1["tau"],
                        op2.sizes["bnd"],
                        op2["bnd_limits_gpt"],
                        input_core_dims=[
                            [],  # nlay
                            [],  # ngpt
                            ["layer", "gpt"],  # tau_inout
                            ["layer", "bnd"],  # tau_in
                            [],  # nbnd
                            ["pair", "bnd"],  # band_lims_gpoint
                        ],
                        output_dtypes=[np.float64],
                        dask="parallelized",
                    )
                else:
                    # 1-stream by 2-stream by band
                    xr.apply_ufunc(
                        inc_1scalar_by_2stream_bybnd,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op1["tau"],
                        op1["ssa"],
                        op2.sizes["bnd"],
                        op2["bnd_limits_gpt"],
                        input_core_dims=[
                            [],  # nlay
                            [],  # ngpt
                            ["layer", "gpt"],  # tau_inout
                            ["layer", "bnd"],  # tau_in
                            ["layer", "bnd"],  # ssa_in
                            [],  # nbnd
                            ["pair", "bnd"],  # bnd_limits_gpt
                        ],
                        output_dtypes=[np.float64],
                        dask="parallelized",
                    )
            else:
                if is_1stream_2:
                    # 2-stream by 1-stream by band
                    xr.apply_ufunc(
                        inc_2stream_by_1scalar_bybnd,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op2["ssa"],
                        op1["tau"],
                        op2.sizes["bnd"],
                        op2["bnd_limits_gpt"],
                        input_core_dims=[
                            [],  # nlay
                            [],  # ngpt
                            ["layer", "gpt"],  # tau_inout
                            ["layer", "gpt"],  # ssa_inout
                            ["layer", "bnd"],  # tau_in
                            [],  # nbnd
                            ["pair", "bnd"],  # band_lims_gpoint
                        ],
                        output_dtypes=[np.float64],
                        dask="parallelized",
                    )
                else:
                    # 2-stream by 2-stream by band
                    xr.apply_ufunc(
                        inc_2stream_by_2stream_bybnd,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op2["ssa"],
                        op2["g"],
                        op1["tau"],
                        op1["ssa"],
                        op1["g"],
                        op2.sizes["bnd"],
                        op2["bnd_limits_gpt"],
                        input_core_dims=[
                            [],  # nlay
                            [],  # ngpt
                            ["layer", "gpt"],  # tau_inout
                            ["layer", "gpt"],  # ssa_inout
                            ["layer", "gpt"],  # g_inout
                            ["layer", "bnd"],  # tau_in
                            ["layer", "bnd"],  # ssa_in
                            ["layer", "bnd"],  # g_in
                            [],  # nbnd
                            ["pair", "bnd"],  # band_lims_gpoint
                        ],
                        output_dtypes=[np.float64],
                        dask="parallelized",
                    )

        for var in ["tau", "ssa", "g"]:
            if var in op2.data_vars:
                other[var] = op2[var].unstack("stacked_cols")

        return other

    def delta_scale_optical_props(
        self, optical_props: xr.Dataset, forward_scattering: np.ndarray | None = None
    ) -> xr.Dataset:
        """Apply delta scaling to 2-stream optical properties.

        Args:
            optical_props: xarray Dataset containing tau, ssa, and g variables
            forward_scattering: Optional array of forward scattering fraction
            (g**2 if not provided) Must have shape (ncol, nlay, ngpt) if provided

        Raises:
            ValueError: If forward_scattering array has incorrect dimensions or values
            outside [0,1]
        """
        # Get dimensions
        layer_dim = optical_props.mapping.get_dim("layer")
        nlay = optical_props.sizes[layer_dim]

        gpt_dim = "gpt" if "gpt" in optical_props.sizes else "bnd"
        ngpt = optical_props.sizes[gpt_dim]

        for var in ["tau", "ssa", "g"]:
            if var in optical_props.data_vars:
                transposed_data = optical_props[var].transpose(
                    "stacked_cols", layer_dim, gpt_dim
                )
                optical_props[var] = (
                    ["stacked_cols", layer_dim, gpt_dim],
                    transposed_data.values,
                )
                optical_props[var].values = np.asfortranarray(optical_props[var].values)

        # Call kernel with forward scattering
        if forward_scattering is not None:
            xr.apply_ufunc(
                delta_scale_2str_f,
                nlay,
                ngpt,
                optical_props["tau"],
                optical_props["ssa"],
                optical_props["g"],
                forward_scattering,
                input_core_dims=[
                    [],  # nlay
                    [],  # ngpt
                    [layer_dim, gpt_dim],  # tau
                    [layer_dim, gpt_dim],  # ssa
                    [layer_dim, gpt_dim],  # g
                    [layer_dim, gpt_dim],  # f
                ],
                output_dtypes=[np.float64],
                dask="parallelized",
            )
        else:
            xr.apply_ufunc(
                delta_scale_2str,
                nlay,
                ngpt,
                optical_props["tau"],
                optical_props["ssa"],
                optical_props["g"],
                input_core_dims=[
                    [],  # nlay
                    [],  # ngpt
                    [layer_dim, gpt_dim],  # tau
                    [layer_dim, gpt_dim],  # ssa
                    [layer_dim, gpt_dim],  # g
                ],
                output_dtypes=[np.float64],
                dask="parallelized",
            )
