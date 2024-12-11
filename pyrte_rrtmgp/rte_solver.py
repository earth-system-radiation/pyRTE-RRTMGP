from typing import Optional

import xarray as xr

from pyrte_rrtmgp.constants import GAUSS_DS, GAUSS_WTS
from pyrte_rrtmgp.data_types import ProblemTypes
from pyrte_rrtmgp.kernels.rte import lw_solver_noscat, sw_solver_2stream


class RTESolver:
    GAUSS_DS = GAUSS_DS
    GAUSS_WTS = GAUSS_WTS

    def _compute_quadrature(
        self, problem_ds: xr.Dataset, site_dim: str, nmus: int
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute quadrature weights and secants for radiative transfer calculations.

        Args:
            problem_ds: Dataset containing the problem specification
            site_dim: Name of the site dimension in the dataset
            nmus: Number of quadrature angles to use

        Returns:
            tuple containing:
                ds (xr.DataArray): Quadrature secants (directional cosines) with dimensions
                    [site, gpt, n_quad_angs].
                weights (xr.DataArray): Quadrature weights with dimension [n_quad_angs].
        """
        n_quad_angs: int = nmus
        ncol = problem_ds.sizes[site_dim]
        ngpt = problem_ds.sizes["gpt"]

        # Extract quadrature secants for the specified number of angles
        ds: xr.DataArray = xr.DataArray(
            self.GAUSS_DS[0:n_quad_angs, n_quad_angs - 1],
            dims=["n_quad_angs"],
            coords={"n_quad_angs": range(n_quad_angs)},
        )
        # Expand dimensions to match problem size
        ds = ds.expand_dims({site_dim: ncol, "gpt": ngpt})

        # Extract quadrature weights for the specified number of angles
        weights: xr.DataArray = xr.DataArray(
            self.GAUSS_WTS[0:n_quad_angs, n_quad_angs - 1],
            dims=["n_quad_angs"],
            coords={"n_quad_angs": range(n_quad_angs)},
        )

        return ds, weights

    def _compute_lw_fluxes_absorption(
        self, problem_ds: xr.Dataset, spectrally_resolved: bool = False
    ) -> xr.Dataset:
        """Compute longwave fluxes for absorption-only radiative transfer.

        Args:
            problem_ds: Dataset containing the problem specification with required variables:
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
            spectrally_resolved: If True, return spectrally resolved fluxes.
                If False, return broadband fluxes. Defaults to False.

        Returns:
            Dataset containing the computed fluxes:
                - lw_flux_up_jacobian: Upward flux Jacobian
                - lw_flux_up_broadband: Broadband upward flux
                - lw_flux_down_broadband: Broadband downward flux
                - lw_flux_up: Spectrally resolved upward flux
                - lw_flux_down: Spectrally resolved downward flux
        """

        site_dim = problem_ds.mapping.get_dim("site")
        layer_dim = problem_ds.mapping.get_dim("layer")
        level_dim = problem_ds.mapping.get_dim("level")

        surface_emissivity_var = problem_ds.mapping.get_var("surface_emissivity")

        nmus: int = 1
        top_at_1: bool = problem_ds[layer_dim][0] < problem_ds[layer_dim][-1]

        if "incident_flux" not in problem_ds:
            incident_flux: xr.DataArray = xr.zeros_like(problem_ds["surface_source"])
        else:
            incident_flux = problem_ds["incident_flux"]

        if "gpt" not in problem_ds[surface_emissivity_var].dims:
            problem_ds[surface_emissivity_var] = problem_ds[
                surface_emissivity_var
            ].expand_dims({"gpt": problem_ds.sizes["gpt"]}, axis=1)

        ds, weights = self._compute_quadrature(problem_ds, site_dim, nmus)
        ssa: xr.DataArray = (
            problem_ds["ssa"] if "ssa" in problem_ds else problem_ds["tau"].copy()
        )
        g: xr.DataArray = (
            problem_ds["g"] if "g" in problem_ds else problem_ds["tau"].copy()
        )

        (
            solver_flux_up_jacobian,
            solver_flux_up_broadband,
            solver_flux_down_broadband,
            solver_flux_up,
            solver_flux_down,
        ) = xr.apply_ufunc(
            lw_solver_noscat,
            problem_ds.sizes[site_dim],
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
            kwargs={"do_broadband": not spectrally_resolved, "top_at_1": top_at_1},
            input_core_dims=[
                [],
                [],
                [],
                [site_dim, "gpt", "n_quad_angs"],  # ds
                ["n_quad_angs"],  # weights
                [site_dim, layer_dim, "gpt"],  # tau
                [site_dim, layer_dim, "gpt"],  # ssa
                [site_dim, layer_dim, "gpt"],  # g
                [site_dim, layer_dim, "gpt"],  # lay_source
                [site_dim, level_dim, "gpt"],  # lev_source
                [site_dim, "gpt"],  # sfc_emis
                [site_dim, "gpt"],  # sfc_src
                [site_dim, "gpt"],  # sfc_src_jac
                [site_dim, "gpt"],  # inc_flux
            ],
            output_core_dims=[
                [site_dim, level_dim],  # solver_flux_up_jacobian
                [site_dim, level_dim],  # solver_flux_up_broadband
                [site_dim, level_dim],  # solver_flux_down_broadband
                [site_dim, level_dim, "gpt"],  # solver_flux_up
                [site_dim, level_dim, "gpt"],  # solver_flux_down
            ],
            vectorize=True,
            dask="allowed",
        )

        return xr.Dataset(
            {
                "lw_flux_up_jacobian": solver_flux_up_jacobian,
                "lw_flux_up_broadband": solver_flux_up_broadband,
                "lw_flux_down_broadband": solver_flux_down_broadband,
                "lw_flux_up": solver_flux_up,
                "lw_flux_down": solver_flux_down,
            }
        )

    def _compute_sw_fluxes(
        self, problem_ds: xr.Dataset, spectrally_resolved: bool = False
    ) -> xr.Dataset:
        """Compute shortwave fluxes using two-stream solver.

        Args:
            problem_ds: Dataset containing problem definition including optical properties,
                surface properties and boundary conditions.
            spectrally_resolved: If True, return spectrally resolved fluxes.
                If False, return broadband fluxes.

        Returns:
            Dataset containing computed shortwave fluxes:
                - sw_flux_up_broadband: Upward broadband flux
                - sw_flux_down_broadband: Downward broadband flux
                - sw_flux_dir_broadband: Direct broadband flux
                - sw_flux_up: Upward spectral flux
                - sw_flux_down: Downward spectral flux
                - sw_flux_dir: Direct spectral flux
        """
        # Expand surface albedo dimensions if needed
        if "gpt" not in problem_ds["surface_albedo_direct"].dims:
            problem_ds["surface_albedo_direct"] = problem_ds[
                "surface_albedo_direct"
            ].expand_dims({"gpt": problem_ds.sizes["gpt"]}, axis=1)
        if "gpt" not in problem_ds["surface_albedo_diffuse"].dims:
            problem_ds["surface_albedo_diffuse"] = problem_ds[
                "surface_albedo_diffuse"
            ].expand_dims({"gpt": problem_ds.sizes["gpt"]}, axis=1)

        # Set diffuse incident flux
        if "incident_flux_dif" not in problem_ds:
            incident_flux_dif = xr.zeros_like(problem_ds["toa_source"])
        else:
            incident_flux_dif = problem_ds["incident_flux_dif"]

        site_dim = problem_ds.mapping.get_dim("site")
        layer_dim = problem_ds.mapping.get_dim("layer")
        level_dim = problem_ds.mapping.get_dim("level")

        # Determine vertical orientation
        top_at_1 = problem_ds[layer_dim][0] < problem_ds[layer_dim][-1]

        # Call solver
        (
            solver_flux_up_broadband,
            solver_flux_down_broadband,
            solver_flux_dir_broadband,
            solver_flux_up,
            solver_flux_down,
            solver_flux_dir,
        ) = xr.apply_ufunc(
            sw_solver_2stream,
            problem_ds.sizes[site_dim],
            problem_ds.sizes[layer_dim],
            problem_ds.sizes["gpt"],
            problem_ds["tau"],
            problem_ds["ssa"],
            problem_ds["g"],
            problem_ds[problem_ds.mapping.get_var("solar_zenith_angle")],
            problem_ds["surface_albedo_direct"],
            problem_ds["surface_albedo_diffuse"],
            problem_ds["toa_source"],
            incident_flux_dif,
            kwargs={"top_at_1": top_at_1, "do_broadband": not spectrally_resolved},
            input_core_dims=[
                [],
                [],
                [],
                [site_dim, layer_dim, "gpt"],  # tau
                [site_dim, layer_dim, "gpt"],  # ssa
                [site_dim, layer_dim, "gpt"],  # g
                [site_dim, layer_dim],  # mu0
                [site_dim, "gpt"],  # sfc_alb_dir
                [site_dim, "gpt"],  # sfc_alb_dif
                [site_dim, "gpt"],  # inc_flux_dir
                [site_dim, "gpt"],  # inc_flux_dif
            ],
            output_core_dims=[
                [site_dim, level_dim, "gpt"],  # solver_flux_up_broadband
                [site_dim, level_dim, "gpt"],  # solver_flux_down_broadband
                [site_dim, level_dim, "gpt"],  # solver_flux_dir_broadband
                [site_dim, level_dim],  # solver_flux_up
                [site_dim, level_dim],  # solver_flux_down
                [site_dim, level_dim],  # solver_flux_dir
            ],
            vectorize=True,
            dask="allowed",
        )

        # Construct output dataset
        fluxes = xr.Dataset(
            {
                "sw_flux_up_broadband": solver_flux_up_broadband,
                "sw_flux_down_broadband": solver_flux_down_broadband,
                "sw_flux_dir_broadband": solver_flux_dir_broadband,
                "sw_flux_up": solver_flux_up,
                "sw_flux_down": solver_flux_down,
                "sw_flux_dir": solver_flux_dir,
            }
        )

        return fluxes * problem_ds["solar_angle_mask"]

    def solve(
        self,
        problem_ds: xr.Dataset,
        add_to_input: bool = True,
        spectrally_resolved: bool = False,
    ) -> Optional[xr.Dataset]:
        """Solve radiative transfer problem based on problem type.

        Args:
            problem_ds: Dataset containing problem definition and inputs
            add_to_input: If True, add computed fluxes to input dataset. If False, return fluxes separately
            spectrally_resolved: If True, return spectrally resolved fluxes. If False, return broadband fluxes

        Returns:
            Dataset containing computed fluxes if add_to_input is False, None otherwise
        """
        if problem_ds.attrs["problem_type"] == ProblemTypes.LW_ABSORPTION.value:
            fluxes = self._compute_lw_fluxes_absorption(problem_ds, spectrally_resolved)
        elif problem_ds.attrs["problem_type"] == ProblemTypes.SW_2STREAM.value:
            fluxes = self._compute_sw_fluxes(problem_ds, spectrally_resolved)
        else:
            raise ValueError(
                f"Unknown problem type: {problem_ds.attrs['problem_type']}"
            )

        if add_to_input:
            problem_ds.assign_coords(fluxes.coords)
            return None
        return fluxes
