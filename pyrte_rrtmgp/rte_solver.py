from typing import Optional

import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import ProblemTypes
from pyrte_rrtmgp.kernels.rte import lw_solver_noscat, sw_solver_2stream
from pyrte_rrtmgp.utils import logger


class RTESolver:
    GAUSS_DS = np.reciprocal(
        np.array(
            [
                [0.6096748751, np.inf, np.inf, np.inf],
                [0.2509907356, 0.7908473988, np.inf, np.inf],
                [0.1024922169, 0.4417960320, 0.8633751621, np.inf],
                [0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973],
            ]
        )
    )

    GAUSS_WTS = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2300253764, 0.7699746236, 0.0, 0.0],
            [0.0437820218, 0.3875796738, 0.5686383044, 0.0],
            [0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087],
        ]
    )

    def _compute_quadrature(
        self, ncol: int, ngpt: int, nmus: int
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute quadrature weights and secants."""
        n_quad_angs = nmus

        # Create DataArray for ds with proper dimensions and coordinates
        ds = xr.DataArray(
            self.GAUSS_DS[0:n_quad_angs, n_quad_angs - 1],
            dims=["n_quad_angs"],
            coords={"n_quad_angs": range(n_quad_angs)},
        )
        # Broadcast to full shape
        ds = ds.expand_dims({"site": ncol, "gpt": ngpt})

        # Create DataArray for weights
        weights = xr.DataArray(
            self.GAUSS_WTS[0:n_quad_angs, n_quad_angs - 1],
            dims=["n_quad_angs"],
            coords={"n_quad_angs": range(n_quad_angs)},
        )

        return ds, weights

    def _compute_lw_fluxes_absorption(
        self, problem_ds: xr.Dataset, spectrally_resolved: bool = False
    ) -> xr.Dataset:
        nmus = 1
        top_at_1 = problem_ds["layer"][0] < problem_ds["layer"][-1]

        if "incident_flux" not in problem_ds:
            incident_flux = xr.zeros_like(problem_ds["surface_source"])
        else:
            incident_flux = problem_ds["incident_flux"]

        if "gpt" not in problem_ds["surface_emissivity"].dims:
            problem_ds["surface_emissivity"] = problem_ds[
                "surface_emissivity"
            ].expand_dims({"gpt": problem_ds.sizes["gpt"]}, axis=1)

        ds, weights = self._compute_quadrature(
            problem_ds.sizes["site"], problem_ds.sizes["gpt"], nmus
        )
        ssa = problem_ds["ssa"] if "ssa" in problem_ds else problem_ds["tau"].copy()
        g = problem_ds["g"] if "g" in problem_ds else problem_ds["tau"].copy()

        (
            solver_flux_up_jacobian,
            solver_flux_up_broadband,
            solver_flux_down_broadband,
            solver_flux_up,
            solver_flux_down,
        ) = xr.apply_ufunc(
            lw_solver_noscat,
            problem_ds.sizes["site"],
            problem_ds.sizes["layer"],
            problem_ds.sizes["gpt"],
            ds,
            weights,
            problem_ds["tau"],
            ssa,
            g,
            problem_ds["layer_source"],
            problem_ds["level_source"],
            problem_ds["surface_emissivity"],
            problem_ds["surface_source"],
            problem_ds["surface_source_jacobian"],
            incident_flux,
            kwargs={"do_broadband": not spectrally_resolved, "top_at_1": top_at_1},
            input_core_dims=[
                [],
                [],
                [],
                ["site", "gpt", "n_quad_angs"],  # ds
                ["n_quad_angs"],  # weights
                ["site", "layer", "gpt"],  # tau
                ["site", "layer", "gpt"],  # ssa
                ["site", "layer", "gpt"],  # g
                ["site", "layer", "gpt"],  # lay_source
                ["site", "level", "gpt"],  # lev_source
                ["site", "gpt"],  # sfc_emis
                ["site", "gpt"],  # sfc_src
                ["site", "gpt"],  # sfc_src_jac
                ["site", "gpt"],  # inc_flux
            ],
            output_core_dims=[
                ["site", "level"],  # solver_flux_up_jacobian
                ["site", "level"],  # solver_flux_up_broadband
                ["site", "level"],  # solver_flux_down_broadband
                ["site", "level", "gpt"],  # solver_flux_up
                ["site", "level", "gpt"],  # solver_flux_down
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
        if "gpt" not in problem_ds["surface_albedo_direct"].dims:
            problem_ds["surface_albedo_direct"] = problem_ds[
                "surface_albedo_direct"
            ].expand_dims({"gpt": problem_ds.sizes["gpt"]}, axis=1)
        if "gpt" not in problem_ds["surface_albedo_diffuse"].dims:
            problem_ds["surface_albedo_diffuse"] = problem_ds[
                "surface_albedo_diffuse"
            ].expand_dims({"gpt": problem_ds.sizes["gpt"]}, axis=1)

        if "incident_flux_dif" not in problem_ds:
            incident_flux_dif = xr.zeros_like(problem_ds["toa_source"])
        else:
            incident_flux_dif = problem_ds["incident_flux_dif"]

        top_at_1 = problem_ds["layer"][0] < problem_ds["layer"][-1]

        (
            solver_flux_up_broadband,
            solver_flux_down_broadband,
            solver_flux_dir_broadband,
            solver_flux_up,
            solver_flux_down,
            solver_flux_dir,
        ) = xr.apply_ufunc(
            sw_solver_2stream,
            problem_ds.sizes["site"],
            problem_ds.sizes["layer"],
            problem_ds.sizes["gpt"],
            problem_ds["tau"],
            problem_ds["ssa"],
            problem_ds["g"],
            problem_ds["solar_zenith_angle"],
            problem_ds["surface_albedo_direct"],
            problem_ds["surface_albedo_diffuse"],
            problem_ds["toa_source"],
            incident_flux_dif,
            kwargs={"top_at_1": top_at_1, "do_broadband": not spectrally_resolved},
            input_core_dims=[
                [],
                [],
                [],
                ["site", "layer", "gpt"],  # tau
                ["site", "layer", "gpt"],  # ssa
                ["site", "layer", "gpt"],  # g
                ["site", "layer"],  # mu0
                ["site", "gpt"],  # sfc_alb_dir
                ["site", "gpt"],  # sfc_alb_dif
                ["site", "gpt"],  # inc_flux_dir
                ["site", "gpt"],  # inc_flux_dif
            ],
            output_core_dims=[
                ["site", "level", "gpt"],  # solver_flux_up_broadband
                ["site", "level", "gpt"],  # solver_flux_down_broadband
                ["site", "level", "gpt"],  # solver_flux_dir_broadband
                ["site", "level"],  # solver_flux_up
                ["site", "level"],  # solver_flux_down
                ["site", "level"],  # solver_flux_dir
            ],
            vectorize=True,
            dask="allowed",
        )

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
        spectrally_resolved: Optional[bool] = False,
    ):
        if problem_ds.attrs["problem_type"] == ProblemTypes.LW_ABSORPTION.value:
            fluxes = self._compute_lw_fluxes_absorption(problem_ds, spectrally_resolved)
        elif problem_ds.attrs["problem_type"] == ProblemTypes.SW_2STREAM.value:
            fluxes = self._compute_sw_fluxes(problem_ds, spectrally_resolved)

        if add_to_input:
            problem_ds.assign_coords(fluxes.coords)
        else:
            return fluxes
