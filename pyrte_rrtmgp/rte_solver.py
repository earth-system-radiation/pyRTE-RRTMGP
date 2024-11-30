import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import ProblemTypes
from pyrte_rrtmgp.kernels.rte import lw_solver_noscat, sw_solver_2stream
from pyrte_rrtmgp.utils import get_usecols


def rte_solve(problem_ds: xr.Dataset, add_to_input: bool = True):
    if problem_ds.attrs["problem_type"] == ProblemTypes.LW_ABSORPTION.value:
        _, solver_flux_up, solver_flux_down, _, _ = lw_solver_noscat(
            tau=problem_ds["tau"].values,
            lay_source=problem_ds["layer_source"].values,
            lev_source=problem_ds["level_source"].values,
            sfc_emis=problem_ds["surface_emissivity"].values,
            sfc_src=problem_ds["surface_source"].values,
            sfc_src_jac=problem_ds["surface_source_jacobian"].values,
        )

        fluxes = xr.Dataset(
            {
                "lw_flux_up": (["site", "level"], solver_flux_up),
                "lw_flux_down": (["site", "level"], solver_flux_down),
            },
            coords={
                "site": problem_ds.site,
                "level": problem_ds.level,
            },
        )

    elif problem_ds.attrs["problem_type"] == ProblemTypes.SW_2STREAM.value:
        _, _, _, flux_up, flux_down, _ = sw_solver_2stream(
            tau=problem_ds["tau"].values,
            ssa=problem_ds["ssa"].values,
            g=problem_ds["g"].values,
            mu0=problem_ds["solar_zenith_angle"].values.T,
            sfc_alb_dir=problem_ds["surface_albedo_direct"].values,
            sfc_alb_dif=problem_ds["surface_albedo_diffuse"].values,
            inc_flux_dir=problem_ds["toa_source"].values,
        )

        fluxes = xr.Dataset(
            {
                "sw_flux_up": (["site", "level"], flux_up),
                "sw_flux_down": (["site", "level"], flux_down),
            },
            coords={
                "site": problem_ds.site,
                "level": problem_ds.level,
            },
        )

        # Post-process results for nighttime columns
        if "solar_angle_mask" in problem_ds:
            fluxes["sw_flux_up"] = fluxes["sw_flux_up"] * problem_ds["solar_angle_mask"]
            fluxes["sw_flux_down"] = (
                fluxes["sw_flux_down"] * problem_ds["solar_angle_mask"]
            )

    if add_to_input:
        problem_ds.assign_coords(fluxes.coords)
    else:
        return fluxes
