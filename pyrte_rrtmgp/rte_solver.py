import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import ProblemTypes
from pyrte_rrtmgp.kernels.rte import (
    lw_solver_2stream,
    lw_solver_noscat,
    sw_solver_2stream,
    sw_solver_noscat,
)
from pyrte_rrtmgp.utils import get_usecols


def rte_solve(problem_ds: xr.Dataset, add_to_input: bool = True):
    if problem_ds.attrs["problem_type"] == ProblemTypes.LW_ABSORPTION.value:
        _, solver_flux_up, solver_flux_down, _, _ = lw_solver_noscat(
            tau=problem_ds["tau"].transpose("site", "layer", "gpt"),
            lay_source=problem_ds["layer_source"].transpose("site", "layer", "gpt"),
            lev_source=problem_ds["level_source"].transpose("site", "level", "gpt"),
            sfc_emis=problem_ds["surface_emissivity"],
            sfc_src=problem_ds["surface_source"].transpose("site", "gpt"),
            sfc_src_jac=problem_ds["surface_source_jacobian"].transpose("site", "gpt"),
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
    elif problem_ds.attrs["problem_type"] == ProblemTypes.LW_2STREAM.value:
        flux_up, flux_down = lw_solver_2stream(
            tau=problem_ds["tau"].transpose("site", "layer", "gpt"),
            ssa=problem_ds["ssa"].transpose("site", "layer", "gpt"),
            g=problem_ds["g"].transpose("site", "layer", "gpt"),
            lay_source=problem_ds["layer_source"].transpose("site", "layer", "gpt"),
            lev_source=problem_ds["level_source"].transpose("site", "level", "gpt"),
            sfc_emis=problem_ds["surface_emissivity"],
            sfc_src=problem_ds["surface_source"].transpose("site", "gpt"),
            inc_flux=problem_ds["toa_source"].transpose("site", "gpt"),
        )

        fluxes = xr.Dataset(
            {
                "lw_flux_up": (["site", "level"], flux_up),
                "lw_flux_down": (["site", "level"], flux_down),
            },
            coords={
                "site": problem_ds.site,
                "level": problem_ds.level,
            },
        )
    elif problem_ds.attrs["problem_type"] == ProblemTypes.SW_DIRECT.value:
        flux_dir = sw_solver_noscat(
            tau=problem_ds["tau"].transpose("site", "layer", "gpt"),
            mu0=problem_ds["solar_zenith_angle"].transpose("site", "layer"),
            inc_flux_dir=problem_ds["toa_source"].transpose("site", "gpt"),
        )

        fluxes = xr.Dataset(
            {
                "sw_flux_dir": (["site", "level"], flux_dir),
            },
            coords={
                "site": problem_ds.site,
                "level": problem_ds.level,
            },
        )

        # Post-process results for nighttime columns
        if "solar_angle_mask" in problem_ds:
            fluxes["sw_flux_dir"] = (
                fluxes["sw_flux_dir"] * problem_ds["solar_angle_mask"]
            )

    elif problem_ds.attrs["problem_type"] == ProblemTypes.SW_2STREAM.value:
        _, _, _, flux_up, flux_down, _ = sw_solver_2stream(
            tau=problem_ds["tau"].transpose("site", "layer", "gpt"),
            ssa=problem_ds["ssa"].transpose("site", "layer", "gpt"),
            g=problem_ds["g"].transpose("site", "layer", "gpt"),
            mu0=problem_ds["solar_zenith_angle"].transpose("site", "layer"),
            sfc_alb_dir=problem_ds["surface_albedo_direct"],
            sfc_alb_dif=problem_ds["surface_albedo_diffuse"],
            inc_flux_dir=problem_ds["toa_source"].transpose("site", "gpt"),
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
