import numpy as np
import xarray as xr

from pyrte_rrtmgp.kernels.rrtmgp import compute_cld_from_table
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
)


def compute_clouds(cloud_optics, ncol, nlay, p_lay, t_lay):
    """Compute cloud properties for radiative transfer calculations.

    Args:
        cloud_optics: Dataset containing cloud optics data
        ncol: Number of columns
        nlay: Number of layers
        p_lay: Pressure profile
        t_lay: Temperature profile
    """
    # Get min/max radii values for liquid and ice
    rel_val = 0.5 * (cloud_optics["radliq_lwr"] + cloud_optics["radliq_upr"])
    rei_val = 0.5 * (cloud_optics["diamice_lwr"] + cloud_optics["diamice_upr"])

    # Create coordinate arrays
    site = np.arange(ncol)
    layer = np.arange(nlay)

    # Convert inputs to xarray DataArrays if they aren't already
    p_lay = xr.DataArray(
        p_lay, dims=["site", "layer"], coords={"site": site, "layer": layer}
    )
    t_lay = xr.DataArray(
        t_lay, dims=["site", "layer"], coords={"site": site, "layer": layer}
    )

    # Create cloud mask using xarray operations
    cloud_mask = (
        (p_lay > 100 * 100) & (p_lay < 900 * 100) & ((site + 1) % 3 != 0).reshape(-1, 1)
    )

    # Initialize arrays as DataArrays with zeros
    lwp = xr.zeros_like(p_lay)
    iwp = xr.zeros_like(p_lay)
    rel = xr.zeros_like(p_lay)
    rei = xr.zeros_like(p_lay)

    # Set values where clouds exist using xarray where operations
    lwp = lwp.where(~(cloud_mask & (t_lay > 263)), 10.0)
    rel = rel.where(~(cloud_mask & (t_lay > 263)), rel_val)

    iwp = iwp.where(~(cloud_mask & (t_lay < 273)), 10.0)
    rei = rei.where(~(cloud_mask & (t_lay < 273)), rei_val)

    return xr.Dataset(
        {
            "lwp": lwp,
            "iwp": iwp,
            "rel": rel,
            "rei": rei,
        }
    )


def compute_cloud_optics(cloud_properties, cloud_optics, lw=True):
    """
    Compute cloud optical properties for liquid and ice clouds.

    Args:
        cloud_properties: Dataset containing cloud properties
        cloud_optics: Dataset containing cloud optics data
        lw (bool): Whether to compute liquid water phase (True) or ice water phase (False)

    Returns:
        tuple: Arrays of optical properties for both liquid and ice phases
    """
    # Get dimensions
    ncol, nlay = cloud_properties.sizes["site"], cloud_properties.sizes["layer"]

    # Create cloud masks
    liq_mask = cloud_properties.lwp > 0
    ice_mask = cloud_properties.iwp > 0

    # Check if cloud optics data is initialized
    if not hasattr(cloud_optics, "extliq"):
        raise ValueError("Cloud optics: no data has been initialized")

    # Validate particle sizes are within bounds
    if np.any(
        (cloud_properties.rel.where(liq_mask) < cloud_optics.radliq_lwr.values)
        | (cloud_properties.rel.where(liq_mask) > cloud_optics.radliq_upr.values)
    ):
        raise ValueError("Cloud optics: liquid effective radius is out of bounds")

    if np.any(
        (cloud_properties.rei.where(ice_mask) < cloud_optics.diamice_lwr.values)
        | (cloud_properties.rei.where(ice_mask) > cloud_optics.diamice_upr.values)
    ):
        raise ValueError("Cloud optics: ice effective radius is out of bounds")

    # Check for negative water paths
    if np.any(cloud_properties.lwp.where(liq_mask) < 0) or np.any(
        cloud_properties.iwp.where(ice_mask) < 0
    ):
        raise ValueError(
            "Cloud optics: negative lwp or iwp where clouds are supposed to be"
        )

    # Determine if we're using band-averaged or spectral properties
    gpt_dim = "nband" if "gpt" not in cloud_optics.sizes else "gpt"
    ngpt = cloud_optics.sizes["nband" if gpt_dim == "nband" else "gpt"]

    # Compute optical properties using lookup tables
    # Liquid phase
    step_size = (cloud_optics.radliq_upr - cloud_optics.radliq_lwr) / (
        cloud_optics.sizes["nsize_liq"] - 1
    )

    ltau, ltaussa, ltaussag = xr.apply_ufunc(
        compute_cld_from_table,
        ncol,
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
            [],  # ncol
            [],  # nlay
            [],
            ["site", "layer"],  # liq_mask
            ["site", "layer"],  # lwp
            ["site", "layer"],  # rel
            [],  # nsize_liq
            [],  # step_size
            [],  # radliq_lwr
            ["nsize_liq", gpt_dim],  # extliq
            ["nsize_liq", gpt_dim],  # ssaliq
            ["nsize_liq", gpt_dim],  # asyliq
        ],
        output_core_dims=[
            ["site", "layer", "gpt"],  # ltau
            ["site", "layer", "gpt"],  # ltaussa
            ["site", "layer", "gpt"],  # ltaussag
        ],
        vectorize=True,
        dask="allowed",
    )

    # Ice phase
    step_size = (cloud_optics.diamice_upr - cloud_optics.diamice_lwr) / (
        cloud_optics.sizes["nsize_ice"] - 1
    )
    ice_roughness = 1

    itau, itaussa, itaussag = xr.apply_ufunc(
        compute_cld_from_table,
        ncol,
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
            [],  # ncol
            [],  # nlay
            [],  # ngpt
            ["site", "layer"],  # ice_mask
            ["site", "layer"],  # iwp
            ["site", "layer"],  # rei
            [],  # nsize_ice
            [],  # step_size
            [],  # diamice_lwr
            ["nsize_ice", gpt_dim],  # extice
            ["nsize_ice", gpt_dim],  # ssaice
            ["nsize_ice", gpt_dim],  # asyice
        ],
        output_core_dims=[
            ["site", "layer", "gpt"],  # itau
            ["site", "layer", "gpt"],  # itaussa
            ["site", "layer", "gpt"],  # itaussag
        ],
        vectorize=True,
        dask="allowed",
    )

    # Combine liquid and ice contributions
    if lw:
        tau = (ltau - ltaussa) + (itau - itaussa)
        return tau.to_dataset(name="tau")
    else:
        tau = ltau + itau
        taussa = ltaussa + itaussa
        taussag = ltaussag + itaussag

        # Calculate derived quantities
        ssa = xr.where(tau > np.finfo(float).eps, taussa / tau, 0.0)
        g = xr.where(taussa > np.finfo(float).eps, taussag / taussa, 0.0)

        return xr.Dataset({"tau": tau, "ssa": ssa, "g": g})


def combine_optical_props(op1, op2):
    """Combine two sets of optical properties, modifying op1 in place.

    Args:
        op1: First set of optical properties, will be modified.
        op2: Second set of optical properties to add.
    """
    ncol = op2.sizes["site"]
    nlay = op2.sizes["layer"]
    ngpt = op2.sizes["gpt"]

    # Check if input has only tau (1-stream) or tau, ssa, g (2-stream)

    is_1stream_1 = "tau" in list(op1.data_vars) and "ssa" not in list(op1.data_vars)
    is_1stream_2 = "tau" in list(op2.data_vars) and "ssa" not in list(op2.data_vars)

    # Check if the g-points are equal between the two datasets
    gpoints_equal = op1.sizes["gpt"] == op2.sizes["gpt"]

    if gpoints_equal:
        if is_1stream_1:
            if is_1stream_2:
                # 1-stream by 1-stream
                increment_1scalar_by_1scalar(
                    ncol, nlay, ngpt, op2["tau"].values, op1["tau"].values
                )
                op2["tau"] = (("site", "layer", "gpt"), op2["tau"].values)
            else:
                # 1-stream by 2-stream
                increment_1scalar_by_2stream(
                    ncol,
                    nlay,
                    ngpt,
                    op2["tau"].values,
                    op1["tau"].values,
                    op1["ssa"].values,
                )
                op2["tau"] = (("site", "layer", "gpt"), op2["tau"].values)
        else:  # 2-stream output
            if is_1stream_2:
                # 2-stream by 1-stream
                increment_2stream_by_1scalar(
                    ncol,
                    nlay,
                    ngpt,
                    op2["tau"].values,
                    op2["ssa"].values,
                    op1["tau"].values,
                )
                op2["tau"] = (("site", "layer", "gpt"), op2["tau"].values)
                op2["ssa"] = (("site", "layer", "gpt"), op2["ssa"].values)
            else:
                # 2-stream by 2-stream
                increment_2stream_by_2stream(
                    ncol,
                    nlay,
                    ngpt,
                    op2["tau"].values,
                    op2["ssa"].values,
                    op2["g"].values,
                    op1["tau"].values,
                    op1["ssa"].values,
                    op1["g"].values,
                )
                op2["tau"] = (("site", "layer", "gpt"), op2["tau"].values)
                op2["ssa"] = (("site", "layer", "gpt"), op2["ssa"].values)
                op2["g"] = (("site", "layer", "gpt"), op2["g"].values)

    else:
        # By-band increment (when op2's ngpt equals op1's nband)
        if op2.sizes["bnd"] != op1.sizes["gpt"]:
            raise ValueError("Incompatible g-point structures for by-band increment")

        if is_1stream_1:
            if is_1stream_2:
                # 1-stream by 1-stream by band
                inc_1scalar_by_1scalar_bybnd(
                    ncol,
                    nlay,
                    ngpt,
                    op2["tau"].values,
                    op1["tau"].values,
                    op2.sizes["bnd"],
                    op2["bnd_limits_gpt"].values.T,
                )
                op2["tau"] = (("site", "layer", "gpt"), op2["tau"].values)
            else:
                # 1-stream by 2-stream by band
                inc_1scalar_by_2stream_bybnd(
                    ncol,
                    nlay,
                    ngpt,
                    op2["tau"].values,
                    op1["tau"].values,
                    op1["ssa"].values,
                    op2.sizes["bnd"],
                    op2["bnd_limits_gpt"].values.T,
                )
                op2["tau"] = (("site", "layer", "gpt"), op2["tau"].values)
        else:
            if is_1stream_2:
                # 2-stream by 1-stream by band
                inc_2stream_by_1scalar_bybnd(
                    ncol,
                    nlay,
                    ngpt,
                    op2["tau"].values,
                    op2["ssa"].values,
                    op1["tau"].values,
                    op2.sizes["bnd"],
                    op2["bnd_limits_gpt"].values.T,
                )
                op2["tau"] = (("site", "layer", "gpt"), op2["tau"].values)
                op2["ssa"] = (("site", "layer", "gpt"), op2["ssa"].values)
            else:
                # 2-stream by 2-stream by band
                inc_2stream_by_2stream_bybnd(
                    ncol,
                    nlay,
                    ngpt,
                    op2["tau"].values,
                    op2["ssa"].values,
                    op2["g"].values,
                    op1["tau"].values,
                    op1["ssa"].values,
                    op1["g"].values,
                    op2.sizes["bnd"],
                    op2["bnd_limits_gpt"].values.T,
                )
                op2["tau"] = (("site", "layer", "gpt"), op2["tau"].values)
                op2["ssa"] = (("site", "layer", "gpt"), op2["ssa"].values)
                op2["g"] = (("site", "layer", "gpt"), op2["g"].values)


def delta_scale_optical_props(optical_props, forward_scattering=None):
    """Apply delta scaling to 2-stream optical properties.

    Args:
        optical_props: xarray Dataset containing tau, ssa, and g variables
        forward_scattering: Optional array of forward scattering fraction (g**2 if not provided)
            Must have shape (ncol, nlay, ngpt) if provided

    Raises:
        ValueError: If forward_scattering array has incorrect dimensions or values outside [0,1]
    """
    # Get dimensions
    ncol = optical_props.sizes["site"]
    nlay = optical_props.sizes["layer"]
    ngpt = optical_props.sizes["gpt"]

    # Get arrays and ensure they're mutable
    tau = optical_props.tau.values
    ssa = optical_props.ssa.values
    g = optical_props.g.values

    if forward_scattering is not None:
        # Validate dimensions
        if forward_scattering.shape != (ncol, nlay, ngpt):
            raise ValueError(
                "delta_scale: dimension of forward_scattering doesn't match optical properties arrays"
            )

        # Validate values are in [0,1]
        if np.any((forward_scattering < 0) | (forward_scattering > 1)):
            raise ValueError(
                "delta_scale: values of forward_scattering out of bounds [0,1]"
            )

        # Call kernel with forward scattering
        delta_scale_2str_f(ncol, nlay, ngpt, tau, ssa, g, forward_scattering)
    else:
        # Call kernel without forward scattering
        delta_scale_2str(ncol, nlay, ngpt, tau, ssa, g)

    # Update the dataset with modified values
    optical_props["tau"] = (("site", "layer", "gpt"), tau)
    optical_props["ssa"] = (("site", "layer", "gpt"), ssa)
    optical_props["g"] = (("site", "layer", "gpt"), g)
