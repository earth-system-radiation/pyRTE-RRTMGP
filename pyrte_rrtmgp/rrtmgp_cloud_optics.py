"""Cloud optics utilities for pyRTE-RRTMGP."""

import os

import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import CloudOpticsFiles
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
from pyrte_rrtmgp.rrtmgp_data import download_rrtmgp_data


def load_cloud_optics(
    file_path: str | None = None,
    cloud_optics_file: CloudOpticsFiles | None = None,
) -> xr.Dataset:
    """Load cloud optics data from a netCDF file.

    Args:
        cloud_optics_file: Enum specifying a predefined cloud optics file from the
            RRTMGP data package. Only used if file_path is None.

    Returns:
        xr.Dataset: Dataset containing the cloud optics data.

    Raises:
        ValueError: If neither file_path nor cloud_optics_file is provided.
    """
    if file_path is not None:
        dataset = xr.load_dataset(file_path)
    elif cloud_optics_file is not None:
        rte_rrtmgp_dir = download_rrtmgp_data()
        dataset = xr.load_dataset(os.path.join(rte_rrtmgp_dir, cloud_optics_file.value))
    else:
        raise ValueError("Either file_path or cloud_optics_file must be provided")

    return dataset


@xr.register_dataset_accessor("compute_cloud_optics")
class CloudOpticsAccessor:
    """Accessor for computing cloud optical properties.

    This accessor allows you to compute cloud optical properties using the
    `compute_cloud_optics` method.

    Example usage:
        `dataset.compute_cloud_optics(cloud_properties)`

    Args:
        cloud_properties (xr.Dataset): Dataset containing cloud properties.
        problem_type (str): Type of problem to solve, either "two-stream" (default)
            or "absorption".
        add_to_input (bool): Whether to add the computed properties to the input
            dataset (default: False).

    Returns:
        xr.Dataset: Dataset containing optical properties for both liquid and ice
            phases.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """Initialize the accessor."""
        self._obj = xarray_obj

    def __call__(
        self,
        cloud_properties: xr.Dataset,
        problem_type: str = "two-stream",
        add_to_input: bool = False,
    ) -> xr.Dataset:
        """
        Compute cloud optical properties for liquid and ice clouds.

        Args:
            cloud_properties: Dataset containing cloud properties.
            lw: Whether to compute liquid water phase (True) or ice water phase
                (False).
            add_to_input (bool): Whether to add the computed properties to the
                input dataset (default: False).

        Returns:
            xr.Dataset: Dataset containing optical properties for both liquid
                and ice phases.
        """
        cloud_optics = self._obj

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
        gpt_out_dim = "bnd" if gpt_dim == "nband" else "gpt"
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
                [],  # ngpt
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
                ["site", "layer", gpt_out_dim],  # ltau
                ["site", "layer", gpt_out_dim],  # ltaussa
                ["site", "layer", gpt_out_dim],  # ltaussag
            ],
            output_dtypes=[np.float64, np.float64, np.float64],
            vectorize=True,
            dask="parallelized",
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
                ["site", "layer", gpt_out_dim],  # itau
                ["site", "layer", gpt_out_dim],  # itaussa
                ["site", "layer", gpt_out_dim],  # itaussag
            ],
            output_dtypes=[np.float64, np.float64, np.float64],
            vectorize=True,
            dask="parallelized",
        )

        # Combine liquid and ice contributions
        if problem_type == "absorption":
            tau = (ltau - ltaussa) + (itau - itaussa)
            props = xr.Dataset({"tau": tau})
        else:
            tau = ltau + itau
            taussa = ltaussa + itaussa
            taussag = ltaussag + itaussag

            # Calculate derived quantities
            ssa = xr.where(tau > np.finfo(float).eps, taussa / tau, 0.0)
            g = xr.where(taussa > np.finfo(float).eps, taussag / taussa, 0.0)

            props = xr.Dataset({"tau": tau, "ssa": ssa, "g": g})

        if add_to_input:
            cloud_properties.update(props)

        return props


@xr.register_dataset_accessor("add_to")
class CombineOpticalPropsAccessor:
    """Accessor for combining optical properties.

    This accessor allows you to combine two sets of optical properties using
    the `add_to` method.

    Example usage:
        `dataset.add_to(other_dataset)`

    Args:
        other (xr.Dataset): Second set of optical properties to add.
        delta_scale (bool): Whether to apply delta scaling to the optical
            properties (default: False).

    Returns:
        xr.Dataset: Combined optical properties.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """Initialize the accessor."""
        self._obj = xarray_obj

    def __call__(self, other: xr.Dataset, delta_scale: bool = False) -> xr.Dataset:
        """
        Combine two sets of optical properties.

        Args:
            other: Second set of optical properties to add.

        Returns:
            xr.Dataset: Combined optical properties.
        """
        op1 = self._obj
        op2 = other

        if delta_scale:
            op1 = self.delta_scale_optical_props(op1)

        ncol = op2.sizes["site"]
        nlay = op2.sizes["layer"]
        ngpt = op2.sizes["gpt"]

        # Check if input has only tau (1-stream) or tau, ssa, g (2-stream)
        is_1stream_1 = "tau" in list(op1.data_vars) and "ssa" not in list(op1.data_vars)
        is_1stream_2 = "tau" in list(op2.data_vars) and "ssa" not in list(op2.data_vars)

        if "gpt" in op1["tau"].sizes:
            if is_1stream_1:
                if is_1stream_2:
                    # 1-stream by 1-stream
                    combined_tau = xr.apply_ufunc(
                        increment_1scalar_by_1scalar,
                        ncol,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op1["tau"],
                        input_core_dims=[
                            [],  # ncol
                            [],  # nlay
                            [],  # ngpt
                            ["site", "layer", "gpt"],  # tau_inout
                            ["site", "layer", "gpt"],  # tau_in
                        ],
                        output_core_dims=[["site", "layer", "gpt"]],
                        output_dtypes=[np.float64],
                        vectorize=True,
                        dask="parallelized",
                    )
                    op2["tau"] = combined_tau
                else:
                    # 1-stream by 2-stream
                    combined_tau = xr.apply_ufunc(
                        increment_1scalar_by_2stream,
                        ncol,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op1["tau"],
                        op1["ssa"],
                        input_core_dims=[
                            [],  # ncol
                            [],  # nlay
                            [],  # ngpt
                            ["site", "layer", "gpt"],  # tau_inout
                            ["site", "layer", "gpt"],  # tau_in
                            ["site", "layer", "gpt"],  # ssa_in
                        ],
                        output_core_dims=[["site", "layer", "gpt"]],
                        output_dtypes=[np.float64],
                        vectorize=True,
                        dask="parallelized",
                    )
                    op2["tau"] = combined_tau
            else:  # 2-stream output
                if is_1stream_2:
                    # 2-stream by 1-stream
                    combined_tau, combined_ssa = xr.apply_ufunc(
                        increment_2stream_by_1scalar,
                        ncol,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op2["ssa"],
                        op1["tau"],
                        input_core_dims=[
                            [],  # ncol
                            [],  # nlay
                            [],  # ngpt
                            ["site", "layer", "gpt"],  # tau_inout
                            ["site", "layer", "gpt"],  # ssa_inout
                            ["site", "layer", "gpt"],  # tau_in
                        ],
                        output_core_dims=[
                            ["site", "layer", "gpt"],
                            ["site", "layer", "gpt"],
                        ],
                        output_dtypes=[np.float64, np.float64],
                        vectorize=True,
                        dask="parallelized",
                    )
                    op2["tau"] = combined_tau
                    op2["ssa"] = combined_ssa
                else:
                    # 2-stream by 2-stream
                    combined_tau, combined_ssa, combined_g = xr.apply_ufunc(
                        increment_2stream_by_2stream,
                        ncol,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op2["ssa"],
                        op2["g"],
                        op1["tau"],
                        op1["ssa"],
                        op1["g"],
                        input_core_dims=[
                            [],  # ncol
                            [],  # nlay
                            [],  # ngpt
                            ["site", "layer", "gpt"],  # tau_inout
                            ["site", "layer", "gpt"],  # ssa_inout
                            ["site", "layer", "gpt"],  # g_inout
                            ["site", "layer", "gpt"],  # tau_in
                            ["site", "layer", "gpt"],  # ssa_in
                            ["site", "layer", "gpt"],  # g_in
                        ],
                        output_core_dims=[
                            ["site", "layer", "gpt"],
                            ["site", "layer", "gpt"],
                            ["site", "layer", "gpt"],
                        ],
                        output_dtypes=[np.float64, np.float64, np.float64],
                        vectorize=True,
                        dask="parallelized",
                    )
                    op2["tau"] = combined_tau
                    op2["ssa"] = combined_ssa
                    op2["g"] = combined_g
        else:
            # By-band increment (when op2's ngpt equals op1's nband)
            if op2.sizes["bnd"] != op1.sizes["bnd"]:
                raise ValueError(
                    "Incompatible g-point structures for by-band increment"
                )

            if is_1stream_1:
                if is_1stream_2:
                    # 1-stream by 1-stream by band
                    combined_tau = xr.apply_ufunc(
                        inc_1scalar_by_1scalar_bybnd,
                        ncol,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op1["tau"],
                        op2.sizes["bnd"],
                        op2["bnd_limits_gpt"],
                        input_core_dims=[
                            [],  # ncol
                            [],  # nlay
                            [],  # ngpt
                            ["site", "layer", "gpt"],  # tau_inout
                            ["site", "layer", "bnd"],  # tau_in
                            [],  # nbnd
                            ["pair", "bnd"],  # band_lims_gpoint
                        ],
                        output_core_dims=[["site", "layer", "gpt"]],
                        output_dtypes=[np.float64],
                        vectorize=True,
                        dask="parallelized",
                    )
                    op2["tau"] = combined_tau
                else:
                    # 1-stream by 2-stream by band
                    combined_tau = xr.apply_ufunc(
                        inc_1scalar_by_2stream_bybnd,
                        ncol,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op1["tau"],
                        op1["ssa"],
                        op2.sizes["bnd"],
                        op2["bnd_limits_gpt"],
                        input_core_dims=[
                            [],  # ncol
                            [],  # nlay
                            [],  # ngpt
                            ["site", "layer", "gpt"],  # tau_inout
                            ["site", "layer", "bnd"],  # tau_in
                            ["site", "layer", "bnd"],  # ssa_in
                            [],  # nbnd
                            ["pair", "bnd"],  # bnd_limits_gpt
                        ],
                        output_core_dims=[["site", "layer", "gpt"]],
                        output_dtypes=[np.float64],
                        vectorize=True,
                        dask="parallelized",
                    )
                    op2["tau"] = combined_tau
            else:
                if is_1stream_2:
                    # 2-stream by 1-stream by band
                    combined_tau = xr.apply_ufunc(
                        inc_2stream_by_1scalar_bybnd,
                        ncol,
                        nlay,
                        ngpt,
                        op2["tau"],
                        op2["ssa"],
                        op1["tau"],
                        op2.sizes["bnd"],
                        op2["bnd_limits_gpt"],
                        input_core_dims=[
                            [],  # ncol
                            [],  # nlay
                            [],  # ngpt
                            ["site", "layer", "gpt"],  # tau_inout
                            ["site", "layer", "gpt"],  # ssa_inout
                            ["site", "layer", "bnd"],  # tau_in
                            [],  # nbnd
                            ["pair", "bnd"],  # band_lims_gpoint
                        ],
                        output_core_dims=[["site", "layer", "gpt"]],
                        output_dtypes=[np.float64],
                        vectorize=True,
                        dask="parallelized",
                    )
                    op2["tau"] = combined_tau
                else:
                    # 2-stream by 2-stream by band
                    combined_tau = xr.apply_ufunc(
                        inc_2stream_by_2stream_bybnd,
                        ncol,
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
                            [],  # ncol
                            [],  # nlay
                            [],  # ngpt
                            ["site", "layer", "gpt"],  # tau_inout
                            ["site", "layer", "gpt"],  # ssa_inout
                            ["site", "layer", "gpt"],  # g_inout
                            ["site", "layer", "bnd"],  # tau_in
                            ["site", "layer", "bnd"],  # ssa_in
                            ["site", "layer", "bnd"],  # g_in
                            [],  # nbnd
                            ["pair", "bnd"],  # band_lims_gpoint
                        ],
                        output_core_dims=[["site", "layer", "gpt"]],
                        output_dtypes=[np.float64],
                        vectorize=True,
                        dask="parallelized",
                    )
                    op2["tau"] = combined_tau

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
        ncol = optical_props.sizes["site"]
        nlay = optical_props.sizes["layer"]
        gpt_dim = "gpt" if "gpt" in optical_props.sizes else "bnd"
        ngpt = optical_props.sizes[gpt_dim]

        # Call kernel with forward scattering
        if forward_scattering is not None:
            tau, ssa, g = xr.apply_ufunc(
                delta_scale_2str_f,
                ncol,
                nlay,
                ngpt,
                optical_props["tau"],
                optical_props["ssa"],
                optical_props["g"],
                forward_scattering,
                input_core_dims=[
                    [],  # ncol
                    [],  # nlay
                    [],  # ngpt
                    ["site", "layer", gpt_dim],  # tau
                    ["site", "layer", gpt_dim],  # ssa
                    ["site", "layer", gpt_dim],  # g
                    ["site", "layer", gpt_dim],  # f
                ],
                output_core_dims=[
                    ["site", "layer", gpt_dim],
                    ["site", "layer", gpt_dim],
                    ["site", "layer", gpt_dim],
                ],
                output_dtypes=[np.float64, np.float64, np.float64],
                vectorize=True,
                dask="parallelized",
            )
        else:
            tau, ssa, g = xr.apply_ufunc(
                delta_scale_2str,
                ncol,
                nlay,
                ngpt,
                optical_props["tau"],
                optical_props["ssa"],
                optical_props["g"],
                input_core_dims=[
                    [],  # ncol
                    [],  # nlay
                    [],  # ngpt
                    ["site", "layer", gpt_dim],  # tau
                    ["site", "layer", gpt_dim],  # ssa
                    ["site", "layer", gpt_dim],  # g
                ],
                output_core_dims=[
                    ["site", "layer", gpt_dim],
                    ["site", "layer", gpt_dim],
                    ["site", "layer", gpt_dim],
                ],
                output_dtypes=[np.float64, np.float64, np.float64],
                vectorize=True,
                dask="parallelized",
            )

        # Update the dataset with modified values
        optical_props["tau"] = tau
        optical_props["ssa"] = ssa
        optical_props["g"] = g

        return optical_props
