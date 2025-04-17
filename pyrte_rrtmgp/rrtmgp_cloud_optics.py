"""Cloud optics utilities for pyRTE-RRTMGP."""

import os

import numpy as np
import xarray as xr

from pyrte_rrtmgp.data_types import CloudOpticsFiles
from pyrte_rrtmgp.data_validation import (
    AtmosphericMapping,
    create_default_mapping,
)
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
from pyrte_rrtmgp.utils import safer_divide


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
        variable_mapping: AtmosphericMapping | None = None,
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

        if variable_mapping is None:
            variable_mapping = create_default_mapping()
        # Set mapping in accessor
        cloud_properties.mapping.set_mapping(variable_mapping)

        layer_dim = cloud_properties.mapping.get_dim("layer")

        # Get dimensions
        nlay = cloud_properties.sizes[layer_dim]

        non_default_dims = [
            d for d in cloud_properties.dims if d not in ["level", "layer", "gpt"]
        ]
        cloud_properties = cloud_properties.stack({"stacked_cols": non_default_dims})

        # Determine if we're using band-averaged or spectral properties
        gpt_dim = "nband" if "gpt" not in cloud_optics.sizes else "gpt"
        gpt_out_dim = "bnd" if gpt_dim == "nband" else "gpt"
        ngpt = cloud_optics.sizes["nband" if gpt_dim == "nband" else "gpt"]

        # Sequentially process each chunk
        # Create cloud masks
        liq_mask = cloud_properties.lwp > 0
        ice_mask = cloud_properties.iwp > 0

        # Compute optical properties using lookup tables
        # Liquid phase
        step_size = (cloud_optics.radliq_upr - cloud_optics.radliq_lwr) / (
            cloud_optics.sizes["nsize_liq"] - 1
        )

        ltau, ltaussa, ltaussag = xr.apply_ufunc(
            compute_cld_from_table,
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
                [],  # nlay
                [],  # ngpt
                [layer_dim],  # liq_mask
                [layer_dim],  # lwp
                [layer_dim],  # rel
                [],  # nsize_liq
                [],  # step_size
                [],  # radliq_lwr
                ["nsize_liq", gpt_dim],  # extliq
                ["nsize_liq", gpt_dim],  # ssaliq
                ["nsize_liq", gpt_dim],  # asyliq
            ],
            output_core_dims=[
                [layer_dim, gpt_out_dim],  # ltau
                [layer_dim, gpt_out_dim],  # ltaussa
                [layer_dim, gpt_out_dim],  # ltaussag
            ],
            output_dtypes=[np.float64, np.float64, np.float64],
            dask_gufunc_kwargs={
                "output_sizes": {
                    gpt_out_dim: ngpt,
                },
            },
            dask="parallelized",
        )

        # Ice phase
        step_size = (cloud_optics.diamice_upr - cloud_optics.diamice_lwr) / (
            cloud_optics.sizes["nsize_ice"] - 1
        )
        ice_roughness = 1

        itau, itaussa, itaussag = xr.apply_ufunc(
            compute_cld_from_table,
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
                [],  # nlay
                [],  # ngpt
                [layer_dim],  # ice_mask
                [layer_dim],  # iwp
                [layer_dim],  # rei
                [],  # nsize_ice
                [],  # step_size
                [],  # diamice_lwr
                ["nsize_ice", gpt_dim],  # extice
                ["nsize_ice", gpt_dim],  # ssaice
                ["nsize_ice", gpt_dim],  # asyice
            ],
            output_core_dims=[
                [layer_dim, gpt_out_dim],  # itau
                [layer_dim, gpt_out_dim],  # itaussa
                [layer_dim, gpt_out_dim],  # itaussag
            ],
            output_dtypes=[np.float64, np.float64, np.float64],
            dask_gufunc_kwargs={
                "output_sizes": {
                    gpt_out_dim: ngpt,
                },
            },
            dask="parallelized",
        )

        ltau = ltau.unstack("stacked_cols")
        ltaussa = ltaussa.unstack("stacked_cols")
        ltaussag = ltaussag.unstack("stacked_cols")
        itau = itau.unstack("stacked_cols")
        itaussa = itaussa.unstack("stacked_cols")
        itaussag = itaussag.unstack("stacked_cols")

        # Combine liquid and ice contributions
        if problem_type == "absorption":
            tau = (ltau - ltaussa) + (itau - itaussa)
            props = xr.Dataset({"tau": tau})
        else:
            tau = ltau + itau
            taussa = ltaussa + itaussa
            taussag = ltaussag + itaussag

            # Apply the function with dask awareness.
            ssa = xr.apply_ufunc(
                safer_divide,
                taussa,
                tau,
                dask="allowed",  # Allows lazy evaluation with dask arrays.
                output_dtypes=[
                    taussa.dtype
                ],  # Ensure the output data type is correctly specified.
            )

            g = xr.apply_ufunc(
                safer_divide,
                taussag,
                taussa,
                dask="allowed",  # Allows lazy evaluation with dask arrays.
                output_dtypes=[
                    taussa.dtype
                ],  # Ensure the output data type is correctly specified.
            )

            props = xr.Dataset({"tau": tau, "ssa": ssa, "g": g})

        if add_to_input:
            cloud_properties.update(props)
            return

        props.mapping.set_mapping(cloud_properties.mapping.mapping)
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
