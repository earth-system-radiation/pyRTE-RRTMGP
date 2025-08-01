"""Cloud optics utilities for pyRTE-RRTMGP."""

import os

import numpy as np
import xarray as xr

from pyrte_rrtmgp.input_mapping import (
    AtmosphericMapping,
    create_default_mapping,
)
from pyrte_rrtmgp.kernels.rrtmgp import compute_cld_from_table
from pyrte_rrtmgp.rrtmgp_data_files import CloudOpticsFiles, download_rrtmgp_data
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
