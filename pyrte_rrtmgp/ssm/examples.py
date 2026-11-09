"""Example data and functions.

Files and functions for Python versions of the clear-sky (RFMIP) and all-sky examples
from the Fortran RTE+RRTMGP package
"""

import os
from enum import StrEnum

import pooch
import xarray as xr


class RTE_EXAMPLES(StrEnum):
    """rte-examples files.

    Each contains state and boundary condition information usable directly by
    pyRTE and pyRRTMGP
    """

    REF_RCE_LW = "examples/rte-examples/reference/ssm-lw-rce.nc"
    REF_RCE_SW = "examples/rte-examples/reference/ssm-sw-rce.nc"
    REF_RFMIP_LW = "examples/rte-examples/reference/ssm-lw-rfmip.nc"
    REF_RFMIP_SW = "examples/rte-examples/reference/ssm-sw-rfmip.nc"
    REF_CKDMIP_LW = "examples/rte-examples/reference/ssm-lw-ckdmip.nc"
    REF_CKDMIP_SW = "examples/rte-examples/reference/ssm-sw-ckdmip.nc"


def download_ssm_data() -> os.PathLike:
    """Download and extract SSM files (example results).

    Downloads the SSM data files from GitHub if not already present in the cache

    Returns:
        str: Path to the extracted data directory
    """
    # URL of the file to download
    REF: str = "v0.0.1"  # Can be a tag (e.g. "v1.8.2") or branch name (e.g. "main")
    DATA_URL: str = (
        "https://github.com/earth-system-radiation/ssm-data/archive/refs/"
        f"{'tags' if REF.startswith('v') else 'heads'}/"
    )
    DATA_HASH = "04c5015d8e7d71f3058f6f59394be071e0cc466f22d9b65ecf59c1035dd67f5a"

    data_dir = pooch.os_cache("ssm-data")

    downloader = pooch.create(
        path=data_dir,
        base_url=DATA_URL,
        registry={
            f"{REF}.zip": DATA_HASH,
        },
    )

    _ = downloader.fetch(f"{REF}.zip", processor=pooch.Unzip(extract_dir=f"{REF}"))

    ref_dirname = REF[1:] if REF.startswith("v") else REF
    return data_dir / f"{REF}" / f"ssm-data-{ref_dirname}"


def load_example_file(file: RTE_EXAMPLES) -> xr.Dataset:
    """Load an example file.

    This would work for any file in the download directory but it's
    currently used only for the examples.

    Args:
        file: The file to load

    Returns:
        xr.Dataset: The loaded dataset
    """
    ssm_dir = download_ssm_data()
    ref_path = ssm_dir / file.value  # type: ignore
    return xr.load_dataset(ref_path, decode_cf=False)
