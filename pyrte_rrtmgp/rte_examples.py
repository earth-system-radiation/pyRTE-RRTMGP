"""Data download utilities and file name enums (lists) for pyRTE-RRTMGP."""

import os
from enum import StrEnum

import pooch
import xarray as xr


class RTEExamplesFiles(StrEnum):
    """Enumeration of sets of atmospheric profiles available in rte-examples."""

    RCE_STATES = "rce-states.nc"
    """Profiles in RCE with varying surface temperatures; variants for CO2 conc."""

    RFMIP_STATES = "rfmip-states.nc"
    """Profiles from the Radiative Forcing MIP """

    CKDMIP_STATES = "ckdmip-states.nc"
    """Profiles chosen from the Correlated k-distribution MIP """


def download_rte_examples() -> os.PathLike:
    """Download and extract rte-examples states.

    Downloads the rte-examples state files from GitHub
    if not already present in the cache

    Returns:
        os.PathLike: Path to the extracted data directory
    """
    # URL of the file to download
    REF: str = "v0.1.1"  # Can be a tag (e.g. "v1.8.2") or branch name (e.g. "main")
    DATA_URL: str = (
        "https://github.com/earth-system-radiation/rte-examples/archive/refs/"
        f"{'tags' if REF.startswith('v') else 'heads'}/"
    )
    DATA_HASH = "e9435961aaa825d95f0aa1a66382de874bd2609e6bffe99a8a5124b89feb5c0c"

    data_dir = pooch.os_cache("rte-examples")

    downloader = pooch.create(
        path=data_dir,
        base_url=DATA_URL,
        registry={
            f"{REF}.zip": DATA_HASH,
        },
    )

    _ = downloader.fetch(f"{REF}.zip", processor=pooch.Unzip(extract_dir=f"{REF}"))

    ref_dirname = REF[1:] if REF.startswith("v") else REF
    return data_dir / f"{REF}" / f"rte-examples-{ref_dirname}"


def load_rte_example_file(file: RTEExamplesFiles) -> xr.Dataset:
    """Load one of the rte-examples files.

    Args:
        file: The file to load

    Returns:
        xr.Dataset: The loaded dataset
    """
    examples_dir = download_rte_examples()
    ref_path = examples_dir / file.value  # type: ignore
    return xr.load_dataset(ref_path, decode_cf=False)
