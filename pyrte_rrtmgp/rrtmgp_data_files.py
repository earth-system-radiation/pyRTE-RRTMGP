"""Data download utilities and file name enums (lists) for pyRTE-RRTMGP."""

import hashlib
import os
import platform
import tarfile
from enum import StrEnum
from pathlib import Path
from typing import Literal, Union

import requests


class GasOpticsFiles(StrEnum):
    """Enumeration of default RRTMGP gas optics data files.

    This enum defines the available pre-configured gas optics data files that can be
    used with RRTMGP. The files contain absorption coefficients and other optical
    properties needed for radiative transfer calculations.
    """

    LW_G128 = "rrtmgp-gas-lw-g128.nc"
    """Longwave gas optics file with 128 g-points"""

    LW_G256 = "rrtmgp-gas-lw-g256.nc"
    """Longwave gas optics file with 256 g-points"""

    SW_G112 = "rrtmgp-gas-sw-g112.nc"
    """Shortwave gas optics file with 112 g-points"""

    SW_G224 = "rrtmgp-gas-sw-g224.nc"
    """Shortwave gas optics file with 224 g-points"""


class CloudOpticsFiles(StrEnum):
    """Enumeration of default RRTMGP cloud optics data files.

    This enum defines the available pre-configured cloud optics data files that can be
    used with RRTMGP. The files contain cloud optical properties needed for radiative
    transfer calculations.
    """

    LW_BND = "rrtmgp-clouds-lw-bnd.nc"
    """Longwave cloud optics file with band points"""

    LW_G128 = "rrtmgp-clouds-lw-g128.nc"
    """Longwave cloud optics file with 128 g-points"""

    LW_G256 = "rrtmgp-clouds-lw-g256.nc"
    """Longwave cloud optics file with 256 g-points"""

    SW_BND = "rrtmgp-clouds-sw-bnd.nc"
    """Shortwave cloud optics file with band points"""

    SW_G112 = "rrtmgp-clouds-sw-g112.nc"
    """Shortwave cloud optics file with 112 g-points"""

    SW_G224 = "rrtmgp-clouds-sw-g224.nc"
    """Shortwave cloud optics file with 224 g-points"""


class AerosolOpticsFiles(StrEnum):
    """Enumeration of default RRTMGP aerosol optics data files.

    This enum defines the available pre-configured aerosol optics data files that can be
    used with RRTMGP. The files contain aerosol optical properties needed for radiative
    transfer calculations.
    """

    LW_MERRA = "rrtmgp-aerosols-merra-lw.nc"
    SW_MERRA = "rrtmgp-aerosols-merra-sw.nc"


# URL of the file to download
REF: str = "v1.9"  # Can be a tag (e.g. "v1.8.2") or branch name (e.g. "main")
DATA_URL: str = (
    "https://github.com/earth-system-radiation/rrtmgp-data/archive/refs/"
    f"{'tags' if REF.startswith('v') else 'heads'}/{REF}.tar.gz"
)


def get_cache_dir() -> str:
    """Get the system-specific cache directory for pyrte_rrtmgp data.

    Returns:
        str: Path to the cache directory
    """
    # Determine the system cache folder
    if platform.system() == "Windows":
        cache_path = os.getenv("LOCALAPPDATA", "")
    elif platform.system() == "Darwin":
        cache_path = os.path.expanduser("~/Library/Caches")
    else:
        cache_path = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_path = os.path.join(cache_path, "pyrte_rrtmgp")

    # Create the directory if it doesn't exist
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    return cache_path


def download_rrtmgp_data() -> str:
    """Download and extract RRTMGP data files.

    Downloads the RRTMGP data files from GitHub if not already present in the cache,
    verifies the checksum, and extracts the contents.

    Returns:
        str: Path to the extracted data directory

    Raises:
        requests.exceptions.RequestException: If download fails
        tarfile.TarError: If extraction fails
    """
    # Directory where the data will be stored
    cache_dir = get_cache_dir()

    # Path to the downloaded file
    file_path = os.path.join(cache_dir, f"{REF}.tar.gz")

    # Path to the file containing the checksum of the downloaded file
    checksum_file_path = os.path.join(cache_dir, f"{REF}.tar.gz.sha256")

    # Download the file if it doesn't exist or if the checksum doesn't match
    if not os.path.exists(file_path) or (
        os.path.exists(checksum_file_path)
        and _get_file_checksum(checksum_file_path)
        != _get_file_checksum(file_path, mode="rb")
    ):
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Save the checksum of the downloaded file
        with open(checksum_file_path, "w") as f:
            f.write(_get_file_checksum(file_path, mode="rb"))

    # Uncompress the file
    with tarfile.open(file_path) as tar:
        tar.extractall(path=cache_dir, filter="data")

    # Handle both tag and branch names in the extracted directory name
    # For tags like "v1.8.2", remove the "v" prefix
    # For branches like "main", use as-is
    ref_dirname = REF[1:] if REF.startswith("v") else REF
    return os.path.join(cache_dir, f"rrtmgp-data-{ref_dirname}")


def _get_file_checksum(
    filepath: Union[str, Path], mode: Literal["r", "rb"] = "r"
) -> str:
    """Calculate SHA256 checksum of a file or read existing checksum.

    Args:
        filepath: Path to the file
        mode: File open mode, "r" for text or "rb" for binary

    Returns:
        str: File content if mode="r", or SHA256 hex digest if mode="rb"
    """
    with open(filepath, mode) as f:
        content = f.read()
        return hashlib.sha256(content).hexdigest() if mode == "rb" else content
