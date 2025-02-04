import hashlib
import os
import platform
import tarfile
from pathlib import Path
from typing import Union

import requests

# URL of the file to download
TAG = "v1.9"
DATA_URL = f"https://github.com/earth-system-radiation/rrtmgp-data/archive/refs/tags/{TAG}.tar.gz"


def get_cache_dir() -> str:
    """Get the system-specific cache directory for pyrte_rrtmgp data.

    Returns:
        str: Path to the cache directory
    """
    # Determine the system cache folder
    if platform.system() == "Windows":
        cache_path = os.getenv("LOCALAPPDATA")
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
    file_path = os.path.join(cache_dir, f"{TAG}.tar.gz")

    # Path to the file containing the checksum of the downloaded file
    checksum_file_path = os.path.join(cache_dir, f"{TAG}.tar.gz.sha256")

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

    return os.path.join(cache_dir, f"rrtmgp-data-{TAG[1:]}")


def _get_file_checksum(filepath: Union[str, Path], mode: str = "r") -> str:
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
