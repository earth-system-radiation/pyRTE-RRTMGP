import hashlib
import os
import platform
import tarfile

import requests

# URL of the file to download
TAG = "v1.8"
DATA_URL = f"https://github.com/earth-system-radiation/rrtmgp-data/archive/refs/tags/{TAG}.tar.gz"


def get_cache_dir():
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


def download_rrtmgp_data():
    # Directory where the data will be stored
    cache_dir = get_cache_dir()

    # Path to the downloaded file
    file_path = os.path.join(cache_dir, f"{TAG}.tar.gz")

    # Path to the file containing the checksum of the downloaded file
    checksum_file_path = os.path.join(cache_dir, f"{TAG}.tar.gz.sha256")

    # Download the file if it doesn't exist or if the checksum doesn't match
    if not os.path.exists(file_path) or (
        os.path.exists(checksum_file_path)
        and open(checksum_file_path).read()
        != hashlib.sha256(open(file_path, "rb").read()).hexdigest()
    ):
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Save the checksum of the downloaded file
        with open(checksum_file_path, "w") as f:
            f.write(hashlib.sha256(open(file_path, "rb").read()).hexdigest())

    # Uncompress the file
    with tarfile.open(file_path) as tar:
        tar.extractall(path=cache_dir)

    return os.path.join(cache_dir, f"rrtmgp-data-{TAG[1:]}")
