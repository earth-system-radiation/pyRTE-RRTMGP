"""Data download utilities and file name enums (lists) for pyRTE-RRTMGP."""

import os
from enum import StrEnum

import pooch


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


def download_rrtmgp_data() -> os.PathLike:
    """Download and extract RRTMGP data files.

    Downloads the RRTMGP data files from GitHub if not already present in the cache

    Returns:
        str: Path to the extracted data directory
    """
    # URL of the file to download
    REF: str = "v1.9.1"  # Can be a tag (e.g. "v1.8.2") or branch name (e.g. "main")
    DATA_URL: str = (
        "https://github.com/earth-system-radiation/rrtmgp-data/archive/refs/"
        f"{'tags' if REF.startswith('v') else 'heads'}/"
    )
    DATA_HASH = "340582fd3eb83c0f85b3393a3f24c694fc6fa71e8ce5fdb5f69e527855714a11"

    data_dir = pooch.os_cache("rrtmgp-data")

    downloader = pooch.create(
        path=data_dir,
        base_url=DATA_URL,
        registry={
            f"{REF}.zip": DATA_HASH,
        },
    )

    _ = downloader.fetch(f"{REF}.zip", processor=pooch.Unzip(extract_dir=f"{REF}"))

    ref_dirname = REF[1:] if REF.startswith("v") else REF
    return data_dir / f"{REF}" / f"rrtmgp-data-{ref_dirname}"
