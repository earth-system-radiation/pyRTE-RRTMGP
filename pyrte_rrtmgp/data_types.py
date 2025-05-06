"""Data types for pyRTE-RRTMGP."""

from enum import StrEnum


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


class ProblemTypes(StrEnum):
    """Enumeration of available radiation calculation types.

    This enum defines the different types of radiation calculations that can be
    performed, including both longwave and shortwave calculations with different
    solution methods.
    """

    LW_ABSORPTION = "Longwave absorption"
    """Longwave absorption-only calculation"""

    LW_2STREAM = "Longwave 2-stream"
    """Longwave two-stream approximation calculation"""

    SW_DIRECT = "Shortwave direct"
    """Shortwave direct beam calculation"""

    SW_2STREAM = "Shortwave 2-stream"
    """Shortwave two-stream approximation calculation"""


class OpticsProblemTypes(StrEnum):
    """Enumeration of available optics problem types.

    This enum defines the different types of optics problems that can be
    solved with RRTMGP.
    """

    ABSORPTION = "absorption"
    """Absorption-only calculation"""

    N_STREAM = "n-stream"
    """N-stream calculation"""

    TWO_STREAM = "two-stream"
    """Two-stream approximation"""
