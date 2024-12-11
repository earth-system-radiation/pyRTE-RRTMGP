from enum import Enum, StrEnum


class GasOpticsFiles(StrEnum):
    """Enumeration of default RRTMGP gas optics data files.

    This enum defines the available pre-configured gas optics data files that can be used
    with RRTMGP. The files contain absorption coefficients and other optical properties
    needed for radiative transfer calculations.

    Attributes:
        LW_G128: Longwave gas optics file with 128 g-points
        LW_G256: Longwave gas optics file with 256 g-points
        SW_G112: Shortwave gas optics file with 112 g-points
        SW_G224: Shortwave gas optics file with 224 g-points
    """

    LW_G128 = "rrtmgp-gas-lw-g128.nc"
    LW_G256 = "rrtmgp-gas-lw-g256.nc"
    SW_G112 = "rrtmgp-gas-sw-g112.nc"
    SW_G224 = "rrtmgp-gas-sw-g224.nc"


class ProblemTypes(StrEnum):
    """Enumeration of available radiation calculation types.

    This enum defines the different types of radiation calculations that can be performed,
    including both longwave and shortwave calculations with different solution methods.

    Attributes:
        LW_ABSORPTION: Longwave absorption-only calculation
        LW_2STREAM: Longwave two-stream approximation calculation
        SW_DIRECT: Shortwave direct beam calculation
        SW_2STREAM: Shortwave two-stream approximation calculation
    """

    LW_ABSORPTION = "Longwave absorption"
    LW_2STREAM = "Longwave 2-stream"
    SW_DIRECT = "Shortwave direct"
    SW_2STREAM = "Shortwave 2-stream"
