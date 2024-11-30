from enum import Enum


class GasOpticsFiles(Enum):
    """Enumeration of default RRTMGP gas optics data files."""

    LW_G128 = "rrtmgp-gas-lw-g128.nc"
    LW_G256 = "rrtmgp-gas-lw-g256.nc"
    SW_G112 = "rrtmgp-gas-sw-g112.nc"
    SW_G224 = "rrtmgp-gas-sw-g224.nc"


class ProblemTypes(Enum):
    LW_ABSORPTION = "Longwave absorption"
    LW_2STREAM = "Longwave 2-stream"
    SW_DIRECT = "Shortwave direct"
    SW_2STREAM = "Shortwave 2-stream"
