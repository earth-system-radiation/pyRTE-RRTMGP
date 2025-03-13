"""Data types for pyRTE-RRTMGP."""

from enum import StrEnum


class GasOpticsFiles(StrEnum):
    """Enumeration of default RRTMGP gas optics data files.

    This enum defines the available pre-configured gas optics data files that can be
    used with RRTMGP. The files contain absorption coefficients and other optical
    properties needed for radiative transfer calculations.

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


class CloudOpticsFiles(StrEnum):
    """Enumeration of default RRTMGP cloud optics data files.

    This enum defines the available pre-configured cloud optics data files that can be
    used with RRTMGP. The files contain cloud optical properties needed for radiative
    transfer calculations.

    Attributes:
        LW_BND: Longwave cloud optics file with band points
        LW_128: Longwave cloud optics file with 128 g-points
        LW_256: Longwave cloud optics file with 256 g-points
        SW_BND: Shortwave cloud optics file with band points
        SW_112: Shortwave cloud optics file with 112 g-points
        SW_224: Shortwave cloud optics file with 224 g-points
    """

    LW_BND = "rrtmgp-clouds-lw-bnd.nc"
    LW_G128 = "rrtmgp-clouds-lw-g128.nc"
    LW_G256 = "rrtmgp-clouds-lw-g256.nc"
    SW_BND = "rrtmgp-clouds-sw-bnd.nc"
    SW_G112 = "rrtmgp-clouds-sw-g112.nc"
    SW_G224 = "rrtmgp-clouds-sw-g224.nc"


class AerosolOpticsFiles(StrEnum):
    """Enumeration of default RRTMGP aerosol optics data files.

    This enum defines the available pre-configured aerosol optics data files that can be
    used with RRTMGP. The files contain aerosol optical properties needed for radiative
    transfer calculations.
    """

    LW_MERRA = "rrtmgp-aerosols-merra-lw.nc"
    SW_MERRA = "rrtmgp-aerosols-merra-sw.nc"


class RFMIPExampleFiles(StrEnum):
    """Enumeration of default RFMIP example files.

    This enum defines the available pre-configured example files that can be used with
    RRTMGP. The files contain example data for radiative transfer calculations.
    """

    RFMIP = (
        "examples/rfmip-clear-sky/inputs/"
        "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
    )
    REFERENCE_RLD = (
        "examples/rfmip-clear-sky/reference/"
        "rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
    )
    REFERENCE_RLU = (
        "examples/rfmip-clear-sky/reference/"
        "rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
    )
    REFERENCE_RSD = (
        "examples/rfmip-clear-sky/reference/"
        "rsd_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
    )
    REFERENCE_RSU = (
        "examples/rfmip-clear-sky/reference/"
        "rsu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc"
    )


class AllSkyExampleFiles(StrEnum):
    """Enumeration of default all-sky example files.

    This enum defines the available pre-configured all-sky example files that can be
    used with RRTMGP. The files contain example data for radiative transfer
    calculations.
    """

    LW_NO_AEROSOL = "examples/all-sky/reference/rrtmgp-allsky-lw-no-aerosols.nc"
    LW = "examples/all-sky/reference/rrtmgp-allsky-lw.nc"
    SW_NO_AEROSOL = "examples/all-sky/reference/rrtmgp-allsky-sw-no-aerosols.nc"
    SW = "examples/all-sky/reference/rrtmgp-allsky-sw.nc"


class ProblemTypes(StrEnum):
    """Enumeration of available radiation calculation types.

    This enum defines the different types of radiation calculations that can be
    performed, including both longwave and shortwave calculations with different
    solution methods.

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


class OpticsProblemTypes(StrEnum):
    """Enumeration of available optics problem types.

    This enum defines the different types of optics problems that can be
    solved with RRTMGP.

    Attributes:
        ABSORPTION: Absorption-only calculation
        N_STREAM: N-stream calculation
        TWO_STREAM: Two-stream approximation
    """

    ABSORPTION = "absorption"
    N_STREAM = "n-stream"
    TWO_STREAM = "two-stream"
