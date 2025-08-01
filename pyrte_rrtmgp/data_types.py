"""Data types for pyRTE-RRTMGP."""

from enum import StrEnum


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


class OpticsTypes(StrEnum):
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
