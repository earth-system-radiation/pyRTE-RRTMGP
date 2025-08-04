"""Data types for pyRTE-RRTMGP."""

from enum import StrEnum


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
