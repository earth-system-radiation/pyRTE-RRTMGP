"""Utility functions for pyRTE-RRTMGP."""

from typing import Any

import numpy as np


def safer_divide(a: Any, b: Any) -> Any:
    """Safer np.divide util func."""
    return np.divide(a, b, out=np.zeros_like(a), where=b > np.finfo(float).eps)
