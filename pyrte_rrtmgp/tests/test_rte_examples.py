import netCDF4  # noqa
import xarray as xr
import numpy as np

import pytest
from typing import Dict, Optional, Any

from pyrte_rrtmgp import rte
from pyrte_rrtmgp.rte_examples import (
    load_rte_example_file,
    RTEExamplesFiles,
)

#
# RRTMGP optics
#
from pyrte_rrtmgp.rrtmgp.examples   import RTE_EXAMPLES
from pyrte_rrtmgp.rrtmgp.data_files import GasOpticsFiles
from pyrte_rrtmgp.rrtmgp import GasOptics as GasOpticsRRTMGP
#
# SSM optics
#
from pyrte_rrtmgp.ssm import (
    SSM_W26,
    GasOptics as GasOpticsSSM,
)

#
# pyRRTMGP LW GasOptics fails using 128 g-points
#
GasOpticsGP_LW: GasOpticsRRTMGP = GasOpticsRRTMGP(gas_optics_file = GasOpticsFiles.LW_G256)
GasOpticsGP_SW: GasOpticsRRTMGP = GasOpticsRRTMGP(gas_optics_file = GasOpticsFiles.SW_G112)

#
# Define SSM optics
#   Wavenumber grid and spectral band widths used by the Planck source terms.
#
nus = np.linspace(50., 3000., 41)
_mids = 0.5 * (nus[:-1] + nus[1:])
dnus = np.concatenate([_mids, [3500.]]) - np.concatenate([[0.], _mids])

GasOpticsSSM_LW: GasOpticsSSM = GasOpticsSSM(
    spectral_data=SSM_W26,
    nus=nus,
    dnus=dnus,
    pref=SSM_W26.pref,
)

def _test_get_fluxes_from_rte_example(
                example: RTEExamplesFiles,
                gas_optics: GasOpticsRRTMGP | GasOpticsSSM,
                use_dask: bool = False) -> xr.Dataset:
    """Runs rte-examples clear-sky problems to exercise gas optics and solvers"""

    atmosphere = load_rte_example_file(RTEExamplesFiles[example])
    if use_dask:
        atmosphere = atmosphere.chunk({"variant": 1})

    # Compute gas optics for the atmosphere
    gas_optics.compute( # type: ignore
        atmosphere,
    )

    # Solve RTE
    fluxes: xr.Dataset = atmosphere.rte.solve(add_to_input=False)

    assert fluxes is not None
    assert ~xr.ufuncs.isnan(fluxes).any()

    return fluxes


#
#### Tests start here
#
# Should add checks against reference answers
#
@pytest.mark.parametrize(
    "example, gas_optics",
    [   (state, gas_optics)
         for gas_optics in [GasOpticsGP_LW, GasOpticsGP_SW, GasOpticsSSM_LW] \
         for state in ["RCE_STATES", "RFMIP_STATES", "CKDMIP_STATES"]
    ],
)
def test_rte_example(example, gas_optics) -> None:
    """ """
    _test_get_fluxes_from_rte_example(
                example,
                gas_optics,
        )

#
# SSM gas optics doesn't play well with dask
#
@pytest.mark.parametrize(
    "example, gas_optics",
    [   (state, gas_optics)
         for gas_optics in [GasOpticsGP_LW, GasOpticsGP_SW] \
         for state in ["RCE_STATES", "RFMIP_STATES", "CKDMIP_STATES"]
    ],
)
def test_rte_example_with_dask(example, gas_optics) -> None:
    """ """
    _test_get_fluxes_from_rte_example(
                example,
                gas_optics,
                use_dask = True,
        )
