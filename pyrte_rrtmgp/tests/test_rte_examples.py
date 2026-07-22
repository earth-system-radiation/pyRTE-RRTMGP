import netCDF4  # noqa
import xarray as xr

import pytest
from typing import Dict, Optional, Any

from pyrte_rrtmgp.rrtmgp.data_files import GasOpticsFiles
from pyrte_rrtmgp.rrtmgp.examples import RTE_EXAMPLES

from pyrte_rrtmgp.rte_examples import (
    load_rte_example_file,
    RTEExamplesFiles,
)

from pyrte_rrtmgp import rte
from pyrte_rrtmgp.rrtmgp import GasOptics as GasOpticsRRTMGP


def _test_get_fluxes_from_rte_example(
                example: RTEExamplesFiles,
                gas_optics: GasOpticsRRTMGP,
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
# Better to build the testing variants (combos of state and gas optics)
#    programatically
#
# pyRRTMGP LW GasOptics fails using 128 g-points
#
# Should add tests against reference answers
#
@pytest.mark.parametrize(
    "example, gas_optics",
    [
        ("RCE_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.LW_G256,)
        ),
        ("RCE_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.SW_G112,)
        ),
        ("RFMIP_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.LW_G256,)
        ),
        ("RFMIP_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.SW_G112,)
        ),
        ("CKDMIP_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.LW_G256,)
        ),
        ("CKDMIP_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.SW_G112,)
        ),
    ],
)

def test_rte_example(example, gas_optics) -> None:
    """ """
    _test_get_fluxes_from_rte_example(
                example,
                gas_optics,
        )

@pytest.mark.parametrize(
    "example, gas_optics",
    [
        ("RCE_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.LW_G256,)
        ),
        ("RCE_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.SW_G112,)
        ),
        ("RFMIP_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.LW_G256,)
        ),
        ("RFMIP_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.SW_G112,)
        ),
        ("CKDMIP_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.LW_G256,)
        ),
        ("CKDMIP_STATES", GasOpticsRRTMGP( # type: ignore
                              gas_optics_file = GasOpticsFiles.SW_G112,)
        ),
    ],
)
def test_rte_example_with_dask(example, gas_optics) -> None:
    """ """
    _test_get_fluxes_from_rte_example(
                example,
                gas_optics,
                use_dask = True,
        )
