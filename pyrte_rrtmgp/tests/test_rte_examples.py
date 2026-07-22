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

from pyrte_rrtmgp.rrtmgp.examples   import RTE_EXAMPLES
from pyrte_rrtmgp.rrtmgp.data_files import GasOpticsFiles
from pyrte_rrtmgp.rrtmgp import GasOptics as GasOpticsRRTMGP
from pyrte_rrtmgp.ssm    import (
    SSM_W26,
    GasOptics as GasOpticsSSM,
)
#
# pyRRTMGP LW GasOptics fails using 128 g-points
#
print("GasOpticsFiles.LW_G256: ", GasOpticsFiles.LW_G256)
GasOpticsGP_LW: GasOpticsRRTMGP = GasOpticsRRTMGP(gas_optics_file = GasOpticsFiles.LW_G256)
GasOpticsGP_SW: GasOpticsRRTMGP = GasOpticsRRTMGP(gas_optics_file = GasOpticsFiles.SW_G112)
# Wavenumber grid and spectral band widths used by the Planck source terms.
_ssm_nus = xr.DataArray(
    np.array([100.0, 200.0, 500.0, 700.0, 1000.0]),
    dims=("gpt",),
    name="nus",
    attrs={"units": "cm^-1"},
)

_ssm_dnus = xr.DataArray(
    np.array([100.0, 100.0, 300.0, 200.0, 300.0]),
    dims=("gpt",),
    coords={"gpt": _ssm_nus["gpt"]},
    name="dnus",
    attrs={"units": "cm^-1"},
)
GasOpticsSSM_LW: GasOpticsSSM = GasOpticsSSM(
    spectral_data=SSM_W26,
    nus=_ssm_nus,
    dnus=_ssm_dnus,
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
# Should add tests against reference answers
#
@pytest.mark.parametrize(
    "example, gas_optics",
    [   (state, gas_optics)
         for gas_optics in [GasOpticsGP_LW, GasOpticsGP_SW] \
         for state in ["RCE_STATES", "RFMIP_STATES", "CKDMIP_STATES"]
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
