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
# from pyrte_rrtmgp.tests import ERROR_TOLERANCE
# Replaced with a relative tolerance of 5e-4 in test_rte_example_values
#   RFMIP states have largest discrepencies

#
# RRTMGP optics
#
from pyrte_rrtmgp.rrtmgp.examples import (
    RTE_EXAMPLES as RTE_EXAMPLES_RRTMGP,
    load_example_file as load_example_results_rrtmgp,
)
from pyrte_rrtmgp.rrtmgp.data_files import GasOpticsFiles
from pyrte_rrtmgp.rrtmgp import GasOptics as GasOpticsRRTMGP

#
# SSM optics
#
from pyrte_rrtmgp.rrtmgp.examples   import (
    RTE_EXAMPLES as RTE_EXAMPLES_SSM,
    load_example_file as load_example_results_ssm,
)
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

def _test_fluxes_from_rte_example(
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
# Check that fluxes can be computed
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
    _test_fluxes_from_rte_example(
                example,
                gas_optics,
        )

#
# Check that fluxes can be computed with dask
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
    _test_fluxes_from_rte_example(
                example,
                gas_optics,
                use_dask = True,
        )


#
# Compare fluxes to reference values
#   These tolerances are relative and looser than Fortran - Fortran comparisons
#
@pytest.mark.parametrize(
    "example, gas_optics, reference",
    [   (state + "_STATES",
         GasOpticsGP_SW, \
         load_example_results_rrtmgp(RTE_EXAMPLES_RRTMGP["REF_" + state + "_SW"]) \
         ) \
         for state in ["RCE", "RFMIP", "CKDMIP"]
    ],
)
def test_rte_example_values(example, gas_optics, reference) -> None:
    """ """
    fluxes = _test_fluxes_from_rte_example(
                example,
                gas_optics,
    )
    xr.testing.assert_allclose(fluxes. \
        transpose("variant", "level", "col"). \
        rename_vars({"sw_flux_down":"sw_flux_dn"}).drop_vars({"col", "variant"}),
        reference,
        rtol = 5e-4)
