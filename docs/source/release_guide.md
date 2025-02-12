# Steps When a New Release is Active from RTE-RRTMGP

## Update Check

When a new release of [RTE-RRTMGP](https://github.com/earth-system-radiation/rte-rrtmgp/) is published and changes are made to the functions we bind and support, a manual update is required in [pybind_interface.cpp](../../pybind_interface.cpp).

A good rule of thumb is to run it blindly and check for any errors that occur.

If errors appear, there are two potential issues:

1. **Differences in function parameters:**
    - New parameters may be added, or some may be removed;
    - Changes in the dimensions;
    - Changes in the data types.

    These changes also affect the Python implementation, which needs to be updated accordingly.
2. **Differences in input and/or output data:**
    - This can often be resolved by updating the version of the [RRTMGP-DATA](https://github.com/earth-system-radiation/rrtmgp-data/) repository.

If a new function is introduced to the bindings, follow this procedure to add it to the Python repository:

- Add the new function to [pybind_interface.cpp](../../pybind_interface.cpp), ensuring proper error checking for valid dimensions and inputs.
- Add the new function to either [rte.py](../../pyrte_rrtmgp/kernels/rte.py) or [rrtmgp.py](../../pyrte_rrtmgp/kernels/rrtmgp.py).
- Implement appropriate tests for the new functionality. If a test for this functionality exists in [RTE-RRTMGP](https://github.com/earth-system-radiation/rte-rrtmgp/), the new test should mimic it.
- Changes to functions may also require updates to input and reference data for the tests in [RRTMGP-DATA](https://github.com/earth-system-radiation/rrtmgp-data/), necessitating corresponding updates in PYRTE-RRTMGP's tests.

## Runing build and tests

To build the wheel locally:
```
pip install -e .
```
To run the tests:
```
pytest
```

If everything passes, you are ready to go!
