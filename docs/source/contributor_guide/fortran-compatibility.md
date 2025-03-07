# Steps When a New Release of RTE-RRTMGP is Active

When a new release of [RTE-RRTMGP](https://github.com/earth-system-radiation/rte-rrtmgp/) is published and changes are made to the functions we bind and support, a manual update is required in [pybind_interface.cpp](../../pybind_interface.cpp).

## Steps to Use the New Version of RTE-RRTMGP (New Source)

In [CMakeLists.txt](../../CMakeLists.txt), to use a different version of RTE-RRTMGP, we need to change the `GIT_TAG` to the desired version.

Clone or download the source code for the version of [RTE-RRTMGP](https://github.com/earth-system-radiation/rte-rrtmgp/) you want to use. To build it, you can use the setup script provided in the source code.

There is a script designed to provide general information about the state of the Pybinds and Cbinds from Fortran:
```
python check_binds.py --c_headers /path/to/rrtmgp_kernels.h /path/to/rte_kernels.h --pybind /path/to/pybind_interface.cpp
```

With the built source of RTE-RRTMGP, you can check the current Pybinds.

An example output looks like this:
```
Total C functions found: 41
Total Py functions bound: 41

Warning: Missing C function bindings:
- rte_net_byband_full
- rte_sum_byband

Warning: Missing Py function bindings:
- rte_lw_source_2str
- rte_sw_source_dir
```

This helps identify missing bindings, naming differences, or functions that have been removed.

## Testing Current Bindings with the New Source

After running the script, a good rule of thumb is to build and test, checking for any errors that occur.
To build, use the following commands:
```
python -m venv .venv_test
source .venv_test/bin/activate

pip install numpy xarray netcdf4 requests
pip install --upgrade cmake scikit-build pybind11 pytest
pip install -e .
```
The expected output should look something like this:
```
...
Successfully built pyrte_rrtmgp
Installing collected packages: pyrte_rrtmgp
Successfully installed pyrte_rrtmgp-0.0.6
```

To run the tests:
```
pytest -v
```

The expected output should look something like this:
```
======================================================================================================== test session starts ========================================================================================================
platform linux -- Python 3.12.8, pytest-8.3.4, pluggy-1.5.0 -- /data/vkm/code/makepath/columbia/.venv_test/bin/python
cachedir: .pytest_cache
rootdir: /data/vkm/code/makepath/columbia
configfile: pyproject.toml
testpaths: pyrte_rrtmgp/tests
collected 47 items

...
pyrte_rrtmgp/tests/test_exported_functions.py::test_rte_increment_nstream_by_nstream PASSED                                                                                                                                   [ 95%]
pyrte_rrtmgp/tests/test_python_frontend/test_lw_solver.py::test_lw_solver_noscat PASSED                                                                                                                                       [ 97%]
pyrte_rrtmgp/tests/test_python_frontend/test_sw_solver.py::test_sw_solver_noscat PASSED                                                                                                                                       [100%]

======================================================================================================== 47 passed in 27.71s ========================================================================================================
```

If errors appear, there are two potential issues:

1. **Differences in function parameters:**
	- New parameters may have been added, or some may have been removed.
	- Changes in dimensions.
	- Changes in data types.

	These changes also affect the Python implementation, which must be updated accordingly.

2. **Differences in input and/or output data:**
	- This can often be resolved by updating the version of the [RRTMGP-DATA](https://github.com/earth-system-radiation/rrtmgp-data/) repository.

Example case: There is a change in the dimensions of the `cl_gas` parameter in `rrtmgp_compute_tau_rayleigh`, the build process for the wheel succeeds, but running pytest results in the following error:
```
pyrte_rrtmgp/kernels/rrtmgp.py:466: RuntimeError
========================================================================================= short test summary info =========================================================================================
FAILED pyrte_rrtmgp/tests/test_python_frontend/test_sw_solver.py::test_sw_solver_noscat - RuntimeError: Invalid size for input array 'col_gas'
====================================================================================== 1 failed, 46 passed in 32.04s ======================================================================================
```

In this situation, the solution is to check the dimensions in cbind and ensure they are correctly reflected in pybind and the Python implementation. Most error cases are similar to this.

If there is a difference in the outputs and everything appears to be correct, the problem is most likely in the input/output data. Check for any changes in the Fortran tests and replicate them accordingly.

## Adding New Bindings

If a new function is introduced to the bindings, follow this procedure to add it to the Python repository:

- Add the new function to [pybind_interface.cpp](../../pybind_interface.cpp), ensuring proper error checking for valid dimensions and inputs.
- Add the new function to either [rte.py](../../pyrte_rrtmgp/kernels/rte.py) or [rrtmgp.py](../../pyrte_rrtmgp/kernels/rrtmgp.py).
- Implement appropriate tests for the new functionality. If a test for this functionality exists in [RTE-RRTMGP](https://github.com/earth-system-radiation/rte-rrtmgp/), the new test should mimic it.
- Changes to functions may also require updates to input and reference data for the tests in [RRTMGP-DATA](https://github.com/earth-system-radiation/rrtmgp-data/), necessitating corresponding updates in PYRTE-RRTMGP’s tests.

Example Blueprint of a Bound Function:
Cbind function:
```c++
void rte_sum_broadband(
	const int& ncol,
	const int& nlev,
	const int& ngpt,
	const Float* spectral_flux, // Dims: (ncol, nlev, ngpt)
	Float* broadband_flux // Dims: (ncol, nlev)
);
```

Pybind function:
```c++
// Declaration of the Pybind function
m.def("rte_sum_broadband",
[](
// Mapping the data types with the corresponding properties for Pybind11
	int ncol,
	int nlev,
	int ngpt,
	py::array_t<Float> gpt_flux,
	py::array_t<Float> flux
) {
// Error checking for dimensions
	if (ncol <= 0 || nlev <= 0 || ngpt <= 0) {
		throw std::runtime_error("ncol, nlev, and ngpt must be positive integers.");
	}

// Error checking for array sizes
	if (gpt_flux.size() != ncol * nlev * ngpt) {
		throw std::runtime_error("Invalid size for input array 'gpt_flux': expected (ncol, nlev, ngpt).");
	}
	if (flux.size() != ncol * nlev) {
		throw std::runtime_error("Invalid size for output array 'flux': expected (ncol, nlev).");
	}

// Request buffer information
	py::buffer_info buf_gpt_flux = gpt_flux.request();
	py::buffer_info buf_flux = flux.request();

// Calling the C-bound function
	fortran::rte_sum_broadband(
		ncol,
		nlev,
		ngpt,
		reinterpret_cast<Float*>(buf_gpt_flux.ptr),
		reinterpret_cast<Float*>(buf_flux.ptr)
	);
});
```
