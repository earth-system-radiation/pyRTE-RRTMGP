import pytest
import numpy as np
import pyrte.pyrte as py

#####################
## test_zero_array ##
#####################

@pytest.mark.parametrize("array,method,error", [
    (np.ones((3, 3)), py.zero_array_1D, "Number of dimensions must be one"),
    (np.ones((3, 3, 3)), py.zero_array_2D, "Number of dimensions must be 2"),
    (np.ones((3, 3)), py.zero_array_3D, "Number of dimensions must be 3"),
    (np.ones((3, 3)), py.zero_array_4D, "Number of dimensions must be 4"),
])
def test_invalid_array_dimension(array, method, error):
    with pytest.raises(RuntimeError) as excinfo:
        method(array)
    assert str(excinfo.value) == error

@pytest.mark.parametrize("array,method", [
    (np.ones((0, )), py.zero_array_1D),
    (np.ones((0, 0)), py.zero_array_2D),
    (np.ones((0, 0, 0)), py.zero_array_3D),
    (np.ones((0, 0, 0, 0)), py.zero_array_4D),
])
def test_empty_array_exception(array, method):
    with pytest.raises(RuntimeError) as excinfo:
        method(array)
    assert str(excinfo.value) == "Array size cannot be 0 or negative"

@pytest.mark.parametrize("shape, fortran_zero_array", [
    ((4, ), py.zero_array_1D),
    ((4, 4), py.zero_array_2D),
    ((4, 4, 4), py.zero_array_3D),
    ((4, 4, 4, 4), py.zero_array_4D),
])
def test_zero_array(shape, fortran_zero_array):
    arr = np.random.rand(*shape)
    fortran_zero_array(arr)
    assert np.all(arr == 0)