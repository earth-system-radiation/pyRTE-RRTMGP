import os
import sys
import pytest
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pyrte.pyrte as py

########################
## test_zero_array_1D ##
########################

def test_invalid_array_dimension():
    with pytest.raises(RuntimeError) as excinfo:
        arr = np.ones((3, 3))
        py.zero_array_1D(arr)
    assert str(excinfo.value) == "Number of dimensions must be one"

def test_empty_array_exception():
    with pytest.raises(RuntimeError) as excinfo:
        arr = np.empty((0, ))
        py.zero_array_1D(arr)
    assert str(excinfo.value) == "Array size cannot be 0 or negative"

def test_zero_array_1D():

    shape = (4)
    arr = np.random.rand(shape)

    py.zero_array_1D(arr)
    assert np.all(arr == 0)

########################
## test_zero_array_2D ##
########################

def test_zero_array_2D():

    shape = (4, 4)
    arr = np.random.rand(*shape)

    py.zero_array_2D(arr)
    assert np.all(arr == 0)

########################
## test_zero_array_3D ##
########################

def test_zero_array_3D():

    shape = (4, 4, 4)
    arr = np.random.rand(*shape)

    py.zero_array_3D(arr)

    assert np.all(arr == 0)

########################
## test_zero_array_4D ##
########################

def test_zero_array_4D():

    shape = (4, 4, 4, 4)
    arr = np.random.rand(*shape)

    py.zero_array_4D(arr)
    assert np.all(arr == 0)