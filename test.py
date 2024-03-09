import rttpy as py
import numpy as np

def dimension_exception_test():
    try:
        arr = np.ones((3, 3))
        py.zero_array_1D(arr)
        print("TEST FAILED")
    except Exception as e:
        if "Number of dimensions must be one" == str(e):
            print("TEST PASSED")
        else:
            print(f"TEST FAILED | {e}")

def size_exception_test():
    try:
        arr = np.empty((0, ))
        py.zero_array_1D(arr)
        print("TEST FAILED")
    except Exception as e:
        if "Array size cannot be 0 or negative" == str(e):
            print("TEST PASSED")
        else:
            print(f"TEST FAILED | {e}")


def zero_nparray_test():
    try:
        arr = np.random.rand(10)
        print(arr)
        py.zero_array_1D(arr)
        print(arr)
    except Exception as e:
        print(f"TEST FAILED | {e}")


dimension_exception_test()
size_exception_test()
zero_nparray_test()
