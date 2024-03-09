import rttpy as py
import numpy as np

def dimension_test():
    try:
        arr = np.ones((3, 3))
        py.zero_array_1D(arr)
        print("TEST FAILED")
    except Exception as e:
        if "Number of dimensions must be one" == str(e):
            print("TEST PASSED")
        else:
            print("TEST FAILED")

def size_test():
    try:
        arr = np.empty((0, ))
        py.zero_array_1D(arr)
        print("TEST FAILED")
    except Exception as e:
        if "Array size cannot be 0 or negative" == str(e):
            print("TEST PASSED")
        else:
            print("TEST FAILED")


dimension_test()
size_test()