#!/usr/bin/env python3

import pyrte.pyrte as py
import numpy as np


def dimension_exception_test():
    try:
        arr = np.ones((3, 3))
        py.zero_array_1D(arr)
        print("[dimension_exception_test] TEST FAILED")
    except Exception as e:
        if "Number of dimensions must be one" == str(e):
            print("[dimension_exception_test] TEST PASSED")
        else:
            print(f"[dimension_exception_test] TEST FAILED | {e}")

def size_exception_test():
    try:
        arr = np.empty((0, ))
        py.zero_array_1D(arr)
        print("[size_exception_test] TEST FAILED")
    except Exception as e:
        if "Array size cannot be 0 or negative" == str(e):
            print("[size_exception_test] TEST PASSED")
        else:
            print(f"[size_exception_test] TEST FAILED | {e}")


def zero_nparray_1D_test():
    try:
        arr = np.random.rand(10)
        print(f"Random array of size {arr.shape} : {arr}")
        py.zero_array_1D(arr)
        print(f"Array after zero_array_1D : {arr}")
    except Exception as e:
        print(f"TEST FAILED | {e}")

def zero_nparray_2D_test():
    try:
        arr = np.random.rand(16)
        arr = arr.reshape((4,4))

        print(f"Random array of size {arr.shape} : {arr}")
        py.zero_array_2D(arr)
        print(f"Array after zero_array_1D : {arr}")
    except Exception as e:
        print(f"TEST FAILED | {e}")


dimension_exception_test()
size_exception_test()
zero_nparray_1D_test()
zero_nparray_2D_test()
