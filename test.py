import pybind_interface as py
import numpy as np

arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.double)

py.zero_array_1D(10, arr)

print(py.add(10, 5))

py.hello_world()