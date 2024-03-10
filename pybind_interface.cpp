#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace fortran {
#include "rte-rrtmgp/rte-kernels/api/rte_kernels.h"
}

namespace py = pybind11;

PYBIND11_MODULE(rrtmgppy, m) {

    m.def("zero_array_1D", [](py::array_t<double> arr){
        py::buffer_info buf_info = arr.request();

        if (buf_info.ndim != 1) {
            throw std::runtime_error("Number of dimensions must be one");
        }

        if (buf_info.size <= 0) {
            throw std::runtime_error("Array size cannot be 0 or negative");
        }
        
        if (buf_info.size >= INT_MAX) {
            throw std::runtime_error("Array size bigger than INT_MAX");
        }

        int ni = int(buf_info.size);
        double *ptr = static_cast<double *>(buf_info.ptr);
        fortran::zero_array_1D(&ni, ptr);
    });
}
