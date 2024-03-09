#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "fortran_interface.h"

namespace py = pybind11;

PYBIND11_MODULE(rttpy, m) {

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
        f::fortran_zero_array_1D(&ni, ptr);
    });

    m.def("add", [](double a, double b) {
        double res = -1;
        f::fortran_add(&a, &b, &res);
        return res;
    });

    m.def("hello_world", []() {
        f::fortran_hello_world();
    });
}
