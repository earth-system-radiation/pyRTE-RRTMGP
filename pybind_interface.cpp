#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "fortran_interface.h"

namespace py = pybind11;


PYBIND11_MODULE(pybind_interface, m) {

    m.def("zero_array_1D", [](int ni, pybind11::array_t<double> arr){
        pybind11::buffer_info buf_info = arr.request();
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
