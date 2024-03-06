#include <pybind11/pybind11.h>
#include "fortran_interface.h"

namespace py = pybind11;

PYBIND11_MODULE(pybind_interface, m) {
    m.def("add", [](double a, double b) {
        double res = -1;
        f::fortran_add(&a, &b, &res);
        return res;
    });

    m.def("hello_world", []() {
        f::fortran_hello_world();
    });
}
