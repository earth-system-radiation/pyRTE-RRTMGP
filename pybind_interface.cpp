#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace fortran {
#include "rte_kernels.h"
}

namespace py = pybind11;

PYBIND11_MODULE(pyrte, m) {

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


    m.def("zero_array_2D", [](py::array_t<double> arr){
        py::buffer_info buf_info = arr.request();

        if (buf_info.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be 2");
        }

        if (buf_info.size <= 0) {
            throw std::runtime_error("Array size cannot be 0 or negative");
        }

        if (buf_info.shape[0] >= INT_MAX || buf_info.shape[1] >= INT_MAX) {
            throw std::runtime_error("Array dim bigger than INT_MAX");
        }

        int ni = int(buf_info.shape[0]);
        int nj = int(buf_info.shape[1]);

        double *ptr = static_cast<double *>(buf_info.ptr);
        fortran::zero_array_2D(&ni, &nj, ptr);
    });

    m.def("zero_array_3D", [](py::array_t<double> arr){
        py::buffer_info buf_info = arr.request();

        if (buf_info.ndim != 3) {
            throw std::runtime_error("Number of dimensions must be 3");
        }

        if (buf_info.size <= 0) {
            throw std::runtime_error("Array size cannot be 0 or negative");
        }

        if (buf_info.shape[0] >= INT_MAX || buf_info.shape[1] >= INT_MAX || buf_info.shape[2] >= INT_MAX) {
            throw std::runtime_error("Array dim bigger than INT_MAX");
        }

        int ni = int(buf_info.shape[0]);
        int nj = int(buf_info.shape[1]);
        int nk = int(buf_info.shape[2]);

        double *ptr = static_cast<double *>(buf_info.ptr);
        fortran::zero_array_3D(&ni, &nj, &nk, ptr);
    });

    m.def("zero_array_4D", [](py::array_t<double> arr){
        py::buffer_info buf_info = arr.request();

        if (buf_info.ndim != 4) {
            throw std::runtime_error("Number of dimensions must be 4");
        }

        if (buf_info.size <= 0) {
            throw std::runtime_error("Array size cannot be 0 or negative");
        }

        if (buf_info.shape[0] >= INT_MAX || buf_info.shape[1] >= INT_MAX || buf_info.shape[2] >= INT_MAX || buf_info.shape[3] >= INT_MAX) {
            throw std::runtime_error("Array dim bigger than INT_MAX");
        }

        int ni = int(buf_info.shape[0]);
        int nj = int(buf_info.shape[1]);
        int nk = int(buf_info.shape[2]);
        int nl = int(buf_info.shape[3]);

        double *ptr = static_cast<double *>(buf_info.ptr);
        fortran::zero_array_4D(&ni, &nj, &nk, &nl, ptr);
    });
}
