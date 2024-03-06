#ifndef FORTRAN_INTERFACE
#define FORTRAN_INTERFACE

using Float = double;

namespace f {
    extern "C" {
        void fortran_add(const double *a, const double *b, double *result);

        void fortran_hello_world();

        void fortran_zero_array_1D(const int* ni, Float* array);
    }
}

#endif //FORTRAN_INTERFACE