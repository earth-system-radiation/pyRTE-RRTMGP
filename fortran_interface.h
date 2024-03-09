#ifndef FORTRAN_INTERFACE
#define FORTRAN_INTERFACE

typedef double Float;

namespace fortran {
    extern "C" {
        void add(const double *a, const double *b, double *result);

        void hello_world();

        void zero_array_1D(const int* ni, Float* array);
    }
}

#endif //FORTRAN_INTERFACE
