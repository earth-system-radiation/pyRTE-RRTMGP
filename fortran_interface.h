#ifndef FORTRAN_INTERFACE
#define FORTRAN_INTERFACE


namespace f {
    extern "C" void fortran_add(const double *a, const double *b, double *result);

    extern "C" void fortran_hello_world();
}

#endif //FORTRAN_INTERFACE