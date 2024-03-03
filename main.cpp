#include <cstdio>

#include "fortran_interface.h"

double add(double a, double b) {
    double res = -1;
    fortran_add(&a, &b, &res);
    return res;
}

int main() {

    fortran_hello_world();

    double c = add(2, 3);

    printf("%d\n", int(c));   

    return 0;
}