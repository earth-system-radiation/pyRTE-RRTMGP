#include <cstdio>

#include "fortran_interface.h"

double add(double a, double b) {
    double res = -1;
    f::fortran_add(&a, &b, &res);
    return res;
}

int main() {

    // Test Hello world. calling a void/void function from C++
    f::fortran_hello_world();

    // Test add. calling a function that takes in 2 doubles and returns a double
    double c = add(2, 3);
    printf("%d\n", int(c));

    // Test zero_array_1D which takes in an array of doubles and zeros it out
    const int ni = 20;
    double arr[ni];
    for (int i = 0; i < ni; i++) {
        arr[i] = i;
    }

    f::fortran_zero_array_1D(&ni, arr);

    for (int i = 0; i < ni; i++) {
        printf("%lf, ", arr[i]);
    }


    return 0;
}