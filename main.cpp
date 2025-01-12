#include <cstdio>

namespace fortran {
#include "rte-rrtmgp/rte-kernels/api/rte_kernels.h"
}

int main() {
    // Test zero_array_1D which takes in an array of doubles and zeros it out
    const int ni = 20;
    double arr[ni];
    for (int i = 0; i < ni; i++) {
        arr[i] = i;
    }

    fortran::zero_array_1D(&ni, arr);

    for (int i = 0; i < ni; i++) {
        printf("%lf, ", arr[i]);
    }


    return 0;
}
