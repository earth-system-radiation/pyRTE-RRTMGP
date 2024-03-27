#!/bin/sh

if [ -n "$CONDA_PREFIX" ] && [ "$OS" = *"Windows"* ]; then
    export FC=gfortran
elif [ -z "$CONDA_PREFIX" ]; then
    export FC=gfortran
fi

make -C rte-rrtmgp/build -j 2
