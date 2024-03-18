# /bin/bash

if [ -n "$CONDA_PREFIX" ] && [ "$OS" == *"Windows"* ] || [ -z "$CONDA_PREFIX" ]; then
    export FC=gfortran
fi

make -C rte-rrtmgp/build -j 2