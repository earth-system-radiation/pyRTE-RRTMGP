#!/bin/bash

# Execute Fortran compilation script
./fcompile.sh

# Execute Python setup script
python3 setup.py build_ext --inplace

cp test.py rttpy/
