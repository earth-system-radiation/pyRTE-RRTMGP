#  Compile Fortran
echo "Compiling"

mkdir -p build
cd build

gfortran -fpic -c ../test.F90
echo "Linking"
# Create a shared library from the fortran code
gfortran -shared -o libtest.so test.o
# Build an executable program from the fortran code
gfortran -o fortran_test.bin test.o

# Print what functions are visible in the dll
nm -g -C --defined-only libtest.so 

# Build example C program using the fortran functions
g++ -L${PWD} -ltest -o c_test.bin ../main.cpp 

# Build C example with debug information
#g++ -g -O0 -L${PWD} -ltest -o c_test.bin ../main.cpp 