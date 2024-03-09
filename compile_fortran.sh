# Prerequisites:
#sudo apt-get update
#sudo apt-get install libnetcdff-dev -y
#sudo apt-get install gfortran-10 

# brew install netcdf

#sudo apt-get install gfortran-11 gfortran-12 -y
#conda env create -n rte_rrtmgp_test -f environment-noplots.yml
#conda activate rte_rrtmgp_test
#export FCFLAGS="-ffree-line-length-none -m64 -std=f2008 -march=native -fbounds-check -fmodule-private -fimplicit-none -finit-real=nan -g -DRTE_USE_CBOOL -DRTE_USE_DP"

export FC=gfortran-13
export FCFLAGS="-ffree-line-length-none -m64 -std=f2008 -march=native -fbounds-check -fPIC -fimplicit-none -finit-real=nan -g -DRTE_USE_CBOOL -DRTE_USE_DP"
export FCINCLUDE=-I/usr/include
export RRTMGP_ROOT=$(pwd)/rte-rrtmgp
export RRTMGP_DATA=$(pwd)/rte-rrtmgp/rrtmgp-data
export FAILURE_THRESHOLD=5.8e-2

$FC --version


make -C $RRTMGP_ROOT clean
make -C $RRTMGP_ROOT libs -j4
