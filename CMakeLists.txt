cmake_minimum_required (VERSION 3.18)
project(FORTRAN_C)

set(HEADERS fortran_interface.h)
set(SOURCES pybind_interface.cpp)

set(TARGET_NAME rttpy)

add_subdirectory(pybind11)

pybind11_add_module(${TARGET_NAME} ${SOURCES} ${HEADERS})

target_compile_definitions(${TARGET_NAME} PRIVATE 
    VERSION_INFO=${VERSION_INFO}
    DBL_EPSILON=${DBL_EPSILON}
)

message("====== ${DBL_EPSILON}")

target_include_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_SOURCE_DIR}/rte-rrtmgp/rte-kernels/api/
)


target_link_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_SOURCE_DIR}/rte-rrtmgp/build
)

target_link_libraries(${TARGET_NAME} PUBLIC 
    gfortran
    rrtmgp
    rte
    netcdff
    netcdf
)
