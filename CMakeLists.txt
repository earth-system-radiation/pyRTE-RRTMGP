cmake_minimum_required(VERSION 3.18...3.26)
project(rte-rrtmgp)

# Add a custom target that always runs
add_custom_target(compile-rte-rrtmgp ALL)

set(LINUX FALSE)
if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# Check if FC environment variable is set
if(DEFINED ENV{FC})
    set(CMAKE_Fortran_COMPILER $ENV{FC})
    message(STATUS "Using Fortran compiler from FC environment variable: $ENV{FC}")
else()
    # Define a list of preferred Fortran compilers
    set(PREFERRED_FC_COMPILERS gfortran ifort gfortran-10 gfortran-11 f77)

    foreach(compiler IN LISTS PREFERRED_FC_COMPILERS)
        find_program(FOUND_COMPILER NAMES ${compiler})
        if(FOUND_COMPILER)
            set(CMAKE_Fortran_COMPILER ${FOUND_COMPILER})
            message(STATUS "Using Fortran compiler: ${FOUND_COMPILER}")
            break()
        endif()
    endforeach()
endif()

if(NOT CMAKE_Fortran_COMPILER)
    message(FATAL_ERROR "No suitable Fortran compiler found")
endif()

include(ExternalProject REQUIRED)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(BUILD_COMMAND_STRING "set FC=%CMAKE_Fortran_COMPILER% && cd build && nmake /A")
else()
    set(BUILD_COMMAND_STRING "FC=${CMAKE_Fortran_COMPILER} make -C build -j ${N}")
endif()

ExternalProject_Add(
    rte-rrtmgp
    GIT_REPOSITORY https://github.com/earth-system-radiation/rte-rrtmgp.git
    GIT_TAG origin/develop
    GIT_SHALLOW TRUE
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/rte-rrtmgp
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE TRUE
    BUILD_COMMAND eval ${BUILD_COMMAND_STRING}
    INSTALL_COMMAND ""
)

# Compile C bindings
find_package(pybind11 REQUIRED)

set(TARGET_NAME pyrte_rrtmgp)
set(SOURCES ${CMAKE_SOURCE_DIR}/pybind_interface.cpp)

pybind11_add_module(${TARGET_NAME} ${SOURCES})

add_dependencies(${TARGET_NAME} rte-rrtmgp)

target_include_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_CURRENT_BINARY_DIR}/rte-rrtmgp/rte-kernels/api/
    ${CMAKE_CURRENT_BINARY_DIR}/rte-rrtmgp/rrtmgp-kernels/api/
)

target_link_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_CURRENT_BINARY_DIR}/rte-rrtmgp/build
)

target_link_libraries(${TARGET_NAME} PUBLIC 
    rrtmgp
    rte
)

target_compile_definitions(${TARGET_NAME} PRIVATE 
    VERSION_INFO=${VERSION_INFO}
    DBL_EPSILON=5.8e-2
    DCMAKE_LIBRARY_OUTPUT_DIRECTORY=pyrte_rrtmgp
)

if (${LINUX})
    target_link_libraries(${TARGET_NAME} PUBLIC gfortran)
endif()

# The install directory is the output (wheel) directory
install(TARGETS pyrte_rrtmgp DESTINATION pyrte_rrtmgp)
