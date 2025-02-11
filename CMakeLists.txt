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

ExternalProject_Add(
    rte-rrtmgp
    # For development only
    GIT_REPOSITORY https://github.com/makepath-alex/rte-rrtmgp.git
    GIT_TAG use-cbool-dev
    GIT_SHALLOW TRUE
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/rte-rrtmgp
    CMAKE_ARGS
        -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
        -DCMAKE_Fortran_FLAGS=${FFLAGS}
        -DRTE_ENABLE_SP=OFF
        -DKERNEL_MODE=default
        -DBUILD_TESTING=OFF
        -DFAILURE_THRESHOLD=7.e-4
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/rte-rrtmgp/
    BUILD_IN_SOURCE TRUE
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target all --parallel
    INSTALL_COMMAND ${CMAKE_COMMAND} --install .
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
    ${CMAKE_CURRENT_BINARY_DIR}/rte-rrtmgp/lib/
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

# Add these checks after the initial Linux check
set(APPLE_ARM FALSE)
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(APPLE_ARM TRUE)
    endif()
endif()

if (${LINUX})
    target_link_libraries(${TARGET_NAME} PUBLIC gfortran)
elseif(APPLE)
    # On macOS, explicitly link against gfortran runtime
    if(APPLE_ARM)
        target_link_directories(${TARGET_NAME} PUBLIC /opt/homebrew/lib/gcc/current)
    else()
        target_link_directories(${TARGET_NAME} PUBLIC /usr/local/lib/gcc/current)
    endif()
    target_link_libraries(${TARGET_NAME} PUBLIC gfortran)
endif()

# The install directory is the output (wheel) directory
install(TARGETS pyrte_rrtmgp DESTINATION pyrte_rrtmgp)
