cmake_minimum_required(VERSION 3.18...3.26)

if (NOT UNIX)
    set(FC "$ENV{CONDA_PREFIX}/Library/bin/x86_64-w64-mingw32-gfortran.exe")
    string(REPLACE "\\" "/" FC "${FC}")
    set(CMAKE_Fortran_COMPILER "${FC}")
endif()

set(rte-rrtmgp_DIR "ENV{CONDA_PREFIX}/lib/rte-rrtmgp/cmake/")

enable_language(Fortran)

project(rte-rrtmgp)

# Add a custom target that always runs
add_custom_target(compile-rte-rrtmgp ALL)

# Determine platform
set(LINUX FALSE)
set(APPLE_ARM FALSE)

if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(APPLE_ARM TRUE)
    endif()
endif()

find_package(rte-rrtmgp REQUIRED)

# Compile Python bindings with pybind11
find_package(pybind11 REQUIRED)

set(TARGET_NAME pyrte_rrtmgp)
set(OUTPUT_DIRECTORY pyrte_rrtmgp)
set(SOURCES ${CMAKE_SOURCE_DIR}/pybind_interface.cpp)

pybind11_add_module(${TARGET_NAME} ${SOURCES})

# TODO: This is outdated, fix it
if(APPLE)
    # On macOS, explicitly link against gfortran runtime
    if(APPLE_ARM)
        target_link_directories(${TARGET_NAME} PUBLIC /opt/homebrew/lib/gcc/current)
    else()
        target_link_directories(${TARGET_NAME} PUBLIC /usr/local/lib/gcc/current)
    endif()
endif()

target_include_directories(${TARGET_NAME} PUBLIC
    $ENV{CONDA_PREFIX}/include
)

target_link_libraries(${TARGET_NAME} PUBLIC
    rte-rrtmgp::rte
    rte-rrtmgp::rrtmgp
    gfortran
)

target_compile_definitions(${TARGET_NAME} PRIVATE
    VERSION_INFO=${VERSION_INFO}
    DBL_EPSILON=5.8e-2
    DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY}
)

# The install directory is the output (wheel) directory
install(TARGETS ${TARGET_NAME} DESTINATION ${OUTPUT_DIRECTORY})
