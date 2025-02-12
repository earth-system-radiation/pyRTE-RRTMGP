cmake_minimum_required(VERSION 3.18...3.26)

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

# Compile Python bindings with pybind11
find_package(pybind11 REQUIRED)

set(TARGET_NAME pyrte_rrtmgp)
set(OUTPUT_DIRECTORY pyrte_rrtmgp)
set(SOURCES ${CMAKE_SOURCE_DIR}/pybind_interface.cpp)

pybind11_add_module(${TARGET_NAME} ${SOURCES})

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

target_link_directories(${TARGET_NAME} PUBLIC
    $ENV{CONDA_PREFIX}/lib
)

target_link_libraries(${TARGET_NAME} PUBLIC
    $ENV{CONDA_PREFIX}/lib/librte.a
    $ENV{CONDA_PREFIX}/lib/librrtmgp.a
    gfortran
)

target_compile_definitions(${TARGET_NAME} PRIVATE
    VERSION_INFO=${VERSION_INFO}
    DBL_EPSILON=5.8e-2
    DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY}
)

# The install directory is the output (wheel) directory
install(TARGETS ${TARGET_NAME} DESTINATION ${OUTPUT_DIRECTORY})

file(GLOB DLL_FILES
    $ENV{CONDA_PREFIX}/Library/bin/libgcc_s_seh-1.dll
    $ENV{CONDA_PREFIX}/Library/bin/libgfortran-5.dll
    $ENV{CONDA_PREFIX}/Library/bin/libstdc++-6.dll
)

install(FILES ${DLL_FILES} DESTINATION ${OUTPUT_DIRECTORY})
