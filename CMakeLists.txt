cmake_minimum_required (VERSION 3.18)
project(FORTRAN_C)

if (NOT DEFINED EXAMPLE_VERSION_INFO)
	set(EXAMPLE_VERSION_INFO "0.0.1")
endif()

set(HEADERS fortran_interface.h)
set(SOURCES pybind_interface.cpp)

set(TARGET_NAME rttpy)

add_subdirectory(pybind11)

pybind11_add_module(${TARGET_NAME} ${SOURCES} ${HEADERS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(${TARGET_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

target_link_directories(${TARGET_NAME} PUBLIC ${CMAKE_BINARY_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC test)