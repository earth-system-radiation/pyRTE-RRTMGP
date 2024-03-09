cmake_minimum_required (VERSION 3.18)
project(FORTRAN_C)

#find_package(Python ${PYTHON_VERSION} COMPONENTS Interpreter Development.Module REQUIRED)
#find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT DEFINED EXAMPLE_VERSION_INFO)
	set(EXAMPLE_VERSION_INFO "0.0.1")
endif()

set(HEADERS
    fortran_interface.h
)

set(SOURCES
    pybind_interface.cpp
)

set(TARGET_NAME rttpy)


add_subdirectory(pybind11)

pybind11_add_module(${TARGET_NAME} ${SOURCES} ${HEADERS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(${TARGET_NAME}
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

#add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})

# target_include_directories(${TARGET_NAME} PUBLIC 
#     pybind11/include
#     ${Python_INCLUDE_DIRS}
# )
target_link_directories(${TARGET_NAME} PUBLIC ${CMAKE_BINARY_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC test ${Python_LIBRARIES})
# target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)