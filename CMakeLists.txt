cmake_minimum_required (VERSION 3.18)
project(FORTRAN_C)

set(HEADERS fortran_interface.h)
set(SOURCES pybind_interface.cpp)

set(TARGET_NAME rttpy)

add_subdirectory(pybind11)

pybind11_add_module(${TARGET_NAME} ${SOURCES} ${HEADERS})

target_compile_definitions(${TARGET_NAME} PRIVATE VERSION_INFO=${VERSION_INFO})

# This is a temporary hack while fortran is built manually so we can find libtest.so
target_link_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_BINARY_DIR} 
    ${CMAKE_BINARY_DIR}/..
)
target_link_libraries(${TARGET_NAME} PUBLIC test)