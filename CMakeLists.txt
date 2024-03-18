cmake_minimum_required(VERSION 3.18...3.26)
project(rte-rrtmgp)

# Add a custom target that always runs
add_custom_target(compile-rte-rrtmgp ALL)

set(LINUX FALSE)
if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# Custom command for the compile-rte-rrtmgp target
add_custom_command(
    TARGET compile-rte-rrtmgp
    COMMAND cd ${CMAKE_SOURCE_DIR} && 
            ./build_rte_rrtmgp.sh &&
            mv rte-rrtmgp/build/*.a ${CMAKE_CURRENT_BINARY_DIR} && 
            mv rte-rrtmgp/build/*.o ${CMAKE_CURRENT_BINARY_DIR} && 
            mv rte-rrtmgp/build/*.mod ${CMAKE_CURRENT_BINARY_DIR}
)

# Compile C bindings
set(HEADERS ${CMAKE_SOURCE_DIR}/rte-rrtmgp/rte-kernels/api/rte_kernels.h)
set(SOURCES ${CMAKE_SOURCE_DIR}/pybind_interface.cpp)
set(TARGET_NAME pyrte)

find_package(pybind11 REQUIRED)

pybind11_add_module(${TARGET_NAME} ${SOURCES} ${HEADERS})

target_compile_definitions(${TARGET_NAME} PRIVATE 
    VERSION_INFO=${VERSION_INFO}
    DBL_EPSILON=5.8e-2
    DCMAKE_LIBRARY_OUTPUT_DIRECTORY=pyrte
)

add_dependencies(pyrte compile-rte-rrtmgp)

target_include_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_SOURCE_DIR}/rte-rrtmgp/rte-kernels/api/
)

target_link_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${TARGET_NAME} PUBLIC 
    rrtmgp
    rte
)

if ($LINUX)
    target_link_libraries(${TARGET_NAME} PUBLIC gfortran)
endif()

# The install directory is the output (wheel) directory
install(TARGETS pyrte DESTINATION pyrte)
