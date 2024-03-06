cmake_minimum_required(3.20)
project(FORTRAN_C)

add_executable(c_test main.cpp)

#target_link_directories(c_test PUBLIC )
target_link_libraries(c_test PUBLIC test)