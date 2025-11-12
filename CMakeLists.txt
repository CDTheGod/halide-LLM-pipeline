cmake_minimum_required(VERSION 3.16)
project(lesson_01)

find_package(Halide CONFIG REQUIRED)

add_executable(lesson_01 lesson_01.cpp)
target_link_libraries(lesson_01 PRIVATE Halide::Halide)
