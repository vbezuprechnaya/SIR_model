cmake_minimum_required(VERSION 3.22)
project(SIR)

set(CMAKE_CXX_STANDARD 14)

add_executable(SIR main.cpp sir.cpp sir.hpp Matrix.h Matrix.cpp GaussNewton.cpp GaussNewton.h)
