cmake_minimum_required(VERSION 3.10)

project(gradient-perceptron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(gradient-perceptron main.cpp)

# To build :
# cd build
# cmake ..
# cmake --build .