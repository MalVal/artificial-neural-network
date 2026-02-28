cmake_minimum_required(VERSION 3.10)

project(simple-perceptron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(simple-perceptron)

target_sources(simple-perceptron PRIVATE
    main.cpp
    SimplePerceptron.cpp
)

target_include_directories(simple-perceptron PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# To build :
# cd build
# cmake ..
# cmake --build .