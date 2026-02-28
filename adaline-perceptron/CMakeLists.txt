cmake_minimum_required(VERSION 3.10)

project(adaline-perceptron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(adaline-perceptron)

target_sources(adaline-perceptron PRIVATE
    main.cpp
    AdalinePerceptron.cpp
)

target_include_directories(adaline-perceptron PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# To build :
# cd build
# cmake ..
# cmake --build .