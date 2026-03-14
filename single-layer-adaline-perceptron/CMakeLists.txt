cmake_minimum_required(VERSION 3.10)

project(single-layer-adaline-perceptron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(single-layer-adaline-perceptron)

target_sources(single-layer-adaline-perceptron PRIVATE
    main.cpp
    ../adaline-perceptron/AdalinePerceptron.cpp
    ../helpers/SingleLayerDatasetConverter.cpp
)

target_include_directories(single-layer-adaline-perceptron PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# To build :
# cd build
# cmake ..
# cmake --build .