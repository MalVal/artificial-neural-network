cmake_minimum_required(VERSION 3.10)

project(single-layer-gradient-perceptron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(single-layer-gradient-perceptron)

target_sources(single-layer-gradient-perceptron PRIVATE
    main.cpp
    ../gradient-perceptron/GradientPerceptron.cpp
    ../helpers/SingleLayerDatasetConverter.cpp
)

target_include_directories(single-layer-gradient-perceptron PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# To build :
# cd build
# cmake ..
# cmake --build .