cmake_minimum_required(VERSION 3.10)

project(multi-layer-perceptron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(multi-layer-perceptron)

target_sources(multi-layer-perceptron PRIVATE
    main.cpp
    HidenLayer.cpp
    HidenNeuron.cpp
    MultiLayerPerceptron.cpp
    ../helpers/ActivateFunctions.cpp
)

target_include_directories(multi-layer-perceptron PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# To build :
# cd build
# cmake ..
# cmake --build .