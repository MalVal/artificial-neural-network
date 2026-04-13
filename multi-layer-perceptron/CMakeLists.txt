cmake_minimum_required(VERSION 3.10)

project(multi-layer-perceptron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# LIBRAIRIE COMMUNE
# =========================
add_library(mlp_core STATIC
    HidenLayer.cpp
    HiddenLayerDefinition.cpp
    HidenNeuron.cpp
    MultiLayerPerceptron.cpp
    ../helpers/ActivateFunctions.cpp
)

target_include_directories(mlp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# =========================
# EXEC 1
# =========================
add_executable(multi-layer-perceptron
    main.cpp
)

target_link_libraries(multi-layer-perceptron PRIVATE mlp_core)

# =========================
# EXEC 2
# =========================
add_executable(multi-layer-perceptron-sign
    main-sign.cpp
)

target_link_libraries(multi-layer-perceptron-sign PRIVATE mlp_core)