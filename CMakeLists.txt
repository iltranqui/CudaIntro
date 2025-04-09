cmake_minimum_required(VERSION 3.18)
project(MNIST_Projects LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures for RTX 4060
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add subdirectories for each project
add_subdirectory(mnist_nn)
add_subdirectory(mnist_cnn)
