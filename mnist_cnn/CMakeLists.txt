cmake_minimum_required(VERSION 3.18)
project(mnist_cnn LANGUAGES CXX CUDA)

# Add the executable
add_executable(mnist_cnn 
    ${CMAKE_SOURCE_DIR}/mnist_cnn.cu
    ${CMAKE_SOURCE_DIR}/mnist_kernels.cu
)

# Set include directories
target_include_directories(mnist_cnn PRIVATE ${CMAKE_SOURCE_DIR})

# Link OpenCV libraries
target_link_libraries(mnist_cnn PRIVATE ${OpenCV_LIBS})

# Set CUDA specific flags
set_target_properties(mnist_cnn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CUDA specific compile options
target_compile_options(mnist_cnn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=compute_89 -code=sm_89>
)
