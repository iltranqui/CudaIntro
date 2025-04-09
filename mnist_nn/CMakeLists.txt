cmake_minimum_required(VERSION 3.18)
project(mnist_nn LANGUAGES CXX CUDA)

# Add the executable
add_executable(mnist_nn 
    ${CMAKE_SOURCE_DIR}/mnist_nn.cu
)

# Set include directories
target_include_directories(mnist_nn PRIVATE ${CMAKE_SOURCE_DIR})

# Link OpenCV libraries
target_link_libraries(mnist_nn PRIVATE ${OpenCV_LIBS})

# Set CUDA specific flags
set_target_properties(mnist_nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CUDA specific compile options
target_compile_options(mnist_nn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=compute_89 -code=sm_89>
)
