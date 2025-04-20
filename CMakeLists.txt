cmake_minimum_required(VERSION 3.18)
project(CudaProject LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures for RTX 4060
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find required packages
find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)

# Find optional packages
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found. Including OpenCV support.")
    add_definitions(-DUSE_OPENCV)
else()
    message(STATUS "OpenCV not found. Building without OpenCV support.")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)

# Add OpenCV include directories if found
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Add Convolution library
add_library(Convolution STATIC
    Convolution/conv2_deform.cu
    Convolution/main.cu
    Convolution/average_pool_1d.cu
    Convolution/max_pooling_1d.cu
)

target_include_directories(Convolution PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Convolution
)

# Add ConvolutionGTest executable
add_executable(ConvolutionGTest
    ConvolutionGTest/gtest_main.cu
    ConvolutionGTest/deform_conv2d_gtest.cu
    ConvolutionGTest/average_pooling_1D_gtest.cu
    ConvolutionGTest/max_pooling_1D_gtest.cu
)

target_link_libraries(ConvolutionGTest
    Convolution
    ${GTEST_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Add pthread on Unix-like systems
if(UNIX)
    target_link_libraries(ConvolutionGTest pthread)
endif()

# Commented out due to compilation errors
# add_executable(CudaMain
#     main.cpp
#     kernel.cu
#     add_vector.cu
# )
#
# target_link_libraries(CudaMain
#     ${CUDA_LIBRARIES}
# )

# Add MNIST neural network executables if OpenCV is found
if(OpenCV_FOUND)
    add_subdirectory(mnist_nn)
    add_subdirectory(mnist_cnn)
endif()

# Set CUDA specific flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Linux/GCC/Clang specific flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
elseif(MSVC)
    # Windows/MSVC specific flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Install targets
install(TARGETS ConvolutionGTest
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Enable testing
enable_testing()
add_test(NAME ConvolutionTests COMMAND ConvolutionGTest)
