# MNIST Neural Network Projects

This repository contains two CUDA-based neural network implementations for the MNIST dataset:

1. `mnist_nn` - A simple neural network implementation
2. `mnist_cnn` - A convolutional neural network implementation

## Prerequisites

- CMake 3.18 or higher
- CUDA Toolkit 11.0 or higher
- Visual Studio 2019 or higher
- OpenCV 4.10 (installed via vcpkg)

## Building the Projects

### Building Both Projects

To build both projects at once, run:

```
build.bat
```

This will:
1. Create a `build` directory
2. Configure CMake
3. Build both projects
4. Copy the executables to the root directory

### Building Individual Projects

To build only the `mnist_nn` project:

```
build_mnist_nn.bat
```

To build only the `mnist_cnn` project:

```
build_mnist_cnn.bat
```

## Project Structure

- `mnist_nn.cu` and `mnist_nn.h` - Simple neural network implementation
- `mnist_cnn.cu`, `mnist_cnn.h`, and `mnist_kernels.cu` - Convolutional neural network implementation
- `mnist_nn/CMakeLists.txt` - CMake configuration for the mnist_nn project
- `mnist_cnn/CMakeLists.txt` - CMake configuration for the mnist_cnn project

## Running the Projects

After building, you can run the executables:

```
mnist_nn.exe
```

or

```
mnist_cnn.exe
```

## Neural Network Architecture

### mnist_nn
- Simple feedforward neural network with multiple hidden layers

### mnist_cnn
- Convolutional neural network with 5 convolutional layers
- Multiple pooling layers
- Multiple fully connected layers before the softmax output
