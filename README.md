# CUDA Personal Notes

Personal Repo to Study CUDA and GPU programming, specifically learn how to implement many kernels from scratch.

## 02/02/2025

Seen the Conv2D simple implentation with Inference and Backpropagation

Understanding the difference between __shared__ and __global__ memory in CUDA

# Roadmap

- [x] Understand CUDA memory hierarchy
- [ ] Implement a simple CUDA kernel
- [ ] Optimize CUDA kernel for memory access
- [ ] Profile CUDA kernel performance
- [ ] Compare CPU vs GPU performance
- [x] Conv_1D CPU vs GPU performance
    - [ ] Easier to perform experiments (parse agrs for kernel size, stride, padding)
- [ ] Conv_2D CPU vs GPU performance
    - [ ] Easier to perform experiments (parse agrs for kernel size, stride, padding)
- [ ] Conv_3D CPU vs GPU performance
    - [ ] Easier to perform experiments (parse agrs for kernel size, stride, padding)
- [ ] Complete MNIST Example in CUDA/C++, this can ve an adventure

### 23 Feb 2025

- Added a Conv1D implementation in CUDA, with GPU vs CPU performance comparison

### 25 Feb 2025

- Added a training loop for Conv2D deformation in CUDA, with lots of comments to understand it

### 04 Mar 2025

- Started a C++ and CUDA project, guide from [here](https://medium.com/@aviatorx/c-and-cuda-project-visual-studio-d07c6ad771e3)
- Next Step: Import OpenCV and open Image with OpenCV. 