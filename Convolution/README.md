# Convolution

IN this folder i am implementing only CONVOLUTION related operations in CUDA.

- [ ] TODO, using #define to add exeution or not of different completed implementaion of kernels
- [ ] TODO, add more kernels for different operations

## Convolution 1D

Made Layer for a 1D Max Pooling Operation with the following code: 

Using **int threadsPerBlock = 256** and **blocksPerGrid = (H_out + threadsPerBlock - 1) / threadsPerBlock;** 

```cpp
// CUDA kernel performing 1D max pooling
__global__ void maxpool1d_kernel(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H, int H_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H_out) {
        // Start with negative infinity for the max value
        float max_val = -INFINITY;
        for (int m = 0; m < kernel_size; m++) {
            // Calculate the input index for this kernel element
            int index = stride * i + dilation * m - padding;
			// If the index is out-of-bound, treat the value as -infinity -> Zero padding does this specifically
            float value = (index < 0 || index >= H) ? -INFINITY : input[index];
            max_val = fmaxf(max_val, value);
        }
        output[i] = max_val;
    }
}
```



## Classical Basic Kernel

```cpp
__global__ void convolution_2d_basic_kernel(float* N, float* F, float* P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += N[inRow * width + inCol] * F[fRow * (2 * r + 1) + fCol];   // 2D convolution main operation done for each pixel of the output image
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}
```

## Constant Memory Kernel

The filter kernel of the convolution is inserted into the shared memory as a constant parameter, into L1 or l2 memory, making the memory access of this parameter easier to the Cpp. This version is slightl faster compaed to the basic kernel version. 

```cpp
__global__ void convolution_2d_const_mem_kernel(float* N, float* P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += N[inRow * width + inCol] * F[fRow][fCol];   // 2D convolution main operation done for each pixel of the output image
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}
```

## Constant Memory Tiles Kernel

Using tiled convolution in CUDA is more efficient than standard convolution because it improves memory access patterns, reduces global memory transactions, and enhances parallel efficiency

- Efficient Shared Memory Usage
- Reducing Global Memory Access
- Better Memory Coalescing
- Increased Computational Density/Efficiency
- Better Scalability for Large Kernels
- Reduced Memory Bandwidth Bottleneck
  - CONS
  - Syncronization Overhead -> more __syncthreads() calls are needed, slowing down the kernel
  - TODO: more exaplanation

```cpp
__global__ void convolution_tiled_2d_const_mem_kernel(float* N, float* P, int width, int height) {

    int Col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int Row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    // loading input tile into shared memory
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (Row >= 0 && Row < height && Col >= 0 && Col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[Row * width + Col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();  // for synchronization -> for some reason is gives me undefined
    // Calculate output element

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // turn off threads that are outside the output tile
    if (Col >= 0 && Col < width && Row >= 0 && Row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Pvalue += N_s[threadIdx.y + fRow][threadIdx.x + fCol] * F_c[fRow][fCol];
                }
            }
            P[Row * width + Col] = Pvalue;
        }
    }
}
```

## EXTERN "C" void  ?? Why use it ?

In C++, function names are mangled by the compiler to include information about their parameters and namespaces. This allows function overloading but makes it difficult to link functions with other languages, such as C or assembly. extern "C" is mainly used to ensure that functions can be called from C and to prevent C++ name mangling.

- Use it when:
  - You need to call C++ functions from C.
  - You are linking with a library that expects C linkage.
  - You are using CUDA and want to ensure interoperability.
  - You want to create a shared library (.so or .dll) that will be used by both C and C++ programs.
- Do not use it when:
  - Your entire program is in C++ and does not need C compatibility.
  - You need function overloading (which extern "C" disables).