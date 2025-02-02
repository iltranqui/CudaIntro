# CUDA Personal Notes

## 02/02/2025

Seen the Conv2D simple implentation with Inference and Backpropagation

Understanding the difference between __shared__ and __global__ memory in CUDA

### **CUDA Variable Storage Qualifiers Summary**  

| Qualifier      | Location       | Scope         | Cached?    | Access Speed | Usage Example |
|---------------|---------------|--------------|------------|--------------|--------------|
| `__shared__`  | **L1 (SMEM)**  | **Block**    | **Yes (L1)** | **Fast** (On-chip) | Shared among threads in a block |
| `__constant__` | **Global (ROM)** | **All threads** | **Yes (L1, L2)** | **Fast (cached L1)** | Read-only, optimized for all threads reading same data |
| `__device__`  | **Global DRAM** | **All threads** | **Yes (L2)** | **Slower (Global memory)** | Persistent across kernel calls |

---  

### ** `__shared__` (L1 Cached, Block Scope)**  
```cpp
__global__ void kernel(float* data) {
    __shared__ float shared_var[256];  // Cached in L1, fast
    int idx = threadIdx.x;
    shared_var[idx] = data[idx] * 2.0f;
    __syncthreads();
    data[idx] = shared_var[idx];
}
```

### ** `__constant__` (L1 Cached, Read-Only)**  
```cpp
__constant__ float constVar[256];  // Fast access for all threads

__global__ void kernel() {
    int idx = threadIdx.x;
    float value = constVar[idx];  // L1 cached read-only
}
```

### ** `__device__` (Global Memory, L2 Cached)**  
```cpp
__device__ float devVar[256];  // Persistent across kernel launches

__global__ void kernel() {
    int idx = threadIdx.x;
    devVar[idx] = idx * 2.0f;  // Slower (L2 cached global memory)
}
```

---

###  Best Practices:**
- Use `__shared__` for **temporary per-block data** (fastest).
- Use `__constant__` for **read-only global data** (optimized caching).
- Use `__device__` for **global persistent data** (but slower).


