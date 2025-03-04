#include <iostream>
#include "kernel.h"

int main() {
    std::cout << "Calling CUDA Kernel from C++" << std::endl;
    launchKernel(); // Call CUDA function
    return 0;
}
