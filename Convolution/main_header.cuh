

int conv2d_deform();
int conv2d_inf();

int launchConvolution2DBasicKernel(int width = 5, int height = 5, int r = 1,
    float* h_N = nullptr, float* h_F = nullptr, float* h_P = nullptr,
    dim3 blockDim = dim3(16, 16), dim3 gridDim = dim3(0, 0));