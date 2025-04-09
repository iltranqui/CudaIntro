#include "mnist_cnn.h"
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <atomic>
#include <condition_variable>

// CUDA Kernels are now defined in mnist_kernels.cu

// MNIST Dataset Implementation
MNISTDataset::MNISTDataset(const std::string& csv_file, const std::string& base_path) : base_path(base_path) {
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << csv_file << std::endl;
        return;
    }

    std::string line;
    // Skip header if exists
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;

        std::getline(ss, item, ',');
        std::string image_path = item;

        std::getline(ss, item, ',');
        int label = std::stoi(item);

        image_paths.push_back(image_path);
        labels.push_back(label);
    }

    std::cout << "Loaded " << image_paths.size() << " images from " << csv_file << std::endl;
}

void MNISTDataset::getBatch(int batch_idx, int batch_size, float* images, int* labels) {
    int start_idx = batch_idx * batch_size;
    int end_idx = std::min(start_idx + batch_size, static_cast<int>(image_paths.size()));
    int actual_batch_size = end_idx - start_idx;

    int images_loaded = 0;
    for (int i = 0; i < actual_batch_size; i++) {
        int idx = start_idx + i;
        std::string img_path = base_path + "/" + image_paths[idx];

        // Load image using OpenCV with explicit flags
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

        // If image is empty, try to load it with different flags
        if (img.empty()) {
            // Try with IMREAD_UNCHANGED
            img = cv::imread(img_path, cv::IMREAD_UNCHANGED);

            if (img.empty()) {
                std::cerr << "Error: Could not read image " << img_path << std::endl;
                // Try to check if the file exists
                std::ifstream f(img_path.c_str());
                if (f.good()) {
                    std::cerr << "File exists but could not be read as an image" << std::endl;
                    // Try to get file size
                    f.seekg(0, std::ios::end);
                    std::streampos fileSize = f.tellg();
                    std::cerr << "File size: " << fileSize << " bytes" << std::endl;

                    // Try to create a blank image instead
                    std::cout << "Creating a blank image instead" << std::endl;
                    img = cv::Mat::zeros(28, 28, CV_8UC1);
                } else {
                    std::cerr << "File does not exist" << std::endl;
                    // Try to list parent directory
                    std::string parent_dir = img_path.substr(0, img_path.find_last_of("/\\"));
                    std::cerr << "Parent directory: " << parent_dir << std::endl;

                    // Create a blank image
                    std::cout << "Creating a blank image instead" << std::endl;
                    img = cv::Mat::zeros(28, 28, CV_8UC1);
                }
            } else if (img.channels() > 1) {
                // Convert to grayscale if it's a color image
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            }
        }

        images_loaded++;

        // Check if the image has any non-zero pixels
        bool has_non_zero = false;
        int non_zero_count = 0;
        int total_pixels = img.rows * img.cols;

        // Normalize pixel values to [0, 1]
        for (int r = 0; r < img.rows; r++) {
            for (int c = 0; c < img.cols; c++) {
                uchar pixel = img.at<uchar>(r, c);
                if (pixel > 0) {
                    has_non_zero = true;
                    non_zero_count++;
                }

                float pixel_value = pixel / 255.0f;
                images[i * (INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH) + r * INPUT_WIDTH + c] = pixel_value;
            }
        }

        labels[i] = this->labels[idx];
    }

    // Pad remaining batch with zeros if needed
    if (actual_batch_size < batch_size) {
        int padding_size = (batch_size - actual_batch_size) * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
        memset(&images[actual_batch_size * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH], 0, padding_size * sizeof(float));

        for (int i = actual_batch_size; i < batch_size; i++) {
            labels[i] = 0; // Default label for padding
        }
    }
}

// CNN Implementation
ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork() {
    // Calculate output dimensions
    conv1_output_height = INPUT_HEIGHT + 2 * CONV1_PADDING - CONV1_KERNEL_SIZE + 1;
    conv1_output_width = INPUT_WIDTH + 2 * CONV1_PADDING - CONV1_KERNEL_SIZE + 1;

    pool1_output_height = conv1_output_height / POOL1_SIZE;
    pool1_output_width = conv1_output_width / POOL1_SIZE;

    conv2_output_height = pool1_output_height + 2 * CONV2_PADDING - CONV2_KERNEL_SIZE + 1;
    conv2_output_width = pool1_output_width + 2 * CONV2_PADDING - CONV2_KERNEL_SIZE + 1;

    pool2_output_height = conv2_output_height / POOL2_SIZE;
    pool2_output_width = conv2_output_width / POOL2_SIZE;

    // New conv3 layer dimensions
    conv3_output_height = pool2_output_height + 2 * CONV3_PADDING - CONV3_KERNEL_SIZE + 1;
    conv3_output_width = pool2_output_width + 2 * CONV3_PADDING - CONV3_KERNEL_SIZE + 1;

    pool3_output_height = conv3_output_height / POOL3_SIZE;
    pool3_output_width = conv3_output_width / POOL3_SIZE;

    // New conv4 layer dimensions
    conv4_output_height = pool3_output_height + 2 * CONV4_PADDING - CONV4_KERNEL_SIZE + 1;
    conv4_output_width = pool3_output_width + 2 * CONV4_PADDING - CONV4_KERNEL_SIZE + 1;

    pool4_output_height = conv4_output_height / POOL4_SIZE;
    pool4_output_width = conv4_output_width / POOL4_SIZE;

    // New conv5 layer dimensions
    conv5_output_height = pool4_output_height + 2 * CONV5_PADDING - CONV5_KERNEL_SIZE + 1;
    conv5_output_width = pool4_output_width + 2 * CONV5_PADDING - CONV5_KERNEL_SIZE + 1;

    // Update flattened size to use the output of conv5 layer
    flattened_size = CONV5_FILTERS * conv5_output_height * conv5_output_width;

    initializeWeights();
    allocateMemory();

    // Print network architecture
    std::cout << "\n===== CNN Architecture =====" << std::endl;
    std::cout << "Input: " << INPUT_CHANNELS << "x" << INPUT_HEIGHT << "x" << INPUT_WIDTH << std::endl;
    std::cout << "Conv1: " << CONV1_FILTERS << " filters, " << CONV1_KERNEL_SIZE << "x" << CONV1_KERNEL_SIZE << " kernel" << std::endl;
    std::cout << "Pool1: " << POOL1_SIZE << "x" << POOL1_SIZE << " max pooling" << std::endl;
    std::cout << "Conv2: " << CONV2_FILTERS << " filters, " << CONV2_KERNEL_SIZE << "x" << CONV2_KERNEL_SIZE << " kernel" << std::endl;
    std::cout << "Pool2: " << POOL2_SIZE << "x" << POOL2_SIZE << " max pooling" << std::endl;
    std::cout << "Conv3: " << CONV3_FILTERS << " filters, " << CONV3_KERNEL_SIZE << "x" << CONV3_KERNEL_SIZE << " kernel" << std::endl;
    std::cout << "Pool3: " << POOL3_SIZE << "x" << POOL3_SIZE << " max pooling" << std::endl;
    std::cout << "Conv4: " << CONV4_FILTERS << " filters, " << CONV4_KERNEL_SIZE << "x" << CONV4_KERNEL_SIZE << " kernel" << std::endl;
    std::cout << "Pool4: " << POOL4_SIZE << "x" << POOL4_SIZE << " max pooling" << std::endl;
    std::cout << "Conv5: " << CONV5_FILTERS << " filters, " << CONV5_KERNEL_SIZE << "x" << CONV5_KERNEL_SIZE << " kernel" << std::endl;
    std::cout << "FC1: " << FC1_SIZE << " neurons" << std::endl;
    std::cout << "FC2: " << FC2_SIZE << " neurons" << std::endl;
    std::cout << "FC3: " << FC3_SIZE << " neurons" << std::endl;
    std::cout << "FC4: " << FC4_SIZE << " neurons" << std::endl;
    std::cout << "FC5: " << FC5_SIZE << " neurons" << std::endl;
    std::cout << "Output: " << OUTPUT_SIZE << " neurons" << std::endl;
    std::cout << "Flattened size: " << flattened_size << std::endl;
    std::cout << "===========================\n" << std::endl;
}

ConvolutionalNeuralNetwork::~ConvolutionalNeuralNetwork() {
    freeMemory();
}

void ConvolutionalNeuralNetwork::initializeWeights() {
    // Initialize weights with Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());

    // Conv1 weights
    float conv1_scale = sqrt(2.0f / (INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE));
    std::normal_distribution<float> conv1_dist(0.0f, conv1_scale);
    h_conv1_weights.resize(CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE);
    for (auto& w : h_conv1_weights) {
        w = conv1_dist(gen);
    }

    // Conv1 bias
    h_conv1_bias.resize(CONV1_FILTERS, 0.01f);

    // Conv2 weights
    float conv2_scale = sqrt(2.0f / (CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE));
    std::normal_distribution<float> conv2_dist(0.0f, conv2_scale);
    h_conv2_weights.resize(CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE);
    for (auto& w : h_conv2_weights) {
        w = conv2_dist(gen);
    }

    // Conv2 bias
    h_conv2_bias.resize(CONV2_FILTERS, 0.01f);

    // Conv3 weights
    float conv3_scale = sqrt(2.0f / (CONV2_FILTERS * CONV3_KERNEL_SIZE * CONV3_KERNEL_SIZE));
    std::normal_distribution<float> conv3_dist(0.0f, conv3_scale);
    h_conv3_weights.resize(CONV3_FILTERS * CONV2_FILTERS * CONV3_KERNEL_SIZE * CONV3_KERNEL_SIZE);
    for (auto& w : h_conv3_weights) {
        w = conv3_dist(gen);
    }

    // Conv3 bias
    h_conv3_bias.resize(CONV3_FILTERS, 0.01f);

    // Conv4 weights
    float conv4_scale = sqrt(2.0f / (CONV3_FILTERS * CONV4_KERNEL_SIZE * CONV4_KERNEL_SIZE));
    std::normal_distribution<float> conv4_dist(0.0f, conv4_scale);
    h_conv4_weights.resize(CONV4_FILTERS * CONV3_FILTERS * CONV4_KERNEL_SIZE * CONV4_KERNEL_SIZE);
    for (auto& w : h_conv4_weights) {
        w = conv4_dist(gen);
    }

    // Conv4 bias
    h_conv4_bias.resize(CONV4_FILTERS, 0.01f);

    // FC1 weights
    float fc1_scale = sqrt(2.0f / flattened_size);
    std::normal_distribution<float> fc1_dist(0.0f, fc1_scale);
    h_fc1_weights.resize(flattened_size * FC1_SIZE);
    for (auto& w : h_fc1_weights) {
        w = fc1_dist(gen);
    }

    // FC1 bias
    h_fc1_bias.resize(FC1_SIZE, 0.01f);

    // FC2 weights
    float fc2_scale = sqrt(2.0f / FC1_SIZE);
    std::normal_distribution<float> fc2_dist(0.0f, fc2_scale);
    h_fc2_weights.resize(FC1_SIZE * FC2_SIZE);
    for (auto& w : h_fc2_weights) {
        w = fc2_dist(gen);
    }

    // FC2 bias
    h_fc2_bias.resize(FC2_SIZE, 0.01f);

    // FC3 weights
    float fc3_scale = sqrt(2.0f / FC2_SIZE);
    std::normal_distribution<float> fc3_dist(0.0f, fc3_scale);
    h_fc3_weights.resize(FC2_SIZE * FC3_SIZE);
    for (auto& w : h_fc3_weights) {
        w = fc3_dist(gen);
    }

    // FC3 bias
    h_fc3_bias.resize(FC3_SIZE, 0.01f);

    // FC4 weights
    float fc4_scale = sqrt(2.0f / FC3_SIZE);
    std::normal_distribution<float> fc4_dist(0.0f, fc4_scale);
    h_fc4_weights.resize(FC3_SIZE * FC4_SIZE);
    for (auto& w : h_fc4_weights) {
        w = fc4_dist(gen);
    }

    // FC4 bias
    h_fc4_bias.resize(FC4_SIZE, 0.01f);

    // FC5 (output) weights
    float fc5_scale = sqrt(2.0f / FC4_SIZE);
    std::normal_distribution<float> fc5_dist(0.0f, fc5_scale);
    h_fc5_weights.resize(FC4_SIZE * OUTPUT_SIZE);
    for (auto& w : h_fc5_weights) {
        w = fc5_dist(gen);
    }

    // FC5 bias
    h_fc5_bias.resize(OUTPUT_SIZE, 0.01f);

    // Conv5 weights
    float conv5_scale = sqrt(2.0f / (CONV4_FILTERS * CONV5_KERNEL_SIZE * CONV5_KERNEL_SIZE));
    std::normal_distribution<float> conv5_dist(0.0f, conv5_scale);
    h_conv5_weights.resize(CONV5_FILTERS * CONV4_FILTERS * CONV5_KERNEL_SIZE * CONV5_KERNEL_SIZE);
    for (auto& w : h_conv5_weights) {
        w = conv5_dist(gen);
    }

    // Conv5 bias
    h_conv5_bias.resize(CONV5_FILTERS, 0.01f);

    std::cout << "Weights initialized with He initialization" << std::endl;
}

void ConvolutionalNeuralNetwork::allocateMemory() {
    // Allocate device memory for weights and biases
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv1_weights, h_conv1_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv1_bias, h_conv1_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv2_weights, h_conv2_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv2_bias, h_conv2_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv3_weights, h_conv3_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv3_bias, h_conv3_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv4_weights, h_conv4_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv4_bias, h_conv4_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc1_weights, h_fc1_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc1_bias, h_fc1_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc2_weights, h_fc2_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc2_bias, h_fc2_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc3_weights, h_fc3_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc3_bias, h_fc3_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc4_weights, h_fc4_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc4_bias, h_fc4_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc5_weights, h_fc5_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc5_bias, h_fc5_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv5_weights, h_conv5_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv5_bias, h_conv5_bias.size() * sizeof(float)));

    // Copy weights and biases to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv1_weights, h_conv1_weights.data(), h_conv1_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv1_bias, h_conv1_bias.data(), h_conv1_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv2_weights, h_conv2_weights.data(), h_conv2_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv2_bias, h_conv2_bias.data(), h_conv2_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv3_weights, h_conv3_weights.data(), h_conv3_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv3_bias, h_conv3_bias.data(), h_conv3_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv4_weights, h_conv4_weights.data(), h_conv4_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv4_bias, h_conv4_bias.data(), h_conv4_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc1_weights, h_fc1_weights.data(), h_fc1_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc1_bias, h_fc1_bias.data(), h_fc1_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc2_weights, h_fc2_weights.data(), h_fc2_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc2_bias, h_fc2_bias.data(), h_fc2_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc3_weights, h_fc3_weights.data(), h_fc3_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc3_bias, h_fc3_bias.data(), h_fc3_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc4_weights, h_fc4_weights.data(), h_fc4_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc4_bias, h_fc4_bias.data(), h_fc4_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc5_weights, h_fc5_weights.data(), h_fc5_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc5_bias, h_fc5_bias.data(), h_fc5_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv5_weights, h_conv5_weights.data(), h_conv5_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv5_bias, h_conv5_bias.data(), h_conv5_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for intermediate results
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv1_output, BATCH_SIZE * CONV1_FILTERS * conv1_output_height * conv1_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool1_output, BATCH_SIZE * CONV1_FILTERS * pool1_output_height * pool1_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool1_max_indices, BATCH_SIZE * CONV1_FILTERS * pool1_output_height * pool1_output_width * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv2_output, BATCH_SIZE * CONV2_FILTERS * conv2_output_height * conv2_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool2_output, BATCH_SIZE * CONV2_FILTERS * pool2_output_height * pool2_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool2_max_indices, BATCH_SIZE * CONV2_FILTERS * pool2_output_height * pool2_output_width * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv3_output, BATCH_SIZE * CONV3_FILTERS * conv3_output_height * conv3_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool3_output, BATCH_SIZE * CONV3_FILTERS * pool3_output_height * pool3_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool3_max_indices, BATCH_SIZE * CONV3_FILTERS * pool3_output_height * pool3_output_width * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv4_output, BATCH_SIZE * CONV4_FILTERS * conv4_output_height * conv4_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool4_output, BATCH_SIZE * CONV4_FILTERS * conv4_output_height/POOL4_SIZE * conv4_output_width/POOL4_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pool4_max_indices, BATCH_SIZE * CONV4_FILTERS * conv4_output_height/POOL4_SIZE * conv4_output_width/POOL4_SIZE * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv5_output, BATCH_SIZE * CONV5_FILTERS * conv5_output_height * conv5_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc1_output, BATCH_SIZE * FC1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc2_output, BATCH_SIZE * FC2_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc3_output, BATCH_SIZE * FC3_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc4_output, BATCH_SIZE * FC4_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc5_output, BATCH_SIZE * FC5_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));

    // Allocate device memory for gradients and initialize to zero
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv1_weights, h_conv1_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv1_weights, 0, h_conv1_weights.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv1_bias, h_conv1_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv1_bias, 0, h_conv1_bias.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv2_weights, h_conv2_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv2_weights, 0, h_conv2_weights.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv2_bias, h_conv2_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv2_bias, 0, h_conv2_bias.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv3_weights, h_conv3_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv3_weights, 0, h_conv3_weights.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv3_bias, h_conv3_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv3_bias, 0, h_conv3_bias.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv4_weights, h_conv4_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv4_weights, 0, h_conv4_weights.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv4_bias, h_conv4_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv4_bias, 0, h_conv4_bias.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc1_weights, h_fc1_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc1_weights, 0, h_fc1_weights.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc1_bias, h_fc1_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc1_bias, 0, h_fc1_bias.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc2_weights, h_fc2_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc2_weights, 0, h_fc2_weights.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc2_bias, h_fc2_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc2_bias, 0, h_fc2_bias.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc3_weights, h_fc3_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc3_weights, 0, h_fc3_weights.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc3_bias, h_fc3_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc3_bias, 0, h_fc3_bias.size() * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_output, 0, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc3_output, BATCH_SIZE * FC2_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc3_output, 0, BATCH_SIZE * FC2_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc2_output, BATCH_SIZE * FC2_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc2_output, 0, BATCH_SIZE * FC2_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_fc1_output, BATCH_SIZE * FC1_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc1_output, 0, BATCH_SIZE * FC1_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_pool3_output, BATCH_SIZE * CONV3_FILTERS * pool3_output_height * pool3_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_pool3_output, 0, BATCH_SIZE * CONV3_FILTERS * pool3_output_height * pool3_output_width * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv4_output, BATCH_SIZE * CONV4_FILTERS * conv4_output_height * conv4_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv4_output, 0, BATCH_SIZE * CONV4_FILTERS * conv4_output_height * conv4_output_width * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_pool2_output, BATCH_SIZE * CONV2_FILTERS * pool2_output_height * pool2_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_pool2_output, 0, BATCH_SIZE * CONV2_FILTERS * pool2_output_height * pool2_output_width * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv3_output, BATCH_SIZE * CONV3_FILTERS * conv3_output_height * conv3_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv3_output, 0, BATCH_SIZE * CONV3_FILTERS * conv3_output_height * conv3_output_width * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv2_output, BATCH_SIZE * CONV2_FILTERS * conv2_output_height * conv2_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv2_output, 0, BATCH_SIZE * CONV2_FILTERS * conv2_output_height * conv2_output_width * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_pool1_output, BATCH_SIZE * CONV1_FILTERS * pool1_output_height * pool1_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_pool1_output, 0, BATCH_SIZE * CONV1_FILTERS * pool1_output_height * pool1_output_width * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_pool0_output, BATCH_SIZE * CONV1_FILTERS * pool1_output_height * pool1_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_pool0_output, 0, BATCH_SIZE * CONV1_FILTERS * pool1_output_height * pool1_output_width * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_conv1_output, BATCH_SIZE * CONV1_FILTERS * conv1_output_height * conv1_output_width * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv1_output, 0, BATCH_SIZE * CONV1_FILTERS * conv1_output_height * conv1_output_width * sizeof(float)));
}

void ConvolutionalNeuralNetwork::freeMemory() {
    // Free device memory for weights and biases
    cudaFree(d_conv1_weights);
    cudaFree(d_conv1_bias);
    cudaFree(d_conv2_weights);
    cudaFree(d_conv2_bias);
    cudaFree(d_conv3_weights);
    cudaFree(d_conv3_bias);
    cudaFree(d_conv4_weights);
    cudaFree(d_conv4_bias);
    cudaFree(d_fc1_weights);
    cudaFree(d_fc1_bias);
    cudaFree(d_fc2_weights);
    cudaFree(d_fc2_bias);
    cudaFree(d_fc3_weights);
    cudaFree(d_fc3_bias);
    cudaFree(d_fc4_weights);
    cudaFree(d_fc4_bias);
    cudaFree(d_fc5_weights);
    cudaFree(d_fc5_bias);
    cudaFree(d_conv5_weights);
    cudaFree(d_conv5_bias);

    // Free device memory for intermediate results
    cudaFree(d_input);
    cudaFree(d_conv1_output);
    cudaFree(d_pool1_output);
    cudaFree(d_pool1_max_indices);
    cudaFree(d_conv2_output);
    cudaFree(d_pool2_output);
    cudaFree(d_pool2_max_indices);
    cudaFree(d_conv3_output);
    cudaFree(d_pool3_output);
    cudaFree(d_pool3_max_indices);
    cudaFree(d_conv4_output);
    cudaFree(d_pool4_output);
    cudaFree(d_pool4_max_indices);
    cudaFree(d_conv5_output);
    cudaFree(d_fc1_output);
    cudaFree(d_fc2_output);
    cudaFree(d_fc3_output);
    cudaFree(d_fc4_output);
    cudaFree(d_fc5_output);
    cudaFree(d_output);
    cudaFree(d_labels);
    cudaFree(d_loss);

    // Free device memory for gradients
    cudaFree(d_grad_conv1_weights);
    cudaFree(d_grad_conv1_bias);
    cudaFree(d_grad_conv2_weights);
    cudaFree(d_grad_conv2_bias);
    cudaFree(d_grad_conv3_weights);
    cudaFree(d_grad_conv3_bias);
    cudaFree(d_grad_conv4_weights);
    cudaFree(d_grad_conv4_bias);
    cudaFree(d_grad_fc1_weights);
    cudaFree(d_grad_fc1_bias);
    cudaFree(d_grad_fc2_weights);
    cudaFree(d_grad_fc2_bias);
    cudaFree(d_grad_fc3_weights);
    cudaFree(d_grad_fc3_bias);
    cudaFree(d_grad_fc4_weights);
    cudaFree(d_grad_fc4_bias);
    cudaFree(d_grad_fc5_weights);
    cudaFree(d_grad_fc5_bias);
    cudaFree(d_grad_conv5_weights);
    cudaFree(d_grad_conv5_bias);
    cudaFree(d_grad_output);
    cudaFree(d_grad_fc5_output);
    cudaFree(d_grad_fc4_output);
    cudaFree(d_grad_fc3_output);
    cudaFree(d_grad_fc2_output);
    cudaFree(d_grad_fc1_output);
    cudaFree(d_grad_pool4_output);
    cudaFree(d_grad_conv5_output);
    cudaFree(d_grad_pool3_output);
    cudaFree(d_grad_conv4_output);
    cudaFree(d_grad_pool2_output);
    cudaFree(d_grad_conv3_output);
    cudaFree(d_grad_pool1_output);
    cudaFree(d_grad_conv2_output);
    cudaFree(d_grad_pool0_output);
    cudaFree(d_grad_conv1_output);
}

void ConvolutionalNeuralNetwork::forwardPass(float* d_input, int* d_labels) {
    // Debug print for input
    float h_input_sample[INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
    cudaError_t error = cudaMemcpy(h_input_sample, d_input, sizeof(float) * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) {
        printf("CUDA error in cudaMemcpy: %s\n", cudaGetErrorString(error));
    } else {
        int non_zero_count = 0;
        for (int i = 0; i < INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH; i++) {
            if (h_input_sample[i] > 0) non_zero_count++;
        }

    }

    // Conv1 layer
    dim3 conv1BlockDim(16, 16);
    dim3 conv1GridDim((conv1_output_width + conv1BlockDim.x - 1) / conv1BlockDim.x,
                     (conv1_output_height + conv1BlockDim.y - 1) / conv1BlockDim.y,
                     CONV1_FILTERS * BATCH_SIZE);

    convolutionForwardKernel<<<conv1GridDim, conv1BlockDim>>>(d_input, d_conv1_output, d_conv1_weights, d_conv1_bias,
                                                BATCH_SIZE, INPUT_CHANNELS, CONV1_FILTERS,
                                                INPUT_HEIGHT, INPUT_WIDTH, conv1_output_height, conv1_output_width,
                                                CONV1_KERNEL_SIZE, CONV1_PADDING, CONV1_STRIDE);

    // ReLU activation
    int relu1Size = BATCH_SIZE * CONV1_FILTERS * conv1_output_height * conv1_output_width;
    int relu1Threads = 256;
    int relu1Blocks = (relu1Size + relu1Threads - 1) / relu1Threads;
    reluActivationKernel<<<relu1Blocks, relu1Threads>>>(d_conv1_output, d_conv1_output, relu1Size);

    // Max pooling
    dim3 pool1BlockDim(16, 16);
    dim3 pool1GridDim((pool1_output_width + pool1BlockDim.x - 1) / pool1BlockDim.x,
                     (pool1_output_height + pool1BlockDim.y - 1) / pool1BlockDim.y,
                     CONV1_FILTERS * BATCH_SIZE);

    maxPoolingForwardKernel<<<pool1GridDim, pool1BlockDim>>>(d_conv1_output, d_pool1_output, d_pool1_max_indices,
                                                 BATCH_SIZE, CONV1_FILTERS,
                                                 conv1_output_height, conv1_output_width, pool1_output_height, pool1_output_width,
                                                 POOL1_SIZE, POOL1_STRIDE);

    // Conv2 layer
    dim3 conv2BlockDim(16, 16);
    dim3 conv2GridDim((conv2_output_width + conv2BlockDim.x - 1) / conv2BlockDim.x,
                     (conv2_output_height + conv2BlockDim.y - 1) / conv2BlockDim.y,
                     CONV2_FILTERS * BATCH_SIZE);

    convolutionForwardKernel<<<conv2GridDim, conv2BlockDim>>>(d_pool1_output, d_conv2_output, d_conv2_weights, d_conv2_bias,
                                                BATCH_SIZE, CONV1_FILTERS, CONV2_FILTERS,
                                                pool1_output_height, pool1_output_width, conv2_output_height, conv2_output_width,
                                                CONV2_KERNEL_SIZE, CONV2_PADDING, CONV2_STRIDE);

    // ReLU activation
    int relu2Size = BATCH_SIZE * CONV2_FILTERS * conv2_output_height * conv2_output_width;
    int relu2Threads = 256;
    int relu2Blocks = (relu2Size + relu2Threads - 1) / relu2Threads;
    reluActivationKernel<<<relu2Blocks, relu2Threads>>>(d_conv2_output, d_conv2_output, relu2Size);

    // Max pooling
    dim3 pool2BlockDim(16, 16);
    dim3 pool2GridDim((pool2_output_width + pool2BlockDim.x - 1) / pool2BlockDim.x,
                     (pool2_output_height + pool2BlockDim.y - 1) / pool2BlockDim.y,
                     CONV2_FILTERS * BATCH_SIZE);

    maxPoolingForwardKernel<<<pool2GridDim, pool2BlockDim>>>(d_conv2_output, d_pool2_output, d_pool2_max_indices,
                                                 BATCH_SIZE, CONV2_FILTERS,
                                                 conv2_output_height, conv2_output_width, pool2_output_height, pool2_output_width,
                                                 POOL2_SIZE, POOL2_STRIDE);

    // Fully connected layer 1
    int fc1Threads = 256;
    int fc1Blocks = (FC1_SIZE + fc1Threads - 1) / fc1Threads;
    dim3 fc1GridDim(fc1Blocks, BATCH_SIZE);

    fullyConnectedForwardKernel<<<fc1GridDim, fc1Threads>>>(d_pool2_output, d_fc1_output, d_fc1_weights, d_fc1_bias,
                                                          BATCH_SIZE, flattened_size, FC1_SIZE);

    // ReLU activation
    int relu3Size = BATCH_SIZE * FC1_SIZE;
    int relu3Threads = 256;
    int relu3Blocks = (relu3Size + relu3Threads - 1) / relu3Threads;
    reluActivationKernel<<<relu3Blocks, relu3Threads>>>(d_fc1_output, d_fc1_output, relu3Size);

    // Fully connected layer 2 (output layer)
    int fc2Threads = 256;
    int fc2Blocks = (OUTPUT_SIZE + fc2Threads - 1) / fc2Threads;
    dim3 fc2GridDim(fc2Blocks, BATCH_SIZE);

    fullyConnectedForwardKernel<<<fc2GridDim, fc2Threads>>>(d_fc1_output, d_output, d_fc2_weights, d_fc2_bias,
                                                          BATCH_SIZE, FC1_SIZE, OUTPUT_SIZE);

    // Softmax activation
    softmaxKernel<<<(BATCH_SIZE + 255) / 256, 256>>>(d_output, BATCH_SIZE, OUTPUT_SIZE);

    // Compute loss
    crossEntropyLossKernel<<<(BATCH_SIZE + 255) / 256, 256>>>(d_output, d_labels, d_loss, BATCH_SIZE, OUTPUT_SIZE);
}

void ConvolutionalNeuralNetwork::backwardPass() {
    // Zero out gradients before backward pass
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv1_weights, 0, h_conv1_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv1_bias, 0, h_conv1_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv2_weights, 0, h_conv2_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_conv2_bias, 0, h_conv2_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc1_weights, 0, h_fc1_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc1_bias, 0, h_fc1_bias.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc2_weights, 0, h_fc2_weights.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_grad_fc2_bias, 0, h_fc2_bias.size() * sizeof(float)));

    // Compute gradients for output layer
    softmaxGradientKernel<<<dim3((OUTPUT_SIZE + 31) / 32, BATCH_SIZE), 32>>>(d_output, d_labels, d_grad_output, BATCH_SIZE, OUTPUT_SIZE);

    // Backprop from output to FC1
    fullyConnectedBackwardKernel<<<(FC1_SIZE * OUTPUT_SIZE + 255) / 256, 256>>>(d_grad_output, d_fc1_output, d_fc2_weights,
                                                                              d_grad_fc2_weights, d_grad_fc2_bias, d_grad_fc1_output,
                                                                              BATCH_SIZE, FC1_SIZE, OUTPUT_SIZE);

    // ReLU backward for FC1
    reluBackwardKernel<<<(BATCH_SIZE * FC1_SIZE + 255) / 256, 256>>>(d_grad_fc1_output, d_fc1_output, d_grad_fc1_output, BATCH_SIZE * FC1_SIZE);

    // Backprop from FC1 to pool2
    fullyConnectedBackwardKernel<<<(flattened_size * FC1_SIZE + 255) / 256, 256>>>(d_grad_fc1_output, d_pool2_output, d_fc1_weights,
                                                                                 d_grad_fc1_weights, d_grad_fc1_bias, d_grad_pool2_output,
                                                                                 BATCH_SIZE, flattened_size, FC1_SIZE);

    // Backprop from pool2 to conv2
    maxPoolingBackwardKernel<<<(BATCH_SIZE * CONV2_FILTERS * pool2_output_height * pool2_output_width + 255) / 256, 256>>>(d_grad_pool2_output, d_pool2_max_indices, d_grad_conv2_output,
                                                                                                                     BATCH_SIZE, CONV2_FILTERS,
                                                                                                                     conv2_output_height, conv2_output_width, pool2_output_height);

    // ReLU backward for conv2
    reluBackwardKernel<<<(BATCH_SIZE * CONV2_FILTERS * conv2_output_height * conv2_output_width + 255) / 256, 256>>>(d_grad_conv2_output, d_conv2_output, d_grad_conv2_output, BATCH_SIZE * CONV2_FILTERS * conv2_output_height * conv2_output_width);

    // Backprop from conv2 to pool1
    conv2dBackwardKernel<<<(CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE + 255) / 256, 256>>>(d_grad_conv2_output, d_pool1_output, d_conv2_weights,
                                                                                                                   d_grad_conv2_weights, d_grad_conv2_bias, d_grad_pool1_output,
                                                                                                                   BATCH_SIZE, CONV1_FILTERS, CONV2_FILTERS,
                                                                                                                   pool1_output_height, pool1_output_width, CONV2_KERNEL_SIZE,
                                                                                                                   conv2_output_height, conv2_output_width, CONV2_STRIDE, CONV2_PADDING);

    // Backprop from pool1 to conv1
    maxPoolingBackwardKernel<<<(BATCH_SIZE * CONV1_FILTERS * pool1_output_height * pool1_output_width + 255) / 256, 256>>>(d_grad_pool1_output, d_pool1_max_indices, d_grad_conv1_output,
                                                                                                                     BATCH_SIZE, CONV1_FILTERS,
                                                                                                                     conv1_output_height, conv1_output_width, pool1_output_height);

    // ReLU backward for conv1
    reluBackwardKernel<<<(BATCH_SIZE * CONV1_FILTERS * conv1_output_height * conv1_output_width + 255) / 256, 256>>>(d_grad_conv1_output, d_conv1_output, d_grad_conv1_output, BATCH_SIZE * CONV1_FILTERS * conv1_output_height * conv1_output_width);

    // Backprop from conv1 to input
    conv2dBackwardKernel<<<(CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE + 255) / 256, 256>>>(d_grad_conv1_output, d_input, d_conv1_weights,
                                                                                                                   d_grad_conv1_weights, d_grad_conv1_bias, nullptr,
                                                                                                                   BATCH_SIZE, INPUT_CHANNELS, CONV1_FILTERS,
                                                                                                                   INPUT_HEIGHT, INPUT_WIDTH, CONV1_KERNEL_SIZE,
                                                                                                                   conv1_output_height, conv1_output_width, CONV1_STRIDE, CONV1_PADDING);
}

void ConvolutionalNeuralNetwork::updateParameters() {
    // Make sure all previous operations are completed
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // No debug output for cleaner console

    // Update conv1 weights and bias
    // Limit the number of threads to 128 to avoid configuration issues
    updateParametersKernel<<<(CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE + 127) / 128, 128>>>(d_conv1_weights, d_grad_conv1_weights, d_conv1_bias, d_grad_conv1_bias,
                                                                                                                       CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE, CONV1_FILTERS, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Update conv2 weights and bias
    updateParametersKernel<<<(CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE + 127) / 128, 128>>>(d_conv2_weights, d_grad_conv2_weights, d_conv2_bias, d_grad_conv2_bias,
                                                                                                                       CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE, CONV2_FILTERS, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Update fc1 weights and bias
    updateParametersKernel<<<(FC1_SIZE * flattened_size + 127) / 128, 128>>>(d_fc1_weights, d_grad_fc1_weights, d_fc1_bias, d_grad_fc1_bias,
                                                                           FC1_SIZE * flattened_size, FC1_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Update fc2 weights and bias
    updateParametersKernel<<<(OUTPUT_SIZE * FC1_SIZE + 127) / 128, 128>>>(d_fc2_weights, d_grad_fc2_weights, d_fc2_bias, d_grad_fc2_bias,
                                                                         OUTPUT_SIZE * FC1_SIZE, OUTPUT_SIZE, LEARNING_RATE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // No debug output for cleaner console
}

void ConvolutionalNeuralNetwork::train(MNISTDataset& train_dataset, MNISTDataset& val_dataset) {
    int num_batches = train_dataset.getNumBatches(BATCH_SIZE);

    // Create a thread pool for data loading
    const int max_threads = std::thread::hardware_concurrency();
    // Use 75% of available threads for data loading to avoid oversubscription
    const int num_data_threads = std::max(1, static_cast<int>(max_threads * 0.75));
    // Keep some threads for other system tasks
    const int batches_per_thread = (num_batches + num_data_threads - 1) / num_data_threads;

    std::cout << "\n===== Thread Information =====" << std::endl;
    std::cout << "Available CPU threads: " << max_threads << std::endl;
    std::cout << "Data loading threads: " << num_data_threads << std::endl;
    std::cout << "Batches per thread: " << batches_per_thread << std::endl;
    std::cout << "Total batches: " << num_batches << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << " samples" << std::endl;
    std::cout << "Total training samples: " << train_dataset.getNumSamples() << std::endl;
    std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "Number of epochs: " << NUM_EPOCHS << std::endl;
    std::cout << "==============================\n" << std::endl;

    // Shared data structures
    std::mutex mtx;

    // Create a vector of threads for the thread pool
    std::vector<std::thread> thread_pool;

    // Open log file for writing training metrics
    std::ofstream log_file("mnist_cnn_training.log");
    if (!log_file.is_open()) {
        std::cerr << "Warning: Could not open log file for writing." << std::endl;
    } else {
        // Write header to log file
        log_file << "epoch,train_loss,train_accuracy,val_loss,val_accuracy,time_seconds" << std::endl;
    }

    std::cout << "Starting training for " << NUM_EPOCHS << " epochs..." << std::endl;
    std::cout << "Total number of batches: " << num_batches << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << std::endl;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct_predictions = 0;
        int total_samples = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Process all batches using a thread pool for data loading
        // Create a queue of batch data to be processed
        std::queue<std::pair<std::vector<float>, std::vector<int>>> batch_queue;
        std::atomic<bool> done_loading(false);
        std::atomic<int> loaded_batches(0);
        std::condition_variable cv;

        // Start the thread pool for data loading
        for (int t = 0; t < num_data_threads; t++) {
            thread_pool.emplace_back([&, t]() {
                // Each thread processes a subset of batches
                for (int b = t; b < num_batches; b += num_data_threads) {
                    // Allocate vectors for this batch
                    std::vector<float> h_input(BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH);
                    std::vector<int> h_labels(BATCH_SIZE);

                    // Get batch data
                    train_dataset.getBatch(b, BATCH_SIZE, h_input.data(), h_labels.data());

                    // Add to queue with lock
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        batch_queue.push({std::move(h_input), std::move(h_labels)});
                        loaded_batches++;
                    }

                    // Notify main thread that new data is available
                    cv.notify_one();
                }
            });
        }

        // Process batches as they become available
        for (int batch = 0; batch < num_batches; batch++) {
            // Wait for data to be available
            std::vector<float> h_input;
            std::vector<int> h_labels;

            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]() { return !batch_queue.empty() || loaded_batches >= num_batches; });

                if (!batch_queue.empty()) {
                    // Get the next batch from the queue
                    auto& batch_data = batch_queue.front();
                    h_input = std::move(batch_data.first);
                    h_labels = std::move(batch_data.second);
                    batch_queue.pop();
                } else {
                    // This should not happen if all batches are loaded correctly
                    std::cerr << "Error: Batch queue is empty but not all batches are processed" << std::endl;
                    continue;
                }
            }

            // Check if d_input and d_labels are properly allocated
            if (d_input == nullptr || d_labels == nullptr) {
                std::cerr << "Error: Device memory not allocated properly" << std::endl;
                return;
            }

            // Copy input data and labels to device
            cudaError_t error = cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                std::cerr << "CUDA error in cudaMemcpy for d_input: " << cudaGetErrorString(error) << std::endl;
                return;
            }

            error = cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(int), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                std::cerr << "CUDA error in cudaMemcpy for d_labels: " << cudaGetErrorString(error) << std::endl;
                return;
            }

            // Forward pass
            forwardPass(d_input, d_labels);

            // Backward pass
            backwardPass();

            // Update parameters
            updateParameters();

            // Make sure all GPU operations are completed
            cudaDeviceSynchronize();

            // Calculate loss and accuracy
            std::vector<float> h_loss(BATCH_SIZE, 0.0f);
            std::vector<float> h_output(BATCH_SIZE * OUTPUT_SIZE, 0.0f);

            // Copy loss and output from device to host
            cudaMemcpy(h_loss.data(), d_loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_output.data(), d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            // Calculate batch statistics
            float batch_loss = 0.0f;
            int batch_correct = 0;

            for (int i = 0; i < BATCH_SIZE; i++) {
                batch_loss += h_loss[i];
                total_loss += h_loss[i];

                // Find predicted class (max probability)
                int predicted_class = 0;
                float max_prob = h_output[i * OUTPUT_SIZE];

                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (h_output[i * OUTPUT_SIZE + j] > max_prob) {
                        max_prob = h_output[i * OUTPUT_SIZE + j];
                        predicted_class = j;
                    }
                }

                if (predicted_class == h_labels[i]) {
                    correct_predictions++;
                    batch_correct++;
                }

                total_samples++;
            }

            // Print batch statistics
            std::cout << "Epoch " << epoch + 1 << ", Batch " << batch + 1 << "/" << num_batches
                      << ", Loss: " << std::fixed << std::setprecision(4) << batch_loss / BATCH_SIZE
                      << ", Acc: " << std::fixed << std::setprecision(2) << (static_cast<float>(batch_correct) / BATCH_SIZE) * 100.0f << "%"
                      << std::endl;
        }

        // Join all threads in the thread pool
        for (auto& thread : thread_pool) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        thread_pool.clear();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        float avg_loss = total_loss / total_samples;
        float accuracy = static_cast<float>(correct_predictions) / total_samples;

        // Evaluate on validation set
        float val_loss = 0.0f;
        float val_accuracy = evaluate(val_dataset, &val_loss);

        // Print to console
        std::cout << "Epoch " << epoch + 1 << "/" << NUM_EPOCHS
                  << ", Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << ", Acc: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "%"
                  << ", Time: " << std::fixed << std::setprecision(2) << elapsed.count() << "s"
                  << std::endl;

        // Log to file
        if (log_file.is_open()) {
            log_file << epoch + 1 << ","
                    << avg_loss << ","
                    << accuracy << ","
                    << val_loss << ","
                    << val_accuracy << ","
                    << elapsed.count() << std::endl;
        }
    }

    // Close log file
    if (log_file.is_open()) {
        log_file.close();
    }
}

float ConvolutionalNeuralNetwork::evaluate(MNISTDataset& dataset, float* val_loss_ptr) {
    int num_batches = (dataset.size() + BATCH_SIZE - 1) / BATCH_SIZE;
    int total_samples = 0;
    int correct_predictions = 0;
    float total_loss = 0.0f;

    // Host memory for input and labels
    std::vector<float> h_input(BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH);
    std::vector<int> h_labels(BATCH_SIZE);

    // Process a subset of batches for faster evaluation
    int eval_batches = std::min(num_batches, 5);

    for (int batch = 0; batch < eval_batches; batch++) {
        // Get batch data
        dataset.getBatch(batch, BATCH_SIZE, h_input.data(), h_labels.data());

        // Copy input data and labels to device
        cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Forward pass only (no backprop during evaluation)
        forwardPass(d_input, d_labels);

        // Copy results back
        std::vector<float> h_loss(BATCH_SIZE, 0.0f);
        std::vector<float> h_output(BATCH_SIZE * OUTPUT_SIZE, 0.0f);

        cudaMemcpy(h_loss.data(), d_loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output.data(), d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        // Calculate accuracy
        for (int i = 0; i < BATCH_SIZE; i++) {
            if (batch * BATCH_SIZE + i < dataset.size()) {
                total_loss += h_loss[i];

                // Find predicted class
                int predicted_class = 0;
                float max_prob = h_output[i * OUTPUT_SIZE];

                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (h_output[i * OUTPUT_SIZE + j] > max_prob) {
                        max_prob = h_output[i * OUTPUT_SIZE + j];
                        predicted_class = j;
                    }
                }

                if (predicted_class == h_labels[i]) {
                    correct_predictions++;
                }

                total_samples++;
            }
        }
    }

    float accuracy = static_cast<float>(correct_predictions) / total_samples;
    float avg_loss = total_loss / total_samples;

    if (val_loss_ptr != nullptr) {
        *val_loss_ptr = avg_loss;
    }

    return accuracy;
}
