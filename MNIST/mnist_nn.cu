#include "mnist_nn.h"
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>

// CUDA Kernels Implementation

// Forward pass: Input to Hidden layer
__global__ void forwardInputToHiddenKernel(float* d_input, float* d_hidden, float* d_weights_ih, float* d_bias_h, int batch_size) {
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (hidden_idx < HIDDEN_SIZE && batch_idx < batch_size) {
        float sum = d_bias_h[hidden_idx];
        
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += d_input[batch_idx * INPUT_SIZE + i] * d_weights_ih[i * HIDDEN_SIZE + hidden_idx];
        }
        
        d_hidden[batch_idx * HIDDEN_SIZE + hidden_idx] = sum;
    }
}

// ReLU Activation function
__global__ void reluActivationKernel(float* d_hidden, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_hidden[idx] = fmaxf(0.0f, d_hidden[idx]);
    }
}

// Forward pass: Hidden to Output layer
__global__ void forwardHiddenToOutputKernel(float* d_hidden, float* d_output, float* d_weights_ho, float* d_bias_o, int batch_size) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (output_idx < OUTPUT_SIZE && batch_idx < batch_size) {
        float sum = d_bias_o[output_idx];
        
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += d_hidden[batch_idx * HIDDEN_SIZE + i] * d_weights_ho[i * OUTPUT_SIZE + output_idx];
        }
        
        d_output[batch_idx * OUTPUT_SIZE + output_idx] = sum;
    }
}

// Softmax activation function
__global__ void softmaxKernel(float* d_output, int batch_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        // Find max value for numerical stability
        float max_val = d_output[batch_idx * OUTPUT_SIZE];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            max_val = fmaxf(max_val, d_output[batch_idx * OUTPUT_SIZE + i]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            d_output[batch_idx * OUTPUT_SIZE + i] = expf(d_output[batch_idx * OUTPUT_SIZE + i] - max_val);
            sum += d_output[batch_idx * OUTPUT_SIZE + i];
        }
        
        // Normalize
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            d_output[batch_idx * OUTPUT_SIZE + i] /= sum;
        }
    }
}

// Compute cross-entropy loss
__global__ void computeLossKernel(float* d_output, int* d_labels, float* d_loss, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        int label = d_labels[batch_idx];
        float prob = d_output[batch_idx * OUTPUT_SIZE + label];
        
        // Cross-entropy loss: -log(p)
        d_loss[batch_idx] = -logf(fmaxf(prob, 1e-10f)); // Clip for numerical stability
    }
}

// Backpropagation: Output to Hidden layer
__global__ void backpropOutputToHiddenKernel(float* d_output, int* d_labels, float* d_hidden, float* d_weights_ho, 
                                            float* d_grad_weights_ho, float* d_grad_bias_o, int batch_size) {
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (hidden_idx < HIDDEN_SIZE) {
        for (int output_idx = 0; output_idx < OUTPUT_SIZE; output_idx++) {
            float grad_sum = 0.0f;
            
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                // Gradient of softmax + cross-entropy: p - y
                float grad = d_output[batch_idx * OUTPUT_SIZE + output_idx];
                if (d_labels[batch_idx] == output_idx) {
                    grad -= 1.0f;
                }
                
                // Accumulate gradient for weights
                grad_sum += grad * d_hidden[batch_idx * HIDDEN_SIZE + hidden_idx];
            }
            
            // Update weight gradients
            d_grad_weights_ho[hidden_idx * OUTPUT_SIZE + output_idx] = grad_sum / batch_size;
        }
    }
    
    // Compute bias gradients (one thread per output neuron)
    if (blockIdx.x == 0 && threadIdx.x < OUTPUT_SIZE) {
        int output_idx = threadIdx.x;
        float grad_sum = 0.0f;
        
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            float grad = d_output[batch_idx * OUTPUT_SIZE + output_idx];
            if (d_labels[batch_idx] == output_idx) {
                grad -= 1.0f;
            }
            grad_sum += grad;
        }
        
        d_grad_bias_o[output_idx] = grad_sum / batch_size;
    }
}

// Backpropagation: Hidden to Input layer
__global__ void backpropHiddenToInputKernel(float* d_grad_hidden, float* d_input, float* d_weights_ih, 
                                           float* d_grad_weights_ih, float* d_grad_bias_h, int batch_size) {
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (input_idx < INPUT_SIZE) {
        for (int hidden_idx = 0; hidden_idx < HIDDEN_SIZE; hidden_idx++) {
            float grad_sum = 0.0f;
            
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                // ReLU gradient
                float hidden_val = d_grad_hidden[batch_idx * HIDDEN_SIZE + hidden_idx];
                float relu_grad = (hidden_val > 0.0f) ? 1.0f : 0.0f;
                
                // Accumulate gradient for weights
                grad_sum += relu_grad * d_input[batch_idx * INPUT_SIZE + input_idx];
            }
            
            // Update weight gradients
            d_grad_weights_ih[input_idx * HIDDEN_SIZE + hidden_idx] = grad_sum / batch_size;
        }
    }
    
    // Compute bias gradients (one thread per hidden neuron)
    if (blockIdx.x == 0 && threadIdx.x < HIDDEN_SIZE) {
        int hidden_idx = threadIdx.x;
        float grad_sum = 0.0f;
        
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            float hidden_val = d_grad_hidden[batch_idx * HIDDEN_SIZE + hidden_idx];
            float relu_grad = (hidden_val > 0.0f) ? 1.0f : 0.0f;
            grad_sum += relu_grad;
        }
        
        d_grad_bias_h[hidden_idx] = grad_sum / batch_size;
    }
}

// Update parameters using gradients
__global__ void updateParametersKernel(float* d_weights, float* d_grad_weights, float* d_bias, float* d_grad_bias, 
                                      int rows, int cols, float learning_rate) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        d_weights[idx] -= learning_rate * d_grad_weights[idx];
    }
    
    // Update bias (one thread per output)
    if (blockIdx.y == 0 && col < cols) {
        d_bias[col] -= learning_rate * d_grad_bias[col];
    }
}

// MNIST Dataset Implementation
MNISTDataset::MNISTDataset(const std::string& csv_file, const std::string& base_path) : base_path(base_path) {
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << csv_file << std::endl;
        return;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        size_t comma_pos = line.find(',');
        if (comma_pos != std::string::npos) {
            std::string path = line.substr(0, comma_pos);
            int label = std::stoi(line.substr(comma_pos + 1));
            
            image_paths.push_back(path);
            labels.push_back(label);
        }
    }
    
    std::cout << "Loaded " << image_paths.size() << " images from " << csv_file << std::endl;
}

void MNISTDataset::getBatch(int batch_idx, std::vector<float>& batch_images, std::vector<int>& batch_labels) {
    int start_idx = batch_idx * BATCH_SIZE;
    int end_idx = std::min(start_idx + BATCH_SIZE, static_cast<int>(image_paths.size()));
    int actual_batch_size = end_idx - start_idx;
    
    batch_images.resize(actual_batch_size * INPUT_SIZE, 0.0f);
    batch_labels.resize(actual_batch_size);
    
    for (int i = 0; i < actual_batch_size; i++) {
        int idx = start_idx + i;
        std::string img_path = base_path + "/" + image_paths[idx];
        
        // Load image using OpenCV
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error: Could not read image " << img_path << std::endl;
            continue;
        }
        
        // Normalize pixel values to [0, 1]
        for (int r = 0; r < img.rows; r++) {
            for (int c = 0; c < img.cols; c++) {
                batch_images[i * INPUT_SIZE + r * img.cols + c] = img.at<uchar>(r, c) / 255.0f;
            }
        }
        
        batch_labels[i] = labels[idx];
    }
}

// Neural Network Implementation
NeuralNetwork::NeuralNetwork() {
    initializeParameters();
    allocateDeviceMemory();
}

NeuralNetwork::~NeuralNetwork() {
    freeDeviceMemory();
}

void NeuralNetwork::initializeParameters() {
    // Initialize weights with Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Input to hidden weights
    float ih_stddev = std::sqrt(2.0f / (INPUT_SIZE + HIDDEN_SIZE));
    std::normal_distribution<float> ih_dist(0.0f, ih_stddev);
    h_weights_ih.resize(INPUT_SIZE * HIDDEN_SIZE);
    for (auto& w : h_weights_ih) {
        w = ih_dist(gen);
    }
    
    // Hidden to output weights
    float ho_stddev = std::sqrt(2.0f / (HIDDEN_SIZE + OUTPUT_SIZE));
    std::normal_distribution<float> ho_dist(0.0f, ho_stddev);
    h_weights_ho.resize(HIDDEN_SIZE * OUTPUT_SIZE);
    for (auto& w : h_weights_ho) {
        w = ho_dist(gen);
    }
    
    // Initialize biases to zero
    h_bias_h.resize(HIDDEN_SIZE, 0.0f);
    h_bias_o.resize(OUTPUT_SIZE, 0.0f);
}

void NeuralNetwork::allocateDeviceMemory() {
    // Allocate memory for weights and biases
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights_ih, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias_h, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights_ho, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias_o, OUTPUT_SIZE * sizeof(float)));
    
    // Copy weights and biases to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights_ih, h_weights_ih.data(), INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias_h, h_bias_h.data(), HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights_ho, h_weights_ho.data(), HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias_o, h_bias_o.data(), OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate memory for temporary buffers
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));
    
    // Allocate memory for gradients
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weights_ih, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_bias_h, HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weights_ho, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_bias_o, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
}

void NeuralNetwork::freeDeviceMemory() {
    // Free weights and biases
    cudaFree(d_weights_ih);
    cudaFree(d_bias_h);
    cudaFree(d_weights_ho);
    cudaFree(d_bias_o);
    
    // Free temporary buffers
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_labels);
    cudaFree(d_loss);
    
    // Free gradients
    cudaFree(d_grad_weights_ih);
    cudaFree(d_grad_bias_h);
    cudaFree(d_grad_weights_ho);
    cudaFree(d_grad_bias_o);
    cudaFree(d_grad_hidden);
}

void NeuralNetwork::forwardPass(float* d_batch_input, int* d_batch_labels, int batch_size) {
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, d_batch_input, batch_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, d_batch_labels, batch_size * sizeof(int), cudaMemcpyHostToDevice));
    
    // Forward pass: Input to Hidden
    dim3 blockDim1(64);
    dim3 gridDim1((HIDDEN_SIZE + blockDim1.x - 1) / blockDim1.x, batch_size);
    forwardInputToHiddenKernel<<<gridDim1, blockDim1>>>(d_input, d_hidden, d_weights_ih, d_bias_h, batch_size);
    
    // Apply ReLU activation
    dim3 blockDim2(512);
    dim3 gridDim2((batch_size * HIDDEN_SIZE + blockDim2.x - 1) / blockDim2.x);
    reluActivationKernel<<<gridDim2, blockDim2>>>(d_hidden, batch_size * HIDDEN_SIZE);
    
    // Forward pass: Hidden to Output
    dim3 blockDim3(64);
    dim3 gridDim3((OUTPUT_SIZE + blockDim3.x - 1) / blockDim3.x, batch_size);
    forwardHiddenToOutputKernel<<<gridDim3, blockDim3>>>(d_hidden, d_output, d_weights_ho, d_bias_o, batch_size);
    
    // Apply Softmax activation
    softmaxKernel<<<batch_size, 1>>>(d_output, batch_size);
    
    // Compute loss
    dim3 blockDim4(256);
    dim3 gridDim4((batch_size + blockDim4.x - 1) / blockDim4.x);
    computeLossKernel<<<gridDim4, blockDim4>>>(d_output, d_labels, d_loss, batch_size);
}

float NeuralNetwork::backwardPass(int batch_size) {
    // Compute gradients for output to hidden layer
    dim3 blockDim1(64);
    dim3 gridDim1((HIDDEN_SIZE + blockDim1.x - 1) / blockDim1.x);
    backpropOutputToHiddenKernel<<<gridDim1, blockDim1>>>(d_output, d_labels, d_hidden, d_weights_ho, 
                                                        d_grad_weights_ho, d_grad_bias_o, batch_size);
    
    // Compute gradients for hidden to input layer
    dim3 blockDim2(64);
    dim3 gridDim2((INPUT_SIZE + blockDim2.x - 1) / blockDim2.x);
    backpropHiddenToInputKernel<<<gridDim2, blockDim2>>>(d_hidden, d_input, d_weights_ih, 
                                                       d_grad_weights_ih, d_grad_bias_h, batch_size);
    
    // Calculate average loss
    std::vector<float> h_loss(batch_size);
    CHECK_CUDA_ERROR(cudaMemcpy(h_loss.data(), d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        total_loss += h_loss[i];
    }
    
    return total_loss / batch_size;
}

void NeuralNetwork::updateParameters() {
    // Update input to hidden weights and biases
    dim3 blockDim1(64);
    dim3 gridDim1((HIDDEN_SIZE + blockDim1.x - 1) / blockDim1.x, INPUT_SIZE);
    updateParametersKernel<<<gridDim1, blockDim1>>>(d_weights_ih, d_grad_weights_ih, d_bias_h, d_grad_bias_h,
                                                  INPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    
    // Update hidden to output weights and biases
    dim3 blockDim2(64);
    dim3 gridDim2((OUTPUT_SIZE + blockDim2.x - 1) / blockDim2.x, HIDDEN_SIZE);
    updateParametersKernel<<<gridDim2, blockDim2>>>(d_weights_ho, d_grad_weights_ho, d_bias_o, d_grad_bias_o,
                                                  HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE);
}

void NeuralNetwork::train(MNISTDataset& train_dataset, MNISTDataset& val_dataset) {
    int num_samples = train_dataset.getNumSamples();
    int num_batches = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;
    
    std::cout << "Starting training for " << NUM_EPOCHS << " epochs..." << std::endl;
    std::cout << "Number of training samples: " << num_samples << std::endl;
    std::cout << "Number of batches per epoch: " << num_batches << std::endl;
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;
        
        // Create random indices for shuffling
        std::vector<int> indices(num_batches);
        for (int i = 0; i < num_batches; i++) {
            indices[i] = i;
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        for (int batch_idx : indices) {
            std::vector<float> batch_images;
            std::vector<int> batch_labels;
            
            train_dataset.getBatch(batch_idx, batch_images, batch_labels);
            
            int actual_batch_size = static_cast<int>(batch_labels.size());
            if (actual_batch_size == 0) continue;
            
            // Forward pass
            forwardPass(batch_images.data(), batch_labels.data(), actual_batch_size);
            
            // Backward pass
            float batch_loss = backwardPass(actual_batch_size);
            epoch_loss += batch_loss;
            
            // Update parameters
            updateParameters();
        }
        
        epoch_loss /= num_batches;
        
        // Evaluate on validation set
        float val_accuracy = evaluate(val_dataset);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::cout << "Epoch " << epoch + 1 << "/" << NUM_EPOCHS 
                  << ", Loss: " << std::fixed << std::setprecision(4) << epoch_loss 
                  << ", Val Accuracy: " << std::fixed << std::setprecision(2) << (val_accuracy * 100.0f) << "%" 
                  << ", Time: " << std::fixed << std::setprecision(2) << elapsed.count() << "s" << std::endl;
    }
    
    std::cout << "Training complete!" << std::endl;
}

float NeuralNetwork::evaluate(MNISTDataset& dataset) {
    int num_samples = dataset.getNumSamples();
    int num_batches = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;
    int correct = 0;
    int total = 0;
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        std::vector<float> batch_images;
        std::vector<int> batch_labels;
        
        dataset.getBatch(batch_idx, batch_images, batch_labels);
        
        int actual_batch_size = static_cast<int>(batch_labels.size());
        if (actual_batch_size == 0) continue;
        
        // Forward pass
        forwardPass(batch_images.data(), batch_labels.data(), actual_batch_size);
        
        // Copy output probabilities back to host
        std::vector<float> output_probs(actual_batch_size * OUTPUT_SIZE);
        CHECK_CUDA_ERROR(cudaMemcpy(output_probs.data(), d_output, actual_batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Find predicted classes
        for (int i = 0; i < actual_batch_size; i++) {
            int pred_class = 0;
            float max_prob = output_probs[i * OUTPUT_SIZE];
            
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output_probs[i * OUTPUT_SIZE + j] > max_prob) {
                    max_prob = output_probs[i * OUTPUT_SIZE + j];
                    pred_class = j;
                }
            }
            
            if (pred_class == batch_labels[i]) {
                correct++;
            }
            total++;
        }
    }
    
    return static_cast<float>(correct) / total;
}

int main() {
    // Set paths
    std::string base_path = "MNIST";
    std::string train_csv = base_path + "/train.csv";
    std::string test_csv = base_path + "/test.csv";
    
    // Load datasets
    MNISTDataset train_dataset(train_csv, base_path);
    MNISTDataset test_dataset(test_csv, base_path);
    
    // Create and train neural network
    NeuralNetwork nn;
    nn.train(train_dataset, test_dataset);
    
    // Final evaluation
    float test_accuracy = nn.evaluate(test_dataset);
    std::cout << "Final Test Accuracy: " << std::fixed << std::setprecision(2) << (test_accuracy * 100.0f) << "%" << std::endl;
    
    return 0;
}
