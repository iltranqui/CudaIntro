#include "mnist_cnn.h"
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string train_csv = "dataset_mnist/train.csv";
    std::string test_csv = "dataset_mnist/test.csv";
    std::string image_dir = "dataset_mnist";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--train" && i + 1 < argc) {
            train_csv = argv[++i];
        } else if (arg == "--test" && i + 1 < argc) {
            test_csv = argv[++i];
        } else if (arg == "--dir" && i + 1 < argc) {
            image_dir = argv[++i];
        }
    }

    // Print CUDA device information
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    }

    // Set device to use
    cudaSetDevice(0);

    // Load datasets
    std::cout << "Loading training dataset from " << train_csv << " in directory " << image_dir << std::endl;
    MNISTDataset train_dataset(train_csv, image_dir);

    std::cout << "Loading test dataset from " << test_csv << " in directory " << image_dir << std::endl;
    MNISTDataset test_dataset(test_csv, image_dir);

    // Create and train the CNN
    std::cout << "Creating CNN model..." << std::endl;
    ConvolutionalNeuralNetwork cnn;

    std::cout << "Training CNN model..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    cnn.train(train_dataset, test_dataset);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Training completed in " << elapsed.count() << " seconds" << std::endl;

    // Skip evaluation for debugging
    std::cout << "Skipping evaluation for debugging..." << std::endl;

    return 0;
}
