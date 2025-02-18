#include "dataloader.h"
#include <fstream>
#include <iostream>

DataLoader::DataLoader(const std::string& dataset_path, int batch_size)
    : batch_size_(batch_size), current_index_(0) {
    load_mnist(dataset_path);
}

void DataLoader::load_mnist(const std::string& dataset_path) {
    // Load MNIST dataset from `dataset_path` (Implement as needed)
    std::cout << "Loading MNIST dataset..." << std::endl;
}

bool DataLoader::next_batch(std::vector<float>& images, std::vector<int>& labels) {
    if (current_index_ >= images_.size()) return false;

    size_t end_index = std::min(current_index_ + batch_size_, images_.size());
    images.assign(images_.begin() + current_index_, images_.begin() + end_index);
    labels.assign(labels_.begin() + current_index_, labels_.begin() + end_index);

    current_index_ = end_index;
    return true;
}
