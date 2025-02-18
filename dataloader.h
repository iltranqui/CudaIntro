#pragma once
#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>

class DataLoader {
public:
    DataLoader(const std::string& dataset_path, int batch_size);
    bool next_batch(std::vector<float>& images, std::vector<int>& labels);

private:
    std::vector<std::vector<float>> images_;
    std::vector<int> labels_;
    size_t current_index_;
    int batch_size_;
    void load_mnist(const std::string& dataset_path);
};

#endif // DATALOADER_H
