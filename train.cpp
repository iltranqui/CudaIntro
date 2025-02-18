#include "train.h"
#include "dataloader.h"
#include "network.h"
#include <iostream>

void train(int epochs, float learning_rate, int batch_size) {
    DataLoader loader("path/to/mnist", batch_size);
    NeuralNet model;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<float> images, output, grad_output;
        std::vector<int> labels;

        while (loader.next_batch(images, labels)) {
            model.forward(images, output);

            // Compute loss & gradients (simple example)
            grad_output = output; // Placeholder for actual gradient computation

            model.backward(grad_output, learning_rate);
        }

        std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
    }
}
