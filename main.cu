#include <iostream>
#include "train.h"

int main() {
    std::cout << "Starting MNIST Training..." << std::endl;

    // Set hyperparameters
    int epochs = 10;
    float learning_rate = 0.01;
    int batch_size = 64;

    // Train the model
    train(epochs, learning_rate, batch_size);

    std::cout << "Training complete!" << std::endl;
    return 0;
}
