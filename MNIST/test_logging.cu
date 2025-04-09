#include <iostream>
#include <fstream>
#include <iomanip>

int main() {
    std::cout << "MNIST CNN Logging Test" << std::endl;
    
    // Create a log file
    std::ofstream log_file("mnist_cnn_test.log");
    if (log_file.is_open()) {
        log_file << "epoch,train_loss,train_accuracy,val_loss,val_accuracy,time_seconds" << std::endl;
        
        // Simulate training data
        for (int epoch = 1; epoch <= 10; epoch++) {
            float train_loss = 2.5f / epoch;
            float train_accuracy = 0.5f + 0.05f * epoch;
            float val_loss = 2.0f / epoch;
            float val_accuracy = 0.55f + 0.04f * epoch;
            float time_seconds = 10.0f + epoch;
            
            log_file << epoch << ","
                    << std::fixed << std::setprecision(4) << train_loss << ","
                    << std::fixed << std::setprecision(4) << train_accuracy << ","
                    << std::fixed << std::setprecision(4) << val_loss << ","
                    << std::fixed << std::setprecision(4) << val_accuracy << ","
                    << std::fixed << std::setprecision(2) << time_seconds << std::endl;
            
            std::cout << "Epoch " << epoch << "/10"
                      << ", Train Loss: " << std::fixed << std::setprecision(4) << train_loss
                      << ", Train Acc: " << std::fixed << std::setprecision(4) << train_accuracy
                      << ", Val Loss: " << std::fixed << std::setprecision(4) << val_loss
                      << ", Val Acc: " << std::fixed << std::setprecision(4) << val_accuracy
                      << ", Time: " << std::fixed << std::setprecision(2) << time_seconds << "s"
                      << std::endl;
        }
        
        log_file.close();
        std::cout << "Training data logged to mnist_cnn_test.log" << std::endl;
    } else {
        std::cerr << "Could not open log file for writing." << std::endl;
    }
    
    return 0;
}
