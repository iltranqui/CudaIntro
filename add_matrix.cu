#define EXECUTION_PARAMETERS  // define this macro to include the execution parameters in the output

#ifdef EXECUTION_PARAMETERS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <random>
#include <stdexcept>
#include <typeinfo>

// Template function to create and populate a 2D matrix with random values
/*
 * @param width: width of the matrix
 * @param height: height of the matrix
 * @param min_value: minimum value for the random number generator
 * @param max_value: maximum value for the random number generator
 * @return: pointer to the 2D matrix
 */
template <typename T>
T** create2DMatrix(size_t width, size_t height, T min_value, T max_value) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_value, max_value);

    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<> int_dis(min_value, max_value);
        dis = int_dis; // Switch to integer distribution for integral types
    }

    // Allocate memory for the matrix
    T** matrix = new T * [height];
    for (size_t i = 0; i < height; ++i) {
        matrix[i] = new T[width];
        for (size_t j = 0; j < width; ++j) {
            matrix[i][j] = dis(gen); // Assign random values
        }
    }
    return matrix;
}

// Function to create a 2D matrix based on type_def
/*
 * @param width: width of the matrix
 * @param height: height of the matrix
 * @param type_def: type of the matrix (int, float, double)
 * @param min_value: minimum value for the random number generator
 * @param max_value: maximum value for the random number generator
 * @return: void pointer to the 2D matrix
 */
void* createMatrix(size_t width, size_t height, const std::string& type_def, double min_value, double max_value) {
    if (type_def == "int") {
        return static_cast<void*>(create2DMatrix<int>(width, height, static_cast<int>(min_value), static_cast<int>(max_value)));
    }
    else if (type_def == "float") {
        return static_cast<void*>(create2DMatrix<float>(width, height, static_cast<float>(min_value), static_cast<float>(max_value)));
    }
    else if (type_def == "double") {
        return static_cast<void*>(create2DMatrix<double>(width, height, min_value, max_value));
    }
    else {
        throw std::invalid_argument("Unsupported type: " + type_def);
    }
}

// Helper function to print an int matrix (for demonstration purposes)ù
/*   
* @param matrix: pointer to the 2D matrix
* @param width: width of the matrix
* @param height: height of the matrix
*/ 
template <typename T>
void printMatrix(T** matrix, size_t width, size_t height) {
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// How to use the createMatrix function
// How to use the createMatrix function
// Create a 2D matrix with random values
//int** matrix = static_cast<int**>(createMatrix(width, height, type_def, min_value, max_value));

int main() {
    // Matrix dimensions
    size_t width = 5;
    size_t height = 5;
    // Type of the matrix
    std::string type_def = "int";
    // Random number generator limits
    double min_value = 0;
    double max_value = 10;
    // Create a 2D matrix with random values
    int** matrix_1 = static_cast<int**>(createMatrix(width, height, type_def, min_value, max_value));
	int** matrix_2 = static_cast<int**>(createMatrix(width, height, type_def, min_value, max_value));

	int** matrix_result = new int* [height];
    // Print the matrix
    //printMatrix(matrix, width, height);

	// allocate memory in GPU
	int* d_matrix_1;
	int* d_matrix_2;
	int* d_matrix_result;
	cudaMalloc(&d_matrix_1, width * height * sizeof(int));
	cudaMalloc(&d_matrix_2, width * height * sizeof(int));
	cudaMalloc(&d_matrix_result, width * height * sizeof(int));

	// copy data to GPU
	cudaMemcpy(d_matrix_1, matrix_1[0], width * height * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_2, matrix_2[0], width * height * sizeof(int), cudaMemcpyHostToDevice);

    // Remember that matrix multiplication is made element wise


}