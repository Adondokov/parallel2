#include <iostream>
#include <vector>
#include <omp.h>

// Function to multiply matrix by vector
void matrixVectorMultiply(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector, std::vector<double>& result) {
    int numRows = matrix.size();
    int numCols = matrix[0].size();
    
    #pragma omp parallel for
    for (int i = 0; i < numRows; ++i) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < numCols; ++j) {
            sum += matrix[i][j] * vector[j];
        }
        result[i] = sum;
    }
}

// Function to initialize matrix and vector
void initialize(std::vector<std::vector<double>>& matrix, std::vector<double>& vector, int N) {
    int numRows = matrix.size();
    int numCols = matrix[0].size();
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            if (i == j) {
                matrix[i][j] = 2.0; // Elements on main diagonal
            } else {
                matrix[i][j] = 1.0; 
            }
        }
        vector[i] = N + 1.0; // All elements in vector b are N + 1
    }
}

int main() {
    const int numThreads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    const int numMatrices[] = {20000, 40000};
    
    for (int n : numMatrices) {
        std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
        std::vector<double> vector(n);
        std::vector<double> result(n);
        initialize(matrix, vector, n);
        
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        std::cout << "Number of Threads |seconds" << std::endl;
        
        for (int numThread : numThreads) {
            omp_set_num_threads(numThread);
            auto startTime = omp_get_wtime();
            matrixVectorMultiply(matrix, vector, result);
            auto endTime = omp_get_wtime();
            std::cout << " " << numThread << " | " << endTime - startTime << std::endl;
        }
        std::cout << std::endl;
    }
    return 0;
}