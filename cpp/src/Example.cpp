#include <math.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include <fstream>
#include <iostream>


// Activation function and its derivative
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

// Init all weights and biases between 0.0 and 1.0
double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

extern "C" {
    #ifdef WIN32
    __declspec(dllexport)
    #endif
    int GetRandom() { return rand(); };

    #ifdef WIN32
    __declspec(dllexport)
    #endif
    int pre_alloc_test(double* ppdoubleArrayReceiver) {
        size_t stSize = sizeof(double) * (3 * 5);
        double doubleArray[3][5];
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                doubleArray[i][j] = i * 10 + j;
            }
        }

        memcpy(ppdoubleArrayReceiver, doubleArray, stSize);
        
        return 0;
    }

    #ifdef WIN32
    __declspec(dllexport)
    #endif
    int alloc_in_test(double* ppdoubleArrayReceiver) {
        size_t stSize = sizeof(double) * (3 * 5);
        double doubleArray[3][5];
        
        ppdoubleArrayReceiver = new double[(3 * 5)];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                doubleArray[i][j] = i * 10 + j;
            }
        }

        memcpy(ppdoubleArrayReceiver, doubleArray, stSize);

        return 3 * 5;
    }

    #ifdef WIN32
    __declspec(dllexport)
    #endif
    void my_free(double* ppdoubleArrayReceiver) { delete[] ppdoubleArrayReceiver; }

    #ifdef WIN32
    __declspec(dllexport)
    #endif
    void write() {
        std::ofstream outfile("test.txt");
        outfile << "my text here!" << std::endl;
        outfile.close();
    }

    // end of test functions

    // Neural network functions
    #ifdef WIN32
    __declspec(dllexport)
    #endif
    // return an array of weights depending on inputs count
    double* create_linear_model(int inputs_count) {
        auto weights = new double[inputs_count + 1];
        for (auto i = 0; i < inputs_count + 1; i++) {
            weights[i] = init_weight();  // rand() / (double)RAND_MAX * 2.0 - 1.0;
        }

        return weights;
    }

    #ifdef WIN32
        __declspec(dllexport)
    #endif  
    void delete_linear_model(double* model) {
        delete[] model;
    }
}