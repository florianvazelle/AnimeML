#include <math.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include <fstream>
#include <iostream>

extern "C" {
    #ifdef WIN32
    __declspec(dllimport)
    #endif
    int GetRandom() { return rand(); };

    #ifdef WIN32
    __declspec(dllimport)
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
    __declspec(dllimport)
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
    __declspec(dllimport)
    #endif
    void my_free(double* ppdoubleArrayReceiver) { delete[] ppdoubleArrayReceiver; }

    #ifdef WIN32
    __declspec(dllimport)
    #endif
    void write() {
        std::ofstream outfile("test.txt");
        outfile << "my text here!" << std::endl;
        outfile.close();
    }
}
