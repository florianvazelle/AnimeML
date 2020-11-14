#include <math.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include <fstream>
#include <iostream>

#ifdef WIN32
#define dllexport __declspec(dllexport)
#else
#define dllexport
#endif

/**
 * Classes
 */

class BaseModel {
    double* data;
    
    public:             
        virtual void print() = 0;
};

class LinearModel : public BaseModel {
    private:
        std::string msg = "Linear";

    public:
        void print() {
            std::ofstream outfile("Linear.txt");
            outfile << msg << std::endl;
            outfile.close();
        }
};

class MLP : public BaseModel {
    private:
        std::string msg = "MLP";

    public:
        void print() {
            std::ofstream outfile("MLP.txt");
            outfile << msg << std::endl;
            outfile.close();
        }
};

// Activation function and its derivative
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

// Init all weights and biases between 0.0 and 1.0
double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

extern "C" {
    // Allow to create a model pointer
    dllexport BaseModel* CreateModel(int flag) { 
        switch(flag) {
            case 0:
                return new LinearModel{};
            case 1: 
                return new MLP{};
        }
        throw("Not a valid flag!");
    };

    // Example usage of call a BaseModel function
    dllexport void Print(BaseModel* model, int flag) { 
        switch(flag) {
            case 0: {
                    LinearModel* l = static_cast<LinearModel*>(model);
                    l->print();
                } break;
            case 1: { 
                    MLP* m = static_cast<MLP*>(model);
                    m->print();
                } break;
        }
    };

    // Allow to delete model pointer to avoid leak
    dllexport void DeleteModel(BaseModel* model) { delete model; };

    // end of test functions

    /**
     * Neural network functions
     */

    // return an array of weights depending on inputs count
    dllexport double* create_linear_model(int inputs_count) {
        auto weights = new double[inputs_count + 1];
        for (auto i = 0; i < inputs_count + 1; i++) {
            weights[i] = init_weight();  // rand() / (double)RAND_MAX * 2.0 - 1.0;
        }

        return weights;
    }

    dllexport void train_linear_model_rosenblatt(double* model, double all_inputs[], int inputs_count, int sample_count,
                                                             double all_expected_outputs[], int expected_output_size, int epochs,
                                                             double learning_rate) {
        // TODO
        return;
    }

    dllexport void delete_linear_model(double* model) {
        delete[] model;
    }
}