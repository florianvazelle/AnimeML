#include <math.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>
#include <random>

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
//double init_weight() { return (((double)rand()) / ((double)RAND_MAX)) * 2 - 1; }
double init_weight() { return ((double)rand()) / ((double)RAND_MAX); }

void shuffleTrainingSetArray(std::vector<int>& array) {
    for (int j = 0; j < array.size(); j++) {
        array[j] = j;
    }

    // https://en.cppreference.com/w/cpp/algorithm/random_shuffle#Example
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(array.begin(), array.end(), g);
}

double updateWeight(double oldWeight, float learningRate, double targetValue, double actualValue, double entryValue) {
    return oldWeight - (learningRate * (actualValue - targetValue) * dSigmoid(actualValue) * entryValue);
}

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
    dllexport double* create_linear_model(int weights_count) {
        auto weights = new double[weights_count + 1];
        for (auto i = 0; i < weights_count + 1; i++) {
            weights[i] = init_weight();  // rand() / (double)RAND_MAX * 2.0 - 1.0;
        }

        return weights;
    }

    dllexport void train_linear_model_rosenblatt(double* model, double all_inputs[], int inputs_count, int sample_count,
                                                             double all_expected_outputs[], int expected_output_size, int epochs,
                                                             double learning_rate) {
        // TODO
    }

    dllexport void train_linear_model_rosenblatt_test(
        double* weights,
        const int weights_count,
        const int sample_count_size,
        double all_inputs[], int inputs_size, 
        double all_outputs[], int outputs_size,
        const int epochs,
        float learningRate
    ) {
        // TODO
        //for (size_t i = 0; i < weights_count; i++) {
        //    weights[i] = weights[i] + 100;
        //}

        inputs_size = 2;
        outputs_size = 1;
        learningRate = 0.5;

        std::vector<int> trainingSetOrder(sample_count_size);

        // Iterate with epochs
        for (int i = 0; i < epochs; i++) {    

            // shuffle the training set
            shuffleTrainingSetArray(trainingSetOrder);

            // for each training set
            for (int j = 0; j < sample_count_size; j++) {
                // select a training set ID
                int trainingSetID = trainingSetOrder[j];

                // create a copy of the inputs of the set
                std::vector<double> setInputs(inputs_size);
                for (int k = 0; k < setInputs.size(); k++) {
                    setInputs[k] = all_inputs[(trainingSetID * inputs_size) + k];
                }

                // create a copy of the outputs of the set
                std::vector<double> setOutputs(outputs_size);
                for (int k = 0; k < setOutputs.size(); k++) {
                    setOutputs[k] = all_outputs[(trainingSetID * outputs_size) + k];
                }

                // *** Predict Function ***

                // compute activation fonction
                double activation = 0;
                // Loop on outputs (here we have only one output)
                for (int k = 0; k < outputs_size; k++) {
                    // Loop on inputs
                    for (int l = 0; l < inputs_size; l++) {
                        activation += setInputs[l] * weights[l];
                    }
                    // compute the sigmoid of the activation
                    activation = sigmoid(activation);
                    // compute the error
                    double squaredError = learningRate * pow((float)(activation - setOutputs[k]), 2);
                }

                // *** Learn Function ***

                // backpropagation of error on weights
                for (int k = 0; k < outputs_size; k++)  // One unique output
                {
                    for (int l = 0; l < inputs_size; l++) {
                        weights[l] = updateWeight(
                            weights[l],
                            0.5f,
                            setOutputs[0],  // one output
                            activation,
                            setInputs[l]);
                    }
                }
            }
        }
    }

    dllexport double predict_linear_model_rosenblatt_test(
        double weights[],
        double setInputs[],
        double setOutputs[], 
        int inputs_size = 2,
        int outputs_size = 1  
    ) {
        // compute activation fonction
        double activation = 0;
        // Loop on outputs (here we have only one output)
        for (int k = 0; k < outputs_size; k++) {
            // Loop on inputs
            for (int l = 0; l < inputs_size; l++) {
                activation += setInputs[l] * weights[l];
            }
            // compute the sigmoid of the activation
            activation = sigmoid(activation);

            // compute the error
            //double squaredError = learningRate * pow((float)(activation - setOutputs[k]), 2);

            return activation;
        }
        return 0;
    }

    dllexport void delete_linear_model(double* model) {
        delete[] model;
    }
}