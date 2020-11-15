#include <LinearModel.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

LinearModel::LinearModel(int weights_count) : BaseModel(weights_count) {
    weights = new double[weights_count];
    // Init all weights and biases between 0.0 and 1.0
    for (int i = 0; i < weights_count; i++) {
        weights[i] = 2 * (((double)rand()) / ((double)RAND_MAX)) - 1.0;
        // rand() / (double)RAND_MAX * 2.0 - 1.0;
    }
}

void LinearModel::train(int sample_count, double* train_inputs, int inputs_size, double* train_outputs, int outputs_size, int epochs,
                        double learning_rate) {
    std::vector<int> trainingSetOrder(sample_count);

    for (int i = 0; i < trainingSetOrder.size(); i++) {
        trainingSetOrder[i] = i;
    }

    // Iterate with epochs
    for (int i = 0; i < epochs; i++) {
        // shuffle the training set
        _shuffle(trainingSetOrder);

        // for each training set
        for (int j = 0; j < sample_count; j++) {
            // select a training set ID
            int trainingSetID = trainingSetOrder[j];

            // create a copy of the inputs of the set
            std::vector<double> setInputs(inputs_size);
            for (int k = 0; k < setInputs.size(); k++) {
                setInputs[k] = train_inputs[(trainingSetID * inputs_size) + k];
            }

            // create a copy of the outputs of the set
            std::vector<double> setOutputs(outputs_size);
            for (int k = 0; k < setOutputs.size(); k++) {
                setOutputs[k] = train_outputs[(trainingSetID * outputs_size) + k];
            }

            // *** Predict Function ***

            // compute activation fonction
            std::vector<double> activation(outputs_size);
            // Loop on outputs (here we have only one output)
            for (int k = 0; k < outputs_size; k++) {
                activation[k] = 0;

                // Loop on inputs
                for (int l = 0; l < inputs_size; l++) {
                    activation[k] += setInputs[l] * weights[l];
                }
                // compute the _sigmoid of the activation
                activation[k] = _sigmoid(activation[k]);

                // compute the error
                double squaredError = learning_rate * pow((float)(activation[k] - setOutputs[k]), 2);
            }

            // *** Learn Function ***

            // Backpropagation of error on weights / Adjust the weights
            for (int k = 0; k < outputs_size; k++) {  // One unique output
                for (int l = 0; l < inputs_size; l++) {
                    weights[l] = _update_weight(weights[l], learning_rate, setOutputs[k], activation[k], setInputs[l]);
                }
            }
        }
    }
}

void LinearModel::_shuffle(std::vector<int>& array) const {
    // https://en.cppreference.com/w/cpp/algorithm/random_shuffle#Example
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(array.begin(), array.end(), g);
}

double LinearModel::_update_weight(double old_weight, double learning_rate, double target_value, double actual_value, double entry_value) const {
    return old_weight - (learning_rate * (actual_value - target_value) * _sigmoid_derivative(actual_value) * entry_value);
}

void LinearModel::predict(double* inputs, int inputs_size, double* outputs, int outputs_size) const {
    // compute activation fonction
    std::vector<double> activation(outputs_size);

    // Loop on outputs (here we have only one output)
    for (int k = 0; k < outputs_size; k++) {
        activation[k] = 0;

        // Loop on inputs
        for (int l = 0; l < inputs_size; l++) {
            activation[k] += inputs[k * inputs_size + l] * weights[l];
        }

        // compute the _sigmoid of the activation
        activation[k] = _sigmoid(activation[k]);

        outputs[k] = activation[k];
    }
}