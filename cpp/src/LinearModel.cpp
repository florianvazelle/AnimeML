#include <LinearModel.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

LinearModel::LinearModel(int weights_count, bool is_classification) : BaseModel(weights_count, is_classification) {
    weights = new double[weights_count + 1];
    // Init all weights and biases between 0.0 and 1.0
    for (int i = 0; i < weights_count + 1; i++) {
        weights[i] = 2 * (((double)rand()) / RAND_MAX) - 1;
        // rand() / (double)RAND_MAX * 2.0 - 1.0;
    }
}

void LinearModel::train(int sample_count, const double* train_inputs, int inputs_size, const double* train_outputs, int outputs_size, int epochs, double learning_rate) {
    std::vector<int> trainingSetOrder(sample_count);

    for (int i = 0; i < trainingSetOrder.size(); i++) {
        trainingSetOrder[i] = i;
    }

    // Iterate with epochs
    for (int i = 0; i < epochs; i++) {
        // shuffle the training set
        std::random_shuffle(trainingSetOrder.begin(), trainingSetOrder.end());

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
            predict(1, setInputs.data(), inputs_size, activation.data(), outputs_size);

            // *** Learn Function ***

            // Backpropagation of error on weights / Adjust the weights
            for (int k = 0; k < outputs_size; k++) {  // One unique output
                for (int l = 0; l < inputs_size; l++) {
                    weights[l] = _update_weight(weights[l], learning_rate, setOutputs[k], activation[k], setInputs[l]);
                }
                weights[weights_count] = _update_weight(weights[weights_count], learning_rate, setOutputs[k], activation[k], 1);
            }
        }
    }
}

double LinearModel::_update_weight(double old_weight, double learning_rate, double target_value, double actual_value, double entry_value) const {
    return old_weight - (learning_rate * (actual_value - target_value) * entry_value);
}

double LinearModel::_activation(double value) const {
    if (is_classification)
        return (std::tanh(value) < 0) ? -1 : 1;
    else
        return value;
}

void LinearModel::predict(int sample_count, const double* inputs, int inputs_size, double* outputs, int outputs_size) const {
    for (int j = 0; j < sample_count; j++) {
        // Loop on outputs (here we have only one output)
        for (int k = 0; k < outputs_size; k++) {
            double activation = weights[weights_count];

            // Loop on inputs
            for (int l = 0; l < inputs_size; l++) {
                activation += inputs[j * inputs_size + l] * weights[l];
            }

            // compute the _sigmoid of the activation
            outputs[j * outputs_size + k] = _activation(activation);
        }
    }
}