#include <LinearModel.hpp>

#include <Eigen/Dense>
#include <Utils.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

LinearModel::LinearModel(int weights_count, bool is_classification) : BaseModel(weights_count, is_classification) {
    weights = new double[weights_count + 1];
    // Init all weights and biases between -1.0 and 1.0
    for (int i = 0; i < weights_count + 1; i++) {
        weights[i] = ml::rand(-1, 1);
    }
}

void LinearModel::train(const Eigen::MatrixXd& train_inputs, const Eigen::MatrixXd& train_outputs, int epochs, double learning_rate) {
    const size_t sample_count = train_inputs.rows();
    const size_t inputs_size = train_inputs.cols();
    const size_t outputs_size = train_outputs.cols();

    if (is_classification) {
        std::vector<int> trainingSetOrder(sample_count);

        for (int i = 0; i < trainingSetOrder.size(); i++) {
            trainingSetOrder[i] = i;
        }

        Eigen::MatrixXd activation(sample_count, outputs_size);

        // Iterate with epochs
        for (int i = 0; i < epochs; i++) {
            // *** Predict Function ***

            // compute activation fonction
            predict(train_inputs, activation);  // or `predict(train_inputs.row(trainingSetID), activation);` in trainingSetOrder loops to predict by row

            // shuffle the training set
            ml::random_shuffle<int>(trainingSetOrder);

            // for each training set
            for (int j = 0; j < trainingSetOrder.size(); j++) {
                // select a training set ID
                int trainingSetID = trainingSetOrder[j];

                // *** Learn Function ***

                // Backpropagation of error on weights / Adjust the weights
                for (int k = 0; k < outputs_size; k++) {
                    const double target_value = train_outputs(trainingSetID, k);
                    const double actual_value = activation(trainingSetID, k);

                    double error = actual_value - target_value;
                    if (is_classification) {
                        error *= (actual_value * actual_value);
                    }

                    for (int l = 0; l < inputs_size; l++) {
                        const double entry_value = train_inputs(trainingSetID, l);

                        weights[l] -= (learning_rate * error * entry_value);
                    }
                    weights[weights_count] -= (learning_rate * error * 1);  // bias
                }
            }
        }
    } else { // to be verified
        // Add a column of one (at the right), for the bias
        Eigen::MatrixXd tmp(train_inputs.rows(), train_inputs.cols() + 1);
        Eigen::VectorXd vec(train_inputs.rows());
        for (int i = 0; i < train_inputs.rows(); i++) {
            vec(i) = 1;
        }
        tmp << train_inputs, vec;

        // Compute the transpose
        Eigen::MatrixXd inputs_transposed = tmp.transpose();
        Eigen::MatrixXd inv_inputs_transposed = (inputs_transposed * tmp).completeOrthogonalDecomposition().pseudoInverse();

        // Compute weights
        Eigen::MatrixXd w = inv_inputs_transposed * inputs_transposed * train_outputs;

        for (int i = 0; i < inputs_size + 1; i++) {
            weights[i] = w(i, 0);
        }
    }
}

double LinearModel::_activation(double value) const {
    if (is_classification) {
        value = std::tanh(value);
        return (value != 0) ? (value > 0) ? 1 : -1 : 0;
    } else {
        return value;
    }
}

void LinearModel::predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs){
    assert(inputs.rows() == outputs.rows());  // or maybe resize outputs

    // for each sample
    for (int i = 0; i < inputs.rows(); i++) {
        // Loop on outputs (here we have only one output)
        for (int j = 0; j < outputs.cols(); j++) {
            double activation = weights[weights_count];

            // Loop on inputs
            for (int k = 0; k < inputs.cols(); k++) {
                activation += inputs(i, k) * weights[k];
            }

            // compute the _sigmoid of the activation
            outputs(i, j) = _activation(activation);
        }
    }
}