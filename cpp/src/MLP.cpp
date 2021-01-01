#include <MLP.hpp>

#include <Utils.hpp>
#include <iostream>

MLP::MLP(std::vector<int> layers, int weights_count, bool is_classification) : BaseModel(weights_count, is_classification) {
    weights = new double[weights_count + 1];
    // Init all weights and biases between -1.0 and 1.0
    for (int i = 0; i < weights_count + 1; i++) {
        weights[i] = ml::rand(-1, 1);
    }

}

void MLP::train(const Eigen::MatrixXd& train_inputs, const Eigen::MatrixXd& train_outputs, int epochs, double learning_rate) {

}

double MLP::_activation(double value) const {
    return 0;
}

void MLP::predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs) const {

}