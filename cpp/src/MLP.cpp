#include <MLP.hpp>

#include <Utils.hpp>
#include <iostream>

MLP::MLP(std::vector<int> layers, int weights_count, bool is_classification) : BaseModel(weights_count, is_classification) {
    weights = new double[weights_count + 1];
    // Init all weights and biases between -1.0 and 1.0
    for (int i = 0; i < weights_count + 1; i++) {
        weights[i] = ml::rand(-1, 1);
    }

    // // Init all weights in 2D vector 
    // for (int i = 0; i < _hiddenLayerNbr; i++) {
    //     std::vector<double> v;
    //     for(int j = 0; j < _weightPerLayer + 1; j++) {
    //         v.push_back(ml::rand(-1, 1));
    //     }
    //     std::cout << "nombre de neurone :" << v.size() << std::endl;
    //     _weightsArray.push_back(v);
    // }
    // std::cout << "nombre de couches :" << _weightsArray.size() << std::endl;

    // For each layers
    for(int i = 0; i < layers.size(); i++) {
        // Create an array for each weights of each neuron
        std::vector<std::vector<double>> layer;
        for(int j = 0; j < layers[i] + 1; j++) { // "+1 for bias neuron"
            std::vector<double> neuron;
            if()
            for (int k = 0; k < layers[i - 1] ; k++) {
                neuron.push_back(ml::rand(-1, 1));
            }
        }
        _weightsArray.push_back(layer);
    }
    
}

void MLP::train(const Eigen::MatrixXd& train_inputs, const Eigen::MatrixXd& train_outputs, int epochs, double learning_rate) {
    const size_t sample_count = train_inputs.rows();
    const size_t inputs_size = train_inputs.cols();
    const size_t outputs_size = train_outputs.cols();

    if (is_classification) {
        std::vector<int> trainingSetOrder(sample_count);

        for (int i = 0; i < trainingSetOrder.size(); i++) {
            trainingSetOrder[i] = i;
        }

        // Matrix of solutions for every examples
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
    }
}

double MLP::_activation(double value) const {
    if (is_classification) {
        value = std::tanh(value);
        return (value != 0) ? (value > 0) ? 1 : -1 : 0;
    } else {
        return value;
    }
}

void MLP::predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs) const {
    assert(inputs.rows() == outputs.rows());  // or maybe resize outputs

    // for each sample
    for (int i = 0; i < inputs.rows(); i++) {
        // Loop on outputs (here we have only one output)
        for (int j = 0; j < outputs.cols(); j++) {
            double activation = weights[weights_count];
            // std::cout << "debug nombre de couches :" << _weightsArray.size() << std::endl;
            // std::cout << "debug nombre de neurones :" << _weightsArray[0].size() << std::endl;
            // std::cout << "id de couches cachees:" <<  _hiddenLayerNbr-1 << std::endl;
            // std::cout << "id de neurone :" << _weightPerLayer << std::endl;
            //double activation = _weightsArray[_hiddenLayerNbr-1][_weightPerLayer]; // access to the bias weight of the last layer
            
            // Loop on inputs
            for (int k = 0; k < inputs.cols(); k++) {
                activation += inputs(i, k) * weights[k];
                //activation += inputs(i, k) * _weightsArray[_hiddenLayerNbr-1][k];
            }

            // compute the _sigmoid of the activation
            outputs(i, j) = _activation(activation);
        }
    }
}