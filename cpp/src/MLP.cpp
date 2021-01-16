#include <MLP.hpp>

#include <Utils.hpp>
#include <iostream>

MLP::MLP(const std::vector<unsigned> &topology, int weights_count, bool is_classification) : BaseModel(weights_count, is_classification) {
    // **************** not used *********************
    weights = new double[weights_count + 1];
    // Init all weights and biases between -1.0 and 1.0
    for (int i = 0; i < weights_count + 1; i++) {
        weights[i] = ml::rand(-1, 1);
    }
    // **************** end *********************

    unsigned int numLayers = (unsigned)topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        _layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        // We have made a new Layer, now fill it with neurons, and
        // add a bias neuron to the layer:
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            if (numOutputs == 0 && neuronNum == topology[layerNum]) {
                continue;
            }
            _layers.back().push_back(Neuron(numOutputs, neuronNum)); // ".back" last layer of vector
            std::cout << "Made a Neuron ! index : " << neuronNum << std::endl;
        }
    }
}

// pass on each samples and send results in outputs
void MLP::predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs){
    // assert if there is the same amount of samples in the inputs/outputs matrix
    assert(inputs.rows() == outputs.rows());  // or maybe resize outputs

    // for each sample
    for (int i = 0; i < inputs.rows(); i++) {
        std::vector<double> matrixInputsVector;
        for (int k = 0; k < inputs.cols(); k++) {
            matrixInputsVector.push_back(inputs(i,k)); // à vérifier
        }
        feedForward(matrixInputsVector);

        std::vector<double> res;
        getResults(res);

        // std::cout << "res : " << std::endl;

        for(auto i : res) {
            std::cout << i << std::endl;
        }
        
        for(int k = 0; k < res.size(); k++) {
            outputs(i, k) = res[k];
        }
    }
}

void MLP::getResults(std::vector<double>& resultVals){
    resultVals.clear();
    unsigned int lastLayerSize = (unsigned)_layers[_layers.size() - 1].size(); // size with bias
    
    std::cout << "last layer size : " << lastLayerSize << std::endl;

    for (unsigned i = 0; i < lastLayerSize - 1; ++i) { // - 1 for bias
        resultVals.push_back(_layers[_layers.size() - 1][i].getOutputVal());
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
        
        //Eigen::MatrixXd activation(sample_count, outputs_size);

        // Iterate with epochs
        for (int i = 0; i < epochs; i++) {
            std::cout << "Tour : " << i << std::endl;
            //predict(train_inputs, activation);

            // shuffle the training set
            ml::random_shuffle<int>(trainingSetOrder);

            // for each training set
            for (int j = 0; j < trainingSetOrder.size(); j++) {
                // chaque example -> forward -> backward
                std::vector<double> matrixInputsVector;
                for (int k = 0; k < train_inputs.cols(); k++) {
                    matrixInputsVector.push_back(train_inputs(trainingSetOrder[j],k));
                }
                feedForward(matrixInputsVector);
                //feedForward(trainingSetOrder[j]);

                std::vector<double> matrixOutputsVector;
                for (int k = 0; k < train_outputs.cols(); k++) {
                    matrixOutputsVector.push_back(train_outputs(trainingSetOrder[j],k));
                }
                //std::cout << "toto" << std::endl;
                backProp(matrixOutputsVector); // pwoblem
                
            }
        }
    }
    // Loop 
        // Get new input data and feed it forward:
        // -> feedForward(inputVals);

        // Collect the net's actual results:
        //getResults(resultVals);

        // Train the net with what the outputs should have been
        // -> backProp(targetVals);

        // Report how well the training is working
        // myNet.getRecentAverageError()

        //getResults(resultVals);
}

void MLP::feedForward(const std::vector<double> &inputVals){
    assert(inputVals.size() == _layers[0].size() - 1);

    // assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        _layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagate
    for (unsigned layerNum = 1; layerNum < _layers.size(); ++layerNum) {
        Layer &prevLayer = _layers[layerNum - 1];
        for (unsigned n = 0; n < _layers[layerNum].size() - 1; ++n) {
            _layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void MLP::backProp(const std::vector<double> &targetVals) {

    // Calculate Overall net error (RMS of output neuron errors) "Root Mean Square Error" 
    Layer &outputLayer = _layers.back();
    _error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        // compute difference between real vs expected value
        double delta = targetVals.at(n) - outputLayer.at(n).getOutputVal();
        _error += delta * delta;
    }
    _error /= outputLayer.size() - 1;
    _error = sqrt(_error); // RMS

    // Implement a recent average measurement: DEBUG !!
    _recentAverageError = 
        ( _recentAverageError * _recentAverageSmoothingFactor + _error)
        / (_recentAverageSmoothingFactor + 1.0);

    // std::cout << "titi" << std::endl;
    // Calculate output layer gradients

    // std::cout << "outputLayer.size() -> " << outputLayer.size() << std::endl;
    // std::cout << "targetVals.size() -> " << targetVals.size() << std::endl;
    for (unsigned n = 0; n < outputLayer.size(); ++n) {
        outputLayer.at(n).calcOutputGradients(targetVals.at(n));
    }

    // std::cout << "tata" << std::endl;
    // Calculate gradients on hidden layers

    for (unsigned layerNum = (unsigned)_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = _layers[layerNum]; // documentation purpose (can be optimize)
        Layer &nextLayer = _layers[layerNum + 1]; // documentation purpose (can be optimize)

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
        // update connection weights
    for (unsigned layerNum = (unsigned)_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = _layers[layerNum];
        Layer &prevLayer = _layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size(); ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
        std::cout << "tete" << std::endl;
    }
    std::cout << "you have finish the world boss" << std::endl;
}

// not used (the activation fonciton is in Neuron class)
double MLP::_activation(double value) const {
    return 0;
}