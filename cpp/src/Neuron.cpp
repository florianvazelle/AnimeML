#include <Neuron.hpp>

#include <vector>
#include <iostream>

double Neuron::learningRate {0.15};
double Neuron::alpha = {0.5};

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c) {
        _outputWeights.push_back(Connection());
        _outputWeights.back().weight = randomWeight();
    }

    _myIndex = myIndex;    
}

void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;

    // Sum the previous layer's outputs
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n]._outputWeights[_myIndex].weight;
    }
    
    _outputVal = Neuron::activationFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - _outputVal;
    _gradient = delta * Neuron::activationFunctionDerivative(_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    // a real deal is that we don't know the target value of hidden neurons ...
    // To mimic that we will look at the sum of the derivative weights of the next layer
    double dow = sumDOW(nextLayer); // DOW for "Derivative Of Weight"
    _gradient = dow * Neuron::activationFunctionDerivative(_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer){
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        std::cout << "1) prev layer neuron : " << n << std::endl;
        std::cout << "myIndex : " << _myIndex << std::endl;
        std::cout << "size of outputWeights : " << neuron._outputWeights.size() << std::endl; 
        double oldDeltaWeight = neuron._outputWeights[_myIndex].deltaWeight; // ca merde
        // std::cout << "2) prev layer neuron : " << n << std::endl;
        double newDeltaWeight = 
            // Individual input. magnified by the gradient an train rate;
            learningRate
            * neuron.getOutputVal()
            * _gradient
            // Also add momentum => a fraction of the previous delta weight
            + alpha // momentum rate
            * oldDeltaWeight;
        
        neuron._outputWeights[_myIndex].deltaWeight = newDeltaWeight;
        neuron._outputWeights[_myIndex].weight += newDeltaWeight;
       
    }
}

double Neuron::activationFunction(double x){ // can be other activation functions
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x){
    return 1.0 - x * x;
}

double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) { // not including the bias
        sum += _outputWeights[n].weight * nextLayer[n]._gradient;
    }

    return sum;
}

