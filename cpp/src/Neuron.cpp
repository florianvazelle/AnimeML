#include <Neuron.hpp>

#include <vector>
#include <iostream>
#include <algorithm>

double Neuron::alpha = {0.5};

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c) {
        _outputWeights.push_back(Connection());
        _outputWeights.back().weight = randomWeight();
        // std::cout << "First weight: " << _outputWeights.back().weight << std::endl;
    }

    _myIndex = myIndex;    
}

void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;

    // Sum the previous layer's outputs
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); ++n) { // include the bias neuron
        //std::cout << " ***************** prevLayer[n].getOutputVal(): " << prevLayer[n].getOutputVal();
        //std::cout << " * prevLayer[n]._outputWeights[_myIndex].weight: " << prevLayer[n]._outputWeights[_myIndex].weight;
        //std::cout << std::endl;
        sum += prevLayer[n].getOutputVal() * prevLayer[n]._outputWeights[_myIndex].weight;
    }
    
    //std::cout << "sum of the previous layer: " << sum << std::endl;
    _outputVal = Neuron::activationFunction(sum);
    //_outputVal = Neuron::activationFunction(sum);
    //std::cout << "sum with activation: " << _outputVal << std::endl;
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

void Neuron::updateInputWeights(Layer& prevLayer, double learning_rate) {
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];

        double oldDeltaWeight = neuron._outputWeights[_myIndex].deltaWeight; // ca merde
        double newDeltaWeight = 
            // Individual input. magnified by the gradient an train rate;
            learning_rate
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
    return std::tanh(x);
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

