#ifndef NEURON_HPP
#define NEURON_HPP

#include <Utils.hpp>
#include <random>
#include <vector>

// struct to define links between Neurons
struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

// type to define the layer (insteed of a layer class)
typedef  std::vector<Neuron> Layer;

// ************************* class Neuron ****************************

class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(const double val) { _outputVal = val; }
    double getOutputVal(void) const { return _outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

    std::vector<Connection> _outputWeights;
    unsigned _myIndex;

  private:
    static double learningRate; // [0 -> 1] overall net training rate
    static double alpha; // [0 -> n] multiplier of last weight change (momentum)
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void) { return ml::rand(-1, 1); }
    double sumDOW(const Layer &nextLayer) const;
    double _outputVal;
    double _gradient;
};

#endif