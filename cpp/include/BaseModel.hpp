#ifndef BASEMODEL_HPP
#define BASEMODEL_HPP

#include <math.h> /* exp */
#include <fstream>
#include <vector>

class BaseModel {
  public:
    BaseModel(int weights_count) : weights_count(weights_count) {}
    virtual ~BaseModel() { delete weights; }

    inline double* getWeigths() const { return weights; }

    virtual void train(int sample_count, double* train_inputs, int inputs_size, double* train_outputs, int output_size, int epochs, double learning_rate) = 0;
    virtual void predict(double* inputs, int inputs_size, double* outputs, int outputs_size) const = 0;

  protected:
    double* weights;
    int weights_count;

    // The Sigmoid function, which describes an S shaped curve.
    // We pass the weighted sum of the inputs through this function to
    // normalise them between 0 and 1.
    inline double _sigmoid(double x) const { return 1 / (1 + exp(-x)); }

    // The derivative of the Sigmoid function.
    // This is the gradient of the Sigmoid curve.
    // It indicates how confident we are about the existing weight
    inline double _sigmoid_derivative(double x) const { return x * (1 - x); }

    void debuglog(std::string msg) {
        std::fstream fout;
        fout.open("debug.txt", std::ios::out | std::ios::app);
        fout << msg << '\n';
        fout.close();
    }
};

#endif