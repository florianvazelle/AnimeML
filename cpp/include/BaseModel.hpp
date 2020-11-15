#ifndef BASEMODEL_HPP
#define BASEMODEL_HPP

#include <math.h> /* exp */
#include <fstream>
#include <vector>

class BaseModel {
  public:
    BaseModel(int weights_count) : weights_count(weights_count) {}

    inline double* getWeigths() const { return weights; }

    virtual void train(int sample_count, double* train_inputs, int inputs_size, double* train_outputs, int output_size, int epochs,
                       double learning_rate)
        = 0;
    virtual void predict(double* inputs, int inputs_size, double* outputs, int outputs_size) const = 0;

  protected:
    double* weights;
    int weights_count;

    // Activation function and its derivative
    inline double _sigmoid(double x) const { return 1 / (1 + exp(-x)); }
    inline double _sigmoid_derivative(double x) const { return x * (1 - x); }

    void debuglog(std::string msg) {
        std::fstream fout;
        fout.open("debug.txt", std::ios::out | std::ios::app);
        fout << msg << '\n';
        fout.close();
    }
};

#endif