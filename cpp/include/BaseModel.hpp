#ifndef BASEMODEL_HPP
#define BASEMODEL_HPP

#include <math.h> /* exp */
#include <fstream>
#include <vector>

class BaseModel {
  public:
    BaseModel(int weights_count, bool is_classification) : weights_count(weights_count), is_classification(is_classification) {}
    BaseModel(std::string filename = "model.csv");
    virtual ~BaseModel() { delete[] weights; }

    inline double* getWeigths() const { return weights; }

    virtual void train(int sample_count, const double* train_inputs, int inputs_size, const double* train_outputs, int output_size, int epochs, double learning_rate) = 0;
    virtual void predict(int sample_count, const double* inputs, int inputs_size, double* outputs, int outputs_size) const = 0;

  protected:
    double* weights;
    int weights_count;
    bool is_classification;

    // The Sigmoid function, which describes an S shaped curve.
    // We pass the weighted sum of the inputs through this function to
    // normalise them between 0 and 1.
    inline double _sigmoid(double x) const { return 1 / (1 + exp(-x)); }

    // The derivative of the Sigmoid function.
    // This is the gradient of the Sigmoid curve.
    // It indicates how confident we are about the existing weight
    inline double _sigmoid_derivative(double x) const { return x * (1 - x); }

    void debuglog(std::string msg);

    // npl is an array of number of weight by layer (for example: npl = [4, 3, 3,
    // 2], the first layer have 4 weigth, the second have 3 weight ...)
    void save(std::vector<int> npl, std::string filename = "model.csv") const;
};

#endif