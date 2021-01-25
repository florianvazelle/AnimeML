#ifndef BASEMODEL_HPP
#define BASEMODEL_HPP

#include <math.h> /* exp */
#include <Eigen/Dense>
#include <fstream>
#include <vector>

class BaseModel {
  public:
    BaseModel(int weights_count, bool is_classification) : weights_count(weights_count), is_classification(is_classification) {}
    virtual ~BaseModel() { delete[] weights; }

    inline double* getWeigths() const { return weights; }

    virtual void train(const Eigen::MatrixXd& train_inputs, const Eigen::MatrixXd& train_outputs, int epochs, double learning_rate) = 0;
    virtual void predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs) = 0;

    // npl is an array of number of weight by layer (for example: npl = [4, 3, 3,
    // 2], the first layer have 4 weigth, the second have 3 weight ...)
    virtual void save(const char* path = "./model.csv") const;
    virtual void load(const char* path = "./model.csv");

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
};

#endif