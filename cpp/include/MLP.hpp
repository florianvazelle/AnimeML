#ifndef MLP_HPP
#define MLP_HPP

#include <BaseModel.hpp>
#include <Neuron.hpp>


class MLP : virtual public BaseModel {
  public:
    MLP(std::vector<int> layers, int weights_count, bool is_classification);

    void train(const Eigen::MatrixXd& train_inputs, const Eigen::MatrixXd& train_outputs, int epochs, double learning_rate);
    void predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs) const;
    
  private:
    std::vector<int> _layers;
    std::vector<std::vector<std::vector<double>>> _weightsArray;
    double _activation(double value) const;
};

#endif