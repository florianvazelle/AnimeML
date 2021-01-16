#ifndef LINEARMODEL_HPP
#define LINEARMODEL_HPP

#include <BaseModel.hpp>

class LinearModel : virtual public BaseModel {
  public:
    LinearModel(int weights_count, bool is_classification);

    void train(const Eigen::MatrixXd& train_inputs, const Eigen::MatrixXd& train_outputs, int epochs, double learning_rate);
    void predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs);

  private:
    double _activation(double value) const;
};

#endif