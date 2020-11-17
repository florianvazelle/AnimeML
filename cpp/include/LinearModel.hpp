#ifndef LINEARMODEL_HPP
#define LINEARMODEL_HPP

#include <BaseModel.hpp>

class LinearModel : virtual public BaseModel {
  public:
    LinearModel(int weights_count, bool is_classification);

    void train(int sample_count, const double* train_inputs, int inputs_size, const double* train_outputs, int output_size, int epochs, double learning_rate);
    void predict(int sample_count, const double* inputs, int inputs_size, double* outputs, int outputs_size) const;

  private:
    double _activation(double value) const;
    double _update_weight(double old_weight, double learning_rate, double target_value, double actual_value, double entry_value) const;
};

#endif