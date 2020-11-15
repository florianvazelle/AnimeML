#ifndef LINEARMODEL_HPP
#define LINEARMODEL_HPP

#include <BaseModel.hpp>

class LinearModel : virtual public BaseModel {
  public:
    LinearModel(int weights_count);

    void train(int sample_count, double* train_inputs, int inputs_size, double* train_outputs, int output_size, int epochs, double learning_rate);
    void predict(double* inputs, int inputs_size, double* outputs, int outputs_size) const;

  private:
    void _shuffle(std::vector<int>& array) const;
    double _update_weight(double old_weight, double learning_rate, double target_value, double actual_value, double entry_value) const;
};

#endif