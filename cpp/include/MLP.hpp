#ifndef MLP_HPP
#define MLP_HPP

#include <BaseModel.hpp>
#include <Neuron.hpp>

class MLP : virtual public BaseModel {
  public:
    MLP(const std::vector<unsigned> &topology, int weights_count, bool is_classification);
    void train(const Eigen::MatrixXd& train_inputs, const Eigen::MatrixXd& train_outputs, int epochs, double learning_rate);
    void predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs);
    
  private:
    std::vector<Layer> _layers; // _Layers[layerNum][neuronNum]
    double _error = 0.0;
    double _recentAverageError; // DEBUG
    double _recentAverageSmoothingFactor; // DEBUG 
    double _activation(double value) const; // to delete

    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double>& resultVals);
};

#endif