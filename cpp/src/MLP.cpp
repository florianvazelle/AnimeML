#include <MLP.hpp>

#include <Utils.hpp>
#include <iostream>

MLP::MLP(const std::vector<unsigned> &topology, int weights_count, bool is_classification) : BaseModel(weights_count, is_classification) {
    // **************** not used *********************
    weights = new double[weights_count + 1];
    // Init all weights and biases between -1.0 and 1.0
    for (int i = 0; i < weights_count + 1; i++) {
        weights[i] = ml::rand(-1, 1);
    }
    // **************** end *********************

    unsigned int numLayers = (unsigned)topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        _layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        // We have made a new Layer, now fill it with neurons, and
        // add a bias neuron to the layer:
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            _layers.back().push_back(Neuron(numOutputs, neuronNum)); // ".back" last layer of vector
            //std::cout << "Made a Neuron ! index : " << neuronNum << " NumOutput: " << numOutputs << std::endl;
        }
        // if it's a bias neuron, it must be initialize with 1 in outputval ! (otherwize it's never initialized)
        _layers.back().back().setOutputVal(1.0);
    }
}

// pass on each samples and send results in outputs
void MLP::predict(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs){
    // assert if there is the same amount of samples in the inputs/outputs matrix
    assert(inputs.rows() == outputs.rows());  // or maybe resize outputs

    // for each sample
    for (int i = 0; i < inputs.rows(); i++) {
        std::vector<double> matrixInputsVector;
        for (int k = 0; k < inputs.cols(); k++) {
            matrixInputsVector.push_back(inputs(i,k)); // à vérifier
        }
        feedForward(matrixInputsVector);

        std::vector<double> res;
        getResults(res);

        // std::cout << "res : " << std::endl;

        for(auto p : res) {
            std::cout << "Output Val: " << p << std::endl;
        }
        
        for(int k = 0; k < res.size(); k++) {
            outputs(i, k) = res[k];
            //std::cout << "outputs(i, k): " << outputs(i, k) << std::endl;
        }
    }
}

// fill a vector with results of the Net
void MLP::getResults(std::vector<double>& resultVals){
    resultVals.clear();
    unsigned int lastLayerSize = (unsigned)_layers[_layers.size() - 1].size(); // size with bias
    
    //std::cout << "last layer size : " << lastLayerSize << std::endl;
    for (unsigned i = 0; i < lastLayerSize - 1; ++i) { // - 1 for bias
        resultVals.push_back(_layers[_layers.size() - 1][i].getOutputVal());
    }
}

void MLP::train(const Eigen::MatrixXd& train_inputs, const Eigen::MatrixXd& train_outputs, int epochs, double learning_rate) {
    const size_t sample_count = train_inputs.rows();
    const size_t inputs_size = train_inputs.cols();
    const size_t outputs_size = train_outputs.cols();
    
    if (is_classification) {
        std::vector<int> trainingSetOrder(sample_count);

        for (int i = 0; i < trainingSetOrder.size(); i++) {
            trainingSetOrder[i] = i;
        }
        
        //Eigen::MatrixXd activation(sample_count, outputs_size);

        // Iterate with epochs
        for (int i = 0; i < epochs; i++) {
            std::cout << "Turn : " << i << std::endl;
            //predict(train_inputs, activation);

            // shuffle the training set
            ml::random_shuffle<int>(trainingSetOrder);

            // for each training set
            for (int j = 0; j < trainingSetOrder.size(); j++) {
                // chaque example -> forward -> backward
                std::vector<double> matrixInputsVector;
                std::vector<double> results;

                for (int k = 0; k < train_inputs.cols(); k++) {
                    matrixInputsVector.push_back(train_inputs(trainingSetOrder[j],k));
                }
                feedForward(matrixInputsVector);
                //feedForward(trainingSetOrder[j]);

                // debug
                // std::cout << "Training set num: " << trainingSetOrder[j] << std::endl;
                getResults(results);
                // for(unsigned i = 0; i < results.size(); i++)
                // {
                //     std::cout << results[i] << " ";
                // }
                // std::cout << "\n" << std::endl;

                std::vector<double> matrixOutputsVector;
                for (int k = 0; k < train_outputs.cols(); k++) {
                    matrixOutputsVector.push_back(train_outputs(trainingSetOrder[j],k));
                }
                backProp(matrixOutputsVector); // pwoblem
            }
        }
    }
    // Loop 
        // Get new input data and feed it forward:
        // -> feedForward(inputVals);

        // Collect the net's actual results:
        //getResults(resultVals);

        // Train the net with what the outputs should have been
        // -> backProp(targetVals);

        // Report how well the training is working
        // myNet.getRecentAverageError()

        //getResults(resultVals);
}

void MLP::feedForward(const std::vector<double> &inputVals){
    assert(inputVals.size() == _layers[0].size() - 1);

    // assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        _layers[0][i].setOutputVal(inputVals[i]);
        // std::cout << "set Input Val: " << _layers[0][i].getOutputVal() << std::endl;
    }

    // Forward propagate
    for (unsigned layerNum = 1; layerNum < _layers.size(); ++layerNum) {
        Layer &prevLayer = _layers[layerNum - 1];
        for (unsigned n = 0; n < _layers[layerNum].size() - 1; ++n) {
            _layers[layerNum][n].feedForward(prevLayer);
            // std::cout << "layer: " << layerNum << " Neuron: " << n << " Output val: " << _layers[layerNum][n].getOutputVal() << std::endl;
        }
    }
}

void MLP::backProp(const std::vector<double> &targetVals) {

    // Calculate Overall net error (RMS of output neuron errors) "Root Mean Square Error" 
    Layer &outputLayer = _layers.back();
    _error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        // compute difference between real vs expected value
        double delta = targetVals.at(n) - outputLayer.at(n).getOutputVal();
        _error += delta * delta;
    }
    _error /= outputLayer.size() - 1;
    _error = sqrt(_error); // RMS

    // Implement a recent average measurement: DEBUG !!
    _recentAverageError = 
        ( _recentAverageError * _recentAverageSmoothingFactor + _error)
        / (_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers
    for (unsigned layerNum = (unsigned)_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = _layers[layerNum]; // documentation purpose (can be optimize)
        Layer &nextLayer = _layers[layerNum + 1]; // documentation purpose (can be optimize)

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
        // update connection weights
    // Loop from the last layer to the second
    for (unsigned layerNum = (unsigned)_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = _layers[layerNum];
        Layer &prevLayer = _layers[layerNum - 1];
        
        // Loop from the first neuron to the last
        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

// not used (the activation fonciton is in Neuron class)
double MLP::_activation(double value) const {
    return 0;
}

// ****** Save & Load Stuff ******

void MLP::save(const char* path) const {
    StringBuffer s;
    Writer<StringBuffer> writer(s);

    writer.StartObject();
    writer.Key("layers");

    writer.StartArray();
    for (const Layer& layer : _layers) {
        writer.StartArray();
        for (const Neuron& neuron : layer) {
            writer.StartObject();
            writer.Key("idx");
            writer.Uint(neuron._myIndex);
            writer.Key("outputWeights");
            writer.StartArray();
            for (const Connection& connection : neuron._outputWeights) {
                writer.StartObject();
                writer.Key("weight");
                writer.Double(connection.weight);
                writer.Key("deltaWeight");
                writer.Double(connection.deltaWeight);
                writer.EndObject();
            }
            writer.EndArray();
            writer.EndObject();
        }
        writer.EndArray();
    }
    writer.EndArray();

    writer.EndObject();

    std::fstream fout;
    fout.open(path, std::ios::out | std::ios::trunc);
    fout << s.GetString();
    fout.close();
}

void MLP::load(const char* path) {
    _layers.clear();

    std::setlocale(LC_NUMERIC, "C");
    std::ifstream in(path);
    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    Document document;
    if (document.Parse(json.c_str()).HasParseError()) return;

    assert(document.IsObject());
    assert(document.HasMember("layers"));

    const Value& jsonLayers = document["layers"];  // Using a reference for consecutive access is handy and faster.
    assert(jsonLayers.IsArray());

    Layer layer;
    for (auto& l : jsonLayers.GetArray()) {
        if (l.IsObject()) {
            const Value& n = l["outputWeights"];

            Neuron neuron(0, l["idx"].GetUint());
            for (auto& c : n.GetArray()) {
                Connection connection;
                for (auto& w : c.GetArray()) {
                    if (w.IsObject()) {
                        connection.weight = w["weight"].GetDouble();
                        connection.deltaWeight = w["deltaWeight"].GetDouble();
                    }
                }

                neuron._outputWeights.push_back(connection);
            }

            layer.push_back(neuron);
        }
    }

    _layers.push_back(layer);
}