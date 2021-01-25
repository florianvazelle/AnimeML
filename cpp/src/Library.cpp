#include <Library.hpp>

static Eigen::MatrixXd ConvertToEigenMatrix(const double* data, int x_dim, int y_dim) {
    Eigen::MatrixXd matrix_data(x_dim, y_dim);
    for (int i = 0; i < x_dim; ++i)
        matrix_data.row(i) = Eigen::VectorXd::Map(&data[i * y_dim], y_dim);
    return matrix_data;
}

/**
 * Allow to create a model pointer
 *
 * @param flag Is the type of the model
 * @param weights_count Is the number of ...
 */
DLLEXPORT BaseModel* CreateModel(int flag, int weights_count, const double* topology, int layers_count, bool is_classification) {
    switch (flag) {
        case 0:
            return new LinearModel(weights_count, is_classification);
        case 1:
            assert(weights_count == topology[0]);
            std::vector<unsigned> dest(layers_count);
            for(int i = 0; i < layers_count; i++) {
                dest[i] = (unsigned)topology[i];
            }

            return new MLP(dest, weights_count, is_classification);
    }
    //throw("Not a valid flag!");
    return nullptr;
}

/**
 * Train the neural network model through a process of trial and error.
 * Adjusting the weights each time.
 * Convert inputs and outputs in matrix
 * @param model Is the pointer to the model
 * @param sample_count Is the number of training input
 * @param train_inputs Are the training input data
 * @param inputs_size Is the size of one set of train_inputs parameter
 * (ex: train_inputs = {{2, 2}, {1, 3}} but is 1D array, thus {2, 2, 1, 3} and inputs_size = 2)
 * (note: also corresponds to the number of neurons of the first layer)
 * @param train_outputs Is the training output data output
 * @param outputs_size Like inputs_size but for train_outputs
 * @param epochs
 * @param learning_rate
 */

DLLEXPORT void Train(BaseModel* model,
                     int sample_count,
                     const double* train_inputs,
                     int inputs_size,
                     const double* train_outputs,
                     int outputs_size,
                     int epochs,
                     double learning_rate) {
    Eigen::MatrixXd inMatrix = ConvertToEigenMatrix(train_inputs, sample_count, inputs_size);
    Eigen::MatrixXd outMatrix = ConvertToEigenMatrix(train_outputs, sample_count, outputs_size);

    model->train(inMatrix, outMatrix, epochs, learning_rate);
}

/**
 * Use a trained model to predict value
 *
 * @param model Is the pointer to the trained model
 * @param inputs Are new set of entries that we want to submit to the model (for each set of input we predict a value)
 * @param inputs_size Is the size of one set of input
 * @param outputs Is the output predicted by the model, (normally empty)
 * @param outputs_size Is the size of one set of output
 */
DLLEXPORT void Predict(BaseModel* model, const int sample_count, const double* inputs, const int inputs_size, double* outputs, const int outputs_size) {
    Eigen::MatrixXd inMatrix = ConvertToEigenMatrix(inputs, sample_count, inputs_size);
    Eigen::MatrixXd outMatrix(sample_count, outputs_size);
    
    model->predict(inMatrix, outMatrix);

    // for each sample and each output 
    for (int i = 0; i < sample_count; i++) {
        for (int j = 0; j < outputs_size; j++) {
            outputs[i * outputs_size + j] = outMatrix(i, j);
        }
    }
}

DLLEXPORT double* GetWeigths(BaseModel* model) { return model->getWeigths(); }

DLLEXPORT void SaveModel(BaseModel* model, const char* path) { model->save(path); }

DLLEXPORT void LoadModel(BaseModel* model, const char* path) { model->load(path); }

/**
 * Expose Library methods.
 *
 * extern "C" specifies that the function is defined
 * elsewhere and uses the C-language calling convention.
 */
DLLEXPORT void DeleteModel(BaseModel* model) { delete model; }

DLLEXPORT int PredictImage(BaseModel* model, const char* path) {
    ImageManager imageManager;
    std::vector<double> predict_inputs;

    // Create a model
    const int inputs_size = 1024;

    // hyper parameters
    unsigned numHiddenLayers = 4;
    unsigned numHiddenNeurons = 683;

    std::vector<unsigned> topology;
    topology.push_back(inputs_size);
    for (unsigned i = 0; i < numHiddenLayers; i++) {
        topology.push_back(numHiddenNeurons);
    }
    topology.push_back(1);

    Eigen::MatrixXd predictInputMatrix, predictOutputMatrix;

    imageManager.load(path, predict_inputs);
    predictInputMatrix = ConvertToEigenMatrix(predict_inputs.data(), 1, inputs_size);
    predictOutputMatrix = predictInputMatrix;  // just for the same size

    model->predict(predictInputMatrix, predictOutputMatrix);

    return predictOutputMatrix(0, 0);
}

// Load pictures and ouputs and pass them with pointers
DLLEXPORT void LoadAsset(const char* path) {
    ImageManager imageManager;
    std::vector<double> inputImagesPixels, predict_inputs;
    std::vector<double> train_outputs, predict_outputs;

    // Load assets
    imageManager.loadAsset(inputImagesPixels, train_outputs);

    // Create a model
    const int sample_count = train_outputs.size();
    const int inputs_size = (int)(inputImagesPixels.size() / train_outputs.size());

    // hyper parameters
    unsigned numHiddenLayers = 4;
    unsigned numHiddenNeurons = 683;

    std::vector<unsigned> topology;
    topology.push_back(inputs_size);
    for (unsigned i = 0; i < numHiddenLayers; i++) {
        topology.push_back(numHiddenNeurons);
    }
    topology.push_back(1);
    
    //Debug display
    std::cout << "topology : { ";
    for (size_t i = 0; i < topology.size() - 1; i++) {
        std::cout << topology[i] << ", ";
    }
    std::cout << topology.back() << " };\n";

    MLP* model = new MLP(topology, 0, true);
    model->load("test_model copy.json");

    Eigen::MatrixXd inputMatrix = ConvertToEigenMatrix(inputImagesPixels.data(), sample_count, inputs_size);
    Eigen::MatrixXd outputMatrix = ConvertToEigenMatrix(train_outputs.data(), sample_count, 1);

    std::cout << "sample_count : " << sample_count << "\n";
    std::cout << "inputs_size : " << inputs_size << "\n";

    model->train(inputMatrix, outputMatrix, 1000, 0.1);

    // ****** Predict ******

    Eigen::MatrixXd predictInputMatrix, predictOutputMatrix;

    if (strcmp(path, "") != 0) {
        imageManager.load(path, predict_inputs);
        predictInputMatrix = ConvertToEigenMatrix(predict_inputs.data(), 1, inputs_size);
        predictOutputMatrix = predictInputMatrix; // just for the same size
    } else {
        predictInputMatrix = inputMatrix;
        predictOutputMatrix = outputMatrix;
    }

    model->predict(predictInputMatrix, predictOutputMatrix);

    DeleteModel(model);
}