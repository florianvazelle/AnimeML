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
DLLEXPORT BaseModel* CreateModel(int flag, int weights_count, bool is_classification) {
    switch (flag) {
        case 0:
            return new LinearModel(weights_count, is_classification);
        case 1:
            // std::vector<int> layers = {2, 1}; // last member "1" is th output layer
            std::vector<unsigned int> topology = {(unsigned int)weights_count, 3, 3, 1};
            return new MLP(topology, weights_count, is_classification);
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

// Load pictures and ouputs and pass them with pointers
DLLEXPORT void LoadAsset() {
    ImageManager imageManager;
    std::vector<double> inputImagesPixels;
    std::vector<double> outputs;
    int imgSize;

    // Load assets
    imageManager.loadAsset(inputImagesPixels, outputs);
    for (unsigned i = 0; i < outputs.size(); i++) {
        std::cout << "outputs[i] : " << outputs[i] << std::endl;
    }
    

    std::cout << "inputImagesPixels.size() : " << inputImagesPixels.size() << " outputs.size() : " << outputs.size() << std::endl;

    imgSize = (int)(inputImagesPixels.size() / outputs.size());
    std::cout << "Size of one image: " << imgSize << std::endl;

    // Create a model

    // hyper parameters
    unsigned numHiddenLayers = 4;
    unsigned numHiddenNeurons = 683;

    std::vector<unsigned> topology;
    topology.push_back(imgSize);
    for (unsigned i = 0; i < numHiddenLayers; i++) {
        topology.push_back(numHiddenNeurons);
    }
    topology.push_back(1);
    
    //Debug display
    for (size_t i = 0; i < topology.size(); i++) {
        std::cout << topology[i] << std::endl;
    }
    
    MLP *model = new MLP(topology, 0, true);

    Eigen::MatrixXd inputMatrix(outputs.size(), imgSize); // A(row, col)
    Eigen::MatrixXd outputMatrix(outputs.size(), 1);

    int count = 0;
    //fill input Matrix
    for (unsigned row = 0; row < inputMatrix.rows(); row++) {
        for (unsigned col = 0; col < inputMatrix.cols(); col++) {
            //inputMatrix(row, col) = (inputImagesPixels[(row * inputMatrix.cols()) + col] / 255) * 2 - 1;
            inputMatrix(row, col) = inputImagesPixels[(row * inputMatrix.cols())];
            count++;
        }
    }

    std::cout << count << std::endl;

    std::cout << "outputs.size() : " << outputs.size() << std::endl;
    std::cout << "outputMatrix.rows() : " << outputMatrix.rows() << std::endl;

    // for (unsigned i = 0; i < outputs.size(); i++) {
    //     std::cout << "outputs[i] : " << outputs[i] << std::endl;
    // }

    //fill output Matrix
    for (unsigned row = 0; row < outputs.size(); row++) {
        outputMatrix(row, 0) = outputs[row];
        std::cout << "outputMatrix(row, 0) : " << outputMatrix(row, 0) << std::endl;
    }

    // model->predict(inputMatrix, outputMatrix);

    // for (unsigned row = 0; row < outputs.size(); row++) {
    //     //outputMatrix(row, 0) = outputs[row];
    //     std::cout << "outputMatrix(row, 0) : " << outputMatrix(row, 0) << std::endl;
    // }

    // for (unsigned i = 0; i < 10; i++) {
    //     std::cout << "first neuron weigts : " << model. << std::endl;
    // }
    //     //outputMatrix(row, 0) = outputs[row];
    //     std::cout << "outputMatrix(row, 0) : " << outputMatrix(row, 0) << std::endl;
    // }
    model->train(inputMatrix, outputMatrix, 100, 0.15);

    model->predict(inputMatrix, outputMatrix);

    DeleteModel(model);
}