#include <Library.hpp>

#include <LinearModel.hpp>

#ifdef WIN32
#    define dllexport __declspec(dllexport)
#else
#    define dllexport
#endif

/**
 * Allow to create a model pointer
 *
 * @param flag Is the type of the model
 * @param weights_count Is the number of ...
 */
BaseModel* Library::CreateModel(int flag, int weights_count, bool is_classification) {
    switch (flag) {
        case LINEAR_MODEL:
            return new LinearModel(weights_count, is_classification);
            // case 1:
            //     return new MLP{};
    }
    throw("Not a valid flag!");
};

/**
 * Train the neural network model through a process of trial and error.
 * Adjusting the weights each time.
 *
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
void Library::Train(BaseModel* model,
                    int sample_count,
                    const double* train_inputs,
                    int inputs_size,
                    const double* train_outputs,
                    int outputs_size,
                    int epochs,
                    double learning_rate) {
    model->train(sample_count, train_inputs, inputs_size, train_outputs, outputs_size, epochs, learning_rate);
};

/**
 * Use a trained model to predict value
 *
 * @param model Is the pointer to the trained model
 * @param inputs Are new set of entries that we want to submit to the model (for each set of input we predict a value)
 * @param inputs_size Is the size of one set of input
 * @param outputs Is the output predicted by the model, (normally empty)
 * @param outputs_size Is the size of one set of output
 */
void Library::Predict(BaseModel* model, int sample_count, const double* inputs, int inputs_size, double* outputs, int outputs_size) {
    model->predict(sample_count, inputs, inputs_size, outputs, outputs_size);
};

double* Library::GetWeigths(BaseModel* model) { return model->getWeigths(); }

/**
 * Allow to delete model pointer to avoid leak
 *
 * @param model Is the pointer to the model
 */
void Library::DeleteModel(BaseModel* model) { delete model; };

/**
 * Expose Library methods.
 *
 * extern "C" specifies that the function is defined
 * elsewhere and uses the C-language calling convention.
 */
extern "C"
{
    dllexport BaseModel* CreateModel(int flag, int weights_count, bool is_classification) { return Library::CreateModel(flag, weights_count, is_classification); };
    dllexport void
    Train(BaseModel* model, int sample_count, const double* train_inputs, int inputs_size, const double* train_outputs, int outputs_size, int epochs, double learning_rate) {
        Library::Train(model, sample_count, train_inputs, inputs_size, train_outputs, outputs_size, epochs, learning_rate);
    };
    dllexport void Predict(BaseModel* model, int sample_count, const double* inputs, int inputs_size, double* outputs, int outputs_size) {
        Library::Predict(model, sample_count, inputs, inputs_size, outputs, outputs_size);
    };
    dllexport double* GetWeigths(BaseModel* model) { return Library::GetWeigths(model); }
    dllexport void DeleteModel(BaseModel* model) { Library::DeleteModel(model); };
}