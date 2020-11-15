#include <Library.hpp>

#include <LinearModel.hpp>

#ifdef WIN32
#    define dllexport __declspec(dllexport)
#else
#    define dllexport
#endif

extern "C"
{
    /**
     * Allow to create a model pointer
     *
     * @param flag Is the type of the model
     * @param weights_count Is the number of ...
     */
    dllexport BaseModel* Library::CreateModel(int flag, int weights_count) {
        switch (flag) {
            case LINEAR_MODEL:
                return new LinearModel(weights_count);
                // case 1:
                //     return new MLP{};
        }
        throw("Not a valid flag!");
    };

    /**
     * Train a model with inputs and outputs
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
    dllexport void Library::Train(BaseModel* model, int sample_count, double* train_inputs, int inputs_size, double* train_outputs, int outputs_size,
                                  int epochs, double learning_rate) {
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
    dllexport void Library::Predict(BaseModel* model, double* inputs, int inputs_size, double* outputs, int outputs_size) {
        model->predict(inputs, inputs_size, outputs, outputs_size);
    };

    dllexport double* Library::GetWeigths(BaseModel* model) { return model->getWeigths(); }

    /**
     * Allow to delete model pointer to avoid leak
     *
     * @param model Is the pointer to the model
     */
    dllexport void Library::DeleteModel(BaseModel* model) { delete model; };
}