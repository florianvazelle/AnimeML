
#include <doctest/doctest.h>

#include <Library.hpp>
#include <LinearModel.hpp>
#include <iostream>

TEST_CASE("Linear Classification") {
    const double EPSILON = 0.1;

    const int trainingSet_Size = 3;

    const int numInputs = 2;
    double inputs[6] = {1, 1, 2, 3, 3, 3};

    const int numOutputs = 1;
    double outputs[3] = {1, 0, 1};

    const int epochs = 10000;
    const double learningRate = 0.5f;

    BaseModel* model = Library::CreateModel(Library::Flags::LINEAR_MODEL, numInputs);

    Library::Train(model,             // weights
                   trainingSet_Size,  // number of training sets
                   inputs,            // all_inputs array
                   numInputs,         // number of inputs for 1 set
                   outputs,           // all_inputs array
                   numOutputs,        // number of inputs for 1 set
                   epochs,            // number of epoch
                   learningRate       // learning rate
    );

    double* results = new double[3];
    Library::Predict(model, inputs, numInputs, results, 3);

    for (int i = 0; i < 3; i++) {
        CHECK(std::abs(results[i] - outputs[i]) < EPSILON);
        std::cout << results[i] << " == " << outputs[i] << std::endl;
    }
}