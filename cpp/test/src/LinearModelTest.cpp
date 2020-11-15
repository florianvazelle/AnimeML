
#include <doctest/doctest.h>

#include <CSVFile.hpp>
#include <Library.hpp>
#include <LinearModel.hpp>
#include <iostream>
#include <vector>

TEST_CASE("Simple Test 1") {
    const double EPSILON = 0.1;

    const int trainingSet_Size = 3;

    const int numInputs = 2;
    double inputs[6] = {1, 1, 2, 3, 3, 3};

    const int numOutputs = 1;
    double outputs[3] = {1, 0, 1};

    const int epochs = 10000;
    const double learningRate = 0.5;

    Library lib;
    BaseModel* model = lib.CreateModel(Library::Flags::LINEAR_MODEL, numInputs);

    lib.Train(model,             // weights
              trainingSet_Size,  // number of training sets
              inputs,            // all_inputs array
              numInputs,         // number of inputs for 1 set
              outputs,           // all_inputs array
              numOutputs,        // number of inputs for 1 set
              epochs,            // number of epoch
              learningRate       // learning rate
    );

    std::vector<double> results(trainingSet_Size);
    lib.Predict(model, inputs, numInputs, results.data(), trainingSet_Size);

    for (int i = 0; i < 3; i++) {
        CHECK(std::abs(results[i] - outputs[i]) < EPSILON);
        std::cout << results[i] << " == " << outputs[i] << std::endl;
    }

    lib.DeleteModel(model);
}

// TEST_CASE("Wine Quality") {
//     const double EPSILON = 0.1;

//     const int numInputs = 11;
//     const int numOutputs = 1;

//     std::vector<double> train_inputs, train_outputs;
//     std::vector<double> predict_inputs, predict_outputs;

//     CSVFile train_file(DATA_PATH "/CSV/winequality-train.csv");
//     train_file.loadAsset(train_inputs, numInputs, train_outputs, numOutputs);

//     CSVFile predict_file(DATA_PATH "/CSV/winequality-predict.csv");
//     predict_file.loadAsset(predict_inputs, numInputs, predict_outputs, numOutputs);

//     const int trainingSet_Size = train_inputs.size() / numInputs;

//     const int epochs = 10000;
//     const double learningRate = 1;

//     BaseModel* model = Library::CreateModel(Library::Flags::LINEAR_MODEL, numInputs);

//     Library::Train(model,                 // weights
//                    trainingSet_Size,      // number of training sets
//                    train_inputs.data(),   // all_inputs array
//                    numInputs,             // number of inputs for 1 set
//                    train_outputs.data(),  // all_inputs array
//                    numOutputs,            // number of inputs for 1 set
//                    epochs,                // number of epoch
//                    learningRate           // learning rate
//     );

//     double* results = new double[3];
//     Library::Predict(model, predict_inputs.data(), numInputs, results, 3);

//     for (int i = 0; i < predict_outputs.size(); i++) {
//         CHECK(std::abs(results[i] - predict_outputs[i]) < EPSILON);
//         std::cout << results[i] << " == " << predict_outputs[i] << std::endl;
//     }
// }