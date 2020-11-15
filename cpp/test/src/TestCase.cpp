
#include <doctest/doctest.h>

#include <CSVFile.hpp>
#include <Library.hpp>
#include <LinearModel.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

static double myrand() { return ((double)rand()) / ((double)RAND_MAX); }

/* Helper to train model and test it on the training input */
static void CheckModel(int flag, int sample_count, double* inputs, int numInputs, double* outputs, int numOutputs) {
    const double EPSILON = 0.1;

    const int epochs = 10000;
    const double learningRate = 0.75;

    Library lib;
    BaseModel* model = lib.CreateModel(flag, numInputs);

    lib.Train(model,         // weights
              sample_count,  // number of training sets
              inputs,        // all_inputs array
              numInputs,     // number of inputs for 1 set
              outputs,       // all_inputs array
              numOutputs,    // number of inputs for 1 set
              epochs,        // number of epoch
              learningRate   // learning rate
    );

    std::vector<double> results(sample_count);
    lib.Predict(model, inputs, numInputs, results.data(), sample_count);

    for (int i = 0; i < 3; i++) {
        CHECK(std::abs(results[i] - outputs[i]) < EPSILON);
        std::cout << results[i] << " == " << outputs[i] << std::endl;
    }

    lib.DeleteModel(model);
}

static void LinearSimple(int flag) {
    const int sample_count = 3;

    const int numInputs = 2;
    double inputs[6] = {1, 1, 2, 3, 3, 3};

    const int numOutputs = 1;
    double outputs[3] = {1, -1 - 1};

    CheckModel(flag, sample_count, inputs, numInputs, outputs, numOutputs);
}

static void LinearMultiple(int flag) {
    const int sample_count = 100;

    const int numInputs = 2;
    std::vector<double> inputs(200);
    for (int i = 0; i < inputs.size(); i++) {
        inputs[i] = myrand() * 0.9;
        if (i < 100) {
            inputs[i] += 1;
        } else {
            inputs[i] += 2;
        }
    }

    const int numOutputs = 1;
    std::vector<double> outputs(100);
    for (int i = 0; i < outputs.size(); i++) {
        outputs[i] = 1;
        if (i >= 50) {
            outputs[i] *= -1.0;
        }
    }

    CheckModel(flag, sample_count, inputs.data(), numInputs, outputs.data(), numOutputs);
}

static void XOR(int flag) {
    const int sample_count = 4;

    const int numInputs = 2;
    double inputs[8] = {1, 0, 0, 1, 0, 0, 1, 1};

    const int numOutputs = 1;
    double outputs[4] = {1, 1, -1 - 1};

    CheckModel(flag, sample_count, inputs, numInputs, outputs, numOutputs);
}

static void Cross(int flag) {
    const int sample_count = 500;

    const int numInputs = 2;
    const int numOutputs = 1;

    std::vector<double> inputs(1000);
    std::vector<double> outputs(500);
    for (int i = 0, j = 0; i < outputs.size(); i++, j += 2) {
        inputs[j] = 2.0 * myrand() - 1.0;
        inputs[j + 1] = 2.0 * myrand() - 1.0;

        outputs[i] = (std::abs(inputs[j]) <= 0.3 || std::abs(inputs[j + 1]) <= 0.3) ? 1 : -1;
    }

    CheckModel(flag, sample_count, inputs.data(), numInputs, outputs.data(), numOutputs);
}

static void MultiLinear3Classes(int flag) {
    const int sample_count = 500;

    const int numInputs = 2;
    const int numOutputs = 3;

    std::vector<double> inputs(1000);
    std::vector<double> outputs(1500);

    for (int i = 0, j = 0; i < outputs.size(); i++, j += 2) {
        inputs[j] = 2.0 * myrand() - 1.0;
        inputs[j + 1] = 2.0 * myrand() - 1.0;

        std::vector<double> res;
        if (-inputs[j] - inputs[j + 1] - 0.5 > 0 && inputs[j + 1] < 0 && inputs[j] - inputs[j + 1] - 0.5 < 0) {
            res = {1, 0, 0};
        } else if (-inputs[j] - inputs[j + 1] - 0.5 < 0 && inputs[j + 1] > 0 && inputs[j] - inputs[j + 1] - 0.5 < 0) {
            res = {0, 1, 0};
        } else if (-inputs[j] - inputs[j + 1] - 0.5 < 0 && inputs[j + 1] < 0 && inputs[j] - inputs[j + 1] - 0.5 > 0) {
            res = {0, 0, 1};
        } else {
            res = {0, 0, 0};
        };
        outputs.insert(outputs.begin() + i, res.begin(), res.end());
    }

    CheckModel(flag, sample_count, inputs.data(), numInputs, outputs.data(), numOutputs);
}

static void MultiCross(int flag) {
    const int sample_count = 1000;

    const int numInputs = 2;
    const int numOutputs = 3;

    std::vector<double> inputs(2000);
    std::vector<double> outputs(3000);
    for (int i = 0, j = 0; i < outputs.size(); i++, j += 2) {
        inputs[j] = 2.0 * myrand() - 1.0;
        inputs[j + 1] = 2.0 * myrand() - 1.0;

        std::vector<double> res;
        if (std::abs(std::fmod(inputs[j], 0.5)) <= 0.25 && std::abs(std::fmod(inputs[j + 1], 0.5)) > 0.25) {
            res = {1, 0, 0};
        } else if (std::abs(std::fmod(inputs[j], 0.5)) > 0.25 && std::abs(std::fmod(inputs[j + 1], 0.5)) <= 0.25) {
            res = {0, 1, 0};
        } else {
            res = {0, 0, 1};
        };
        outputs.insert(outputs.begin() + i, res.begin(), res.end());
    }

    CheckModel(flag, sample_count, inputs.data(), numInputs, outputs.data(), numOutputs);
}

TEST_CASE("Classification") {
    // For all Flags
    for (int i = 0; i < 1; i++) {
        SUBCASE("Linear Simple") { LinearSimple(i); }
        SUBCASE("Linear Multiple") { LinearMultiple(i); }
        SUBCASE("XOR") { XOR(i); }
        SUBCASE("Cross") { Cross(i); }
        SUBCASE("Multi Linear 3 classes") { MultiLinear3Classes(i); }
        SUBCASE("Multi Cross") { MultiCross(i); }
    }
}

static void LinearSimple2D(int flag) {
    const int sample_count = 2;

    const int numInputs = 1;
    double inputs[2] = {1, 2};

    const int numOutputs = 1;
    double outputs[2] = {2, 3};

    CheckModel(flag, sample_count, inputs, numInputs, outputs, numOutputs);
}

static void NonLinearSimple2D(int flag) {
    const int sample_count = 3;

    const int numInputs = 1;
    double inputs[3] = {1, 2, 3};

    const int numOutputs = 1;
    double outputs[3] = {2, 3, 2.5};

    CheckModel(flag, sample_count, inputs, numInputs, outputs, numOutputs);
}

static void LinearSimple3D(int flag) {
    const int sample_count = 3;

    const int numInputs = 2;
    double inputs[6] = {1, 1, 2, 2, 3, 1};

    const int numOutputs = 1;
    double outputs[3] = {2, 3, 2.5};

    CheckModel(flag, sample_count, inputs, numInputs, outputs, numOutputs);
}

static void LinearTricky3D(int flag) {
    const int sample_count = 3;

    const int numInputs = 2;
    double inputs[6] = {1, 1, 2, 2, 3, 3};

    const int numOutputs = 1;
    double outputs[3] = {1, 2, 3};

    CheckModel(flag, sample_count, inputs, numInputs, outputs, numOutputs);
}

static void NonLinearSimple3D(int flag) {
    const int sample_count = 4;

    const int numInputs = 2;
    double inputs[8] = {1, 0, 0, 1, 1, 1, 0, 0};

    const int numOutputs = 1;
    double outputs[4] = {2, 1, -2, -1};

    CheckModel(flag, sample_count, inputs, numInputs, outputs, numOutputs);
}

// TEST_CASE("Regression") {
//     // For all Flags
//     for (int i = 0; i < 1; i++) {
//         SUBCASE("Linear Simple 2D") { LinearSimple2D(i); }
//         SUBCASE("Non Linear Multiple") { NonLinearMultiple(i); }
//         SUBCASE("Linear Simple 3D") { LinearSimple3D(i); }
//         SUBCASE("Linear Tricky 3D") { LinearTricky3D(i); }
//         SUBCASE("Non Linear Simple 3D") { NonLinearSimple3D(i); }
//     }
// }