
#include <doctest/doctest.h>

#include <Library.hpp>
#include <LinearModel.hpp>
#include <Utils.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

static auto seed = unsigned(std::time(0));

// to shuffle is the same order at runtime
static void _no_random_shuffle(std::vector<double>& vec) {
    std::srand(seed);
    ml::random_shuffle<double>(vec);
}

static bool CheckModel(const int flag,
                       const bool is_classification,
                       const int train_sample_count,
                       const int predict_sample_count,
                       const std::vector<double>& train_inputs,
                       const std::vector<double>& train_outputs,
                       const std::vector<double>& predict_inputs,
                       const std::vector<double>& predict_outputs,
                       const int epochs = 1000,
                       const double learning_rate = 0.1) {
    const double EPSILON = 0.1;

    const int input_size = train_inputs.size() / train_sample_count;
    const int output_size = train_outputs.size() / train_sample_count;

    BaseModel* model = CreateModel(flag, input_size, is_classification);

    Train(model,                 // weights
          train_sample_count,    // number of training sets
          train_inputs.data(),   // all_inputs array
          input_size,            // number of inputs for 1 set
          train_outputs.data(),  // all_inputs array
          output_size,           // number of inputs for 1 set
          epochs,                // number of epoch
          learning_rate          // learning rate
    );

    std::vector<double> results(predict_sample_count * output_size);
    Predict(model, predict_sample_count, predict_inputs.data(), input_size, results.data(), output_size);

    bool valid = true;
    for (int i = 0; i < predict_sample_count; i++) {
        valid = valid && ml::double_equals(results[i], predict_outputs[i]);
        // if (!ml::double_equals(results[i], predict_outputs[i])) {
        //     std::cout << std::setprecision(5) << results[i] << " == " << std::setprecision(5) << predict_outputs[i] << "\n";
        // }
    }

    DeleteModel(model);

    return valid;
}

static bool CheckModelWithSameTrainPredict(const int flag,
                                           const bool is_classification,
                                           const int sample_count,
                                           const std::vector<double>& inputs,
                                           const std::vector<double>& outputs,
                                           const int epochs = 1000,
                                           const double learning_rate = 0.1) {
    return CheckModel(flag, is_classification, sample_count, sample_count, inputs, outputs, inputs, outputs, epochs, learning_rate);
}

static bool CheckModelWithSplitTrainPredict(const int flag,
                                            const bool is_classification,
                                            const int sample_count,
                                            const int predict_sample_count,
                                            std::vector<double>& inputs,
                                            std::vector<double>& outputs,
                                            const int epochs = 1000,
                                            const double learning_rate = 0.1) {
    _no_random_shuffle(inputs);
    _no_random_shuffle(outputs);

    std::vector<double> train_inputs(inputs.begin(), inputs.end() - predict_sample_count);
    std::vector<double> train_outputs(outputs.begin(), outputs.end() - predict_sample_count);

    std::vector<double> predict_inputs(inputs.begin() + sample_count - predict_sample_count, inputs.end());
    std::vector<double> predict_outputs(outputs.begin() + sample_count - predict_sample_count, outputs.end());

    return CheckModel(flag, is_classification, sample_count - predict_sample_count, predict_sample_count, train_inputs, train_outputs, predict_inputs, predict_outputs, epochs,
                      learning_rate);
}

static void LinearSimple(int flag) {
    const int sample_count = 3;

    std::vector<double> inputs({
        1, 1,  // 1st
        2, 3,  // 2nd
        3, 3   // 3th
    });
    std::vector<double> outputs({
        1,   // 1st
        -1,  // 2nd
        -1   // 3th
    });

    CHECK(CheckModelWithSameTrainPredict(flag, true, sample_count, inputs, outputs, 100, 0.5));
}

static void LinearMultiple(int flag) {
    const int sample_count = 100;

    std::vector<double> inputs(200);
    std::generate(inputs.begin(), inputs.end(), [i = 0]() mutable { return (ml::rand() * 0.9) + ((i++ < 100) ? 1 : 2); });

    std::vector<double> outputs(100);
    std::generate(outputs.begin(), outputs.end(), [j = 0]() mutable { return (j++ < 50) ? 1 : -1; });

    CHECK(CheckModelWithSameTrainPredict(flag, true, sample_count, inputs, outputs));
}

static void XOR(int flag) {
    const int sample_count = 4;

    std::vector<double> inputs({1, 0, 0, 1, 0, 0, 1, 1});
    std::vector<double> outputs({1, 1, -1, -1});

    CHECK(!CheckModelWithSameTrainPredict(flag, true, sample_count, inputs, outputs));
}

static void Cross(int flag) {
    const int sample_count = 500;

    std::vector<double> inputs(1000);
    std::vector<double> outputs(500);
    for (int i = 0, j = 0; i < sample_count; i++, j += 2) {
        inputs[j] = ml::rand(-1, 1);
        inputs[j + 1] = ml::rand(-1, 1);

        outputs[i] = (std::abs(inputs[j]) <= 0.3 || std::abs(inputs[j + 1]) <= 0.3) ? 1 : -1;
    }

    CHECK(!CheckModelWithSameTrainPredict(flag, true, sample_count, inputs, outputs));
}

static void MultiLinear3Classes(int flag) {
    const int sample_count = 500;

    std::vector<double> inputs(1000);
    std::vector<double> outputs(1500);

    for (int i = 0, j = 0, k = 0; i < sample_count; i++, j += 2, k += 3) {
        inputs[j] = ml::rand(-1, 1);
        inputs[j + 1] = ml::rand(-1, 1);

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
        outputs.insert(outputs.begin() + k, res.begin(), res.end());
    }

    CHECK(CheckModelWithSameTrainPredict(flag, true, sample_count, inputs, outputs));
}

static void MultiCross(int flag) {
    const int sample_count = 1000;

    std::vector<double> inputs(2000);
    std::vector<double> outputs(3000);
    for (int i = 0, j = 0, k = 0; i < sample_count; i++, j += 2, k += 3) {
        inputs[j] = ml::rand(-1, 1);
        inputs[j + 1] = ml::rand(-1, 1);

        std::vector<double> res;
        if (std::abs(std::fmod(inputs[j], 0.5)) <= 0.25 && std::abs(std::fmod(inputs[j + 1], 0.5)) > 0.25) {
            res = {1, 0, 0};
        } else if (std::abs(std::fmod(inputs[j], 0.5)) > 0.25 && std::abs(std::fmod(inputs[j + 1], 0.5)) <= 0.25) {
            res = {0, 1, 0};
        } else {
            res = {0, 0, 1};
        };
        outputs.insert(outputs.begin() + k, res.begin(), res.end());
    }

    CHECK(!CheckModelWithSameTrainPredict(flag, true, sample_count, inputs, outputs));
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

    std::vector<double> inputs({1, 2});
    std::vector<double> outputs({2, 3});

    CHECK(CheckModelWithSameTrainPredict(flag, false, sample_count, inputs, outputs));
}

static void NonLinearSimple2D(int flag) {
    const int sample_count = 3;

    std::vector<double> inputs({1, 2, 3});
    std::vector<double> outputs({2, 3, 2.5});

    CHECK(!CheckModelWithSameTrainPredict(flag, false, sample_count, inputs, outputs));
}

static void LinearSimple3D(int flag) {
    const int sample_count = 3;

    std::vector<double> inputs({1, 1, 2, 2, 3, 1});
    std::vector<double> outputs({2, 3, 2.5});

    CHECK(CheckModelWithSameTrainPredict(flag, false, sample_count, inputs, outputs));
}

static void LinearTricky3D(int flag) {
    const int sample_count = 3;

    std::vector<double> inputs({1, 1, 2, 2, 3, 3});
    std::vector<double> outputs({1, 2, 3});

    CHECK(CheckModelWithSameTrainPredict(flag, false, sample_count, inputs, outputs));
}

static void NonLinearSimple3D(int flag) {
    const int sample_count = 4;

    std::vector<double> inputs({1, 0, 0, 1, 1, 1, 0, 0});
    std::vector<double> outputs({2, 1, -2, -1});

    CHECK(!CheckModelWithSameTrainPredict(flag, false, sample_count, inputs, outputs));
}

TEST_CASE("Regression") {
    // For all Flags
    // std::cout << " ----------- \n";

    for (int i = 0; i < 1; i++) {
        SUBCASE("Linear Simple 2D") { LinearSimple2D(i); }
        SUBCASE("Non Linear Simple 2D") { NonLinearSimple2D(i); }
        SUBCASE("Linear Simple 3D") { LinearSimple3D(i); }
        SUBCASE("Linear Tricky 3D") { LinearTricky3D(i); }
        SUBCASE("Non Linear Simple 3D") { NonLinearSimple3D(i); }
    }
}