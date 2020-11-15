
// #include <doctest/doctest.h>

// #include <CSVFile.hpp>
// #include <vector>

// TEST_CASE("Load a CSV file") {
//     std::vector<double> inputs;
//     std::vector<double> outputs;

//     CSVFile file(DATA_PATH "/CSV/winequality-train.csv");
//     file.loadAsset(inputs, 11, outputs, 1);

//     // To check if the csv is good
//     CHECK(inputs.size() > 0);
//     CHECK(inputs.size() == 4893 * 11);
//     CHECK(outputs.size() == 4893);
//     CHECK(inputs[0] == 7.0);
//     CHECK(inputs[1] == 0.27);
//     CHECK(inputs[11] == 6.3);
//     CHECK(outputs[0] == 6.0);

//     for (auto output : outputs) {
//         CHECK(0 <= output);
//         CHECK(output <= 10);
//     }
// }