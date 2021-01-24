#include <doctest/doctest.h>

#include <BaseModel.hpp>
#include <Library.hpp>
#include <MLP.hpp>

//// Test pour savoir si la fonction de sauvegarde et load fonctionne bien
//TEST_CASE("Save/Load") {
//    // *** Init and save ***
//    BaseModel* model1 = CreateModel(0, 3, true);
//    double* weights1 = model1->getWeigths();
//    weights1[0] = 0.25;
//    weights1[1] = 3;
//    weights1[2] = -2;
//    model1->save("test_model.csv");
//
//    // *** Load and check ***
//    BaseModel* model2 = CreateModel(0, 3, true);
//    model2->load("test_model.csv");
//    double* weights2 = model2->getWeigths();
//    CHECK(weights2[0] == 0.25);
//    CHECK(weights2[1] == 3);
//    CHECK(weights2[2] == -2);
//}

TEST_CASE("Save/Load") {
    unsigned int weights_count = 3;

    std::vector<unsigned int> topology = {weights_count, 3, 3, 1};

    // *** Init and save ***
    MLP model1 = MLP(topology, weights_count, true);

    int i = 0;
    for (Layer& layer : model1._layers) {
        for (Neuron& neuron : layer) {
            for (Connection& connection : neuron._outputWeights) {
                connection.weight = i;
                connection.deltaWeight = i;
                i++; 
            }
        }
    }

    model1.save("test_model.csv");

    // *** Load and check ***
    MLP model2 = MLP(topology, weights_count, true);
    model2.load("test_model.csv");

    i = 0;
    for (Layer& layer : model2._layers) {
        for (Neuron& neuron : layer) {
            for (Connection& connection : neuron._outputWeights) {
                CHECK(connection.weight == i);
                CHECK(connection.deltaWeight == i);
                i++;
            }
        }
    }
}