#include <doctest/doctest.h>

#include <BaseModel.hpp>
#include <Library.hpp>

TEST_CASE("Save/Load") {
    // *** Init and save ***
    BaseModel* model1 = CreateModel(0, 3, true);
    double* weights1 = model1->getWeigths();
    weights1[0] = 0.25;
    weights1[1] = 3;
    weights1[2] = -2;
    model1->save("test_model.csv");

    // *** Load and check ***
    BaseModel* model2 = CreateModel(0, 3, true);
    model2->load("test_model.csv");
    double* weights2 = model2->getWeigths();
    CHECK(weights2[0] == 0.25);
    CHECK(weights2[1] == 3);
    CHECK(weights2[2] == -2);
}