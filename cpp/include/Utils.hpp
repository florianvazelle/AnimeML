#pragma once

#include <vector>
#include <random>

namespace ml {
    static double rand(double min = 0, double max = 1) {
        double f = ((double)std::rand()) / ((double)RAND_MAX);
        return min + f * (max - min);
    }

    template <typename T> static void random_shuffle(std::vector<T>& vec) {
        std::random_device rng;
        std::mt19937 urng(rng());
        std::shuffle(vec.begin(), vec.end(), urng);
    }

    static inline bool double_equals(double a, double b, double epsilon = 0.001) { return std::abs(a - b) < epsilon; }
};