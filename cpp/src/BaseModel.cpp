#include <BaseModel.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

void BaseModel::debuglog(std::string msg) {
    std::fstream fout;
    fout.open("debug.txt", std::ios::out | std::ios::app);
    fout << msg << '\n';
    fout.close();
}

void BaseModel::save(const char* path) const {
    // TODO : assert (sum of npl == size of model array)
    std::fstream fout;
    fout.open(path, std::ios::out | std::ios::trunc);

    int offset = 0;
    for (int i = 0; i < 1 /* npl.size()*/; i++) {
        fout << weights[offset];

        for (int j = 1; j < weights_count /* npl[i] */; j++) {
            fout << ", " << weights[offset + j];
        }

        fout << '\n';

        // offset += npl[i];
    }
    fout.close();
}

void BaseModel::load(const char* path) {
    std::vector<double> tmp;
    int i = 0;

    std::ifstream data(path);
    std::string line;

    while (std::getline(data, line)) {
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ',')) {
            tmp.push_back(atof(cell.c_str()));
        }
    }

    std::copy(tmp.begin(), tmp.end(), weights);
}
