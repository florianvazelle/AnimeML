#include <BaseModel.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

BaseModel::BaseModel(std::string filename) {
    std::vector<double> tmp;
    int i = 0;

    std::ifstream data(filename);
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

void BaseModel::debuglog(std::string msg) {
    std::fstream fout;
    fout.open("debug.txt", std::ios::out | std::ios::app);
    fout << msg << '\n';
    fout.close();
}

void BaseModel::save(std::vector<int> npl, std::string filename) const {
    // TODO : assert (sum of npl == size of model array)
    std::fstream fout;
    fout.open(filename, std::ios::out | std::ios::trunc);

    int offset = 0;
    for (int i = 0; i < npl.size(); i++) {
        fout << weights[offset];

        for (int j = 1; j < npl[i]; j++) {
            fout << ", " << weights[offset + j];
        }

        fout << '\n';

        offset += npl[i];
    }
    fout.close();
}
