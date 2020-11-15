#include <CSVFile.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

CSVFile::CSVFile(std::string path, char separator) : path(path) {
    // *** Read File ***
    int i = 0;

    std::ifstream targetFile(path);
    std::string line;

    while (std::getline(targetFile, line)) {
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, separator)) {
            data.push_back(cell);
        }
        row++;
    }
    targetFile.close();

    column = data.size() / row;
}

void CSVFile::loadAsset(std::vector<double>& input_images, int inputs_size, std::vector<double>& outputs, int outputs_size) const {
    assert(inputs_size + outputs_size == column);

    // we assume that the outputs are at the end of the line
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < inputs_size; j++) {
            input_images.push_back(atof(data[i * column + j].c_str()));
        }
        for (int j = 0; j < outputs_size; j++) {
            outputs.push_back(atof(data[i * column + j + inputs_size].c_str()));
        }
    }
}