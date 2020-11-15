#ifndef CSVMANAGER_HPP
#define CSVMANAGER_HPP

#include <iostream>
#include <vector>

class CSVFile {
  public:
    CSVFile(std::string path, char separator = ';');

    void loadAsset(std::vector<double>& input_images, int inputs_size, std::vector<double>& outputs, int outputs_size) const;

  private:
    std::string path;
    std::vector<std::string> data;
    int row, column;
};

#endif