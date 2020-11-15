#ifndef IMAGEMANAGER_HPP
#define IMAGEMANAGER_HPP

#include <array>
#include <iostream>
#include <string>
#include <vector>

class ImageManager {
  public:
    ImageManager() : width(32), height(32) {}
    ImageManager(int size) : width(size), height(size) {}
    ImageManager(int width, int height) : width(width), height(height) {}

    void loadAsset(std::vector<double>& input_images, std::vector<double>& outputs) const;
    static void getFilesInDirectory(std::vector<std::string>& out, const std::string& directory);

  private:
    int width, height;
    const double isAnime = 0, isBD = 1;
};

#endif