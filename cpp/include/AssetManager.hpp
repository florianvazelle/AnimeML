#ifndef ASSETMANAGER_HPP
#define ASSETMANAGER_HPP

#include <array>
#include <iostream>
#include <string>
#include <vector>

class AssetManager {
  public:
    AssetManager() : width(32), height(32) {}
    AssetManager(int size) : width(size), height(size) {}
    AssetManager(int width, int height) : width(width), height(height) {}

    void loadAsset(std::vector<double>& input_images, std::vector<double>& outputs) const;
    static void getFilesInDirectory(std::vector<std::string>& out, const std::string& directory);

  private:
    int width, height;
    const double isAnime = 0, isBD = 1;
};

#endif