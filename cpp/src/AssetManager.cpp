#include <AssetManager.hpp>

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <Image.hpp>

namespace fs = std::filesystem;  // g++ -v >= 9

/* Retourne la liste de chemin contenu dans le repertoire passer en parametre */
void AssetManager::getFilesInDirectory(std::vector<std::string>& out, const std::string& directory) {
    for (const fs::directory_entry& p : fs::directory_iterator(directory)) {
        fs::path path = p.path();
        fs::path ext = path.extension();

        // On ne selectionne que des image
        if (ext == ".png" || ext == ".jpg" || ext == ".bmp") {
            std::string path_str = path.string();

            out.push_back(path_str);
            // if (out.size() >= 15) break;
        }
    }
}

void AssetManager::loadAsset(std::vector<double>& input_images, std::vector<double>& outputs) const {
    // load all anime path
    std::vector<std::string> anime_images_path;
    getFilesInDirectory(anime_images_path, DATA_PATH "/Anime");

    for (const std::string& path : anime_images_path) {
        Image img(path.c_str());
        img.resize(width, height);

        input_images.insert(input_images.end(), img.begin(), img.end());
        outputs.push_back(isAnime);
    }

    // load all BD image
    std::vector<std::string> bd_images_path;
    getFilesInDirectory(bd_images_path, DATA_PATH "/BD");

    for (const std::string& path : bd_images_path) {
        Image img(path.c_str());
        img.resize(width, height);

        input_images.insert(input_images.end(), img.begin(), img.end());
        outputs.push_back(isBD);
    }
}