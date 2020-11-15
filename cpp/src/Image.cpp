#include <Image.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdexcept>

Image::Image(char const* filename) {
    int c;
    uint8_t* pixels = stbi_load(filename, &m_width, &m_height, &c, STBI_rgb_alpha);
    if (pixels) {
        m_pixels.resize(m_height * m_width);

        for (int i = 0; i < m_height * m_width; ++i) {
            int j = 4 * i;
            m_pixels[i] = static_cast<double>(pixels[j + 1]);
            // m_pixels[i + 1] = pixels[j + 2];
            // m_pixels[i + 2] = pixels[j + 3];
        }

        stbi_image_free(pixels);
    } else {
        throw std::runtime_error("Cannot load image");
    }
}

Image::Image(int w, int h) {
    m_height = h;
    m_width = w;
    m_pixels.resize(w * h);
}

void Image::resize(int width, int height) {
    int scalex = m_width / width;
    int scaley = m_height / height;
    Image tmp(width, height);

    double* filtered_row = new double[width];
    for (int r = 0; r < height; r++) {
        if (r * scaley < m_height) {
            double* row = (*this)[r * scaley];
            for (int c = 0; c < width; c++) {
                if (c * scalex < m_width) {
                    filtered_row[c] = row[c * scalex];
                }
            }
            std::copy(filtered_row, filtered_row + width, tmp[r]);
        }
    }

    (*this) = tmp;
}