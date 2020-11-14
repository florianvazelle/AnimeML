#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <array>
#include <iostream>
#include <string>
#include <vector>

class Image {
  public:
    Image(char const* filename);
    Image(int w, int h);

    auto operator[](int r) { return m_pixels.data() + r * m_width; }
    auto operator[](int r) const { return m_pixels.data() + r * m_width; }

    inline auto begin() { return m_pixels.begin(); }
    inline auto end() { return m_pixels.end(); }

    void resize(int width, int height);

  private:
    int m_width, m_height;
    std::vector<double> m_pixels;
};

#endif