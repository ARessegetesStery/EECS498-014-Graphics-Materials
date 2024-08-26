#include "image.hpp"

#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../thirdparty/stb/stb_image.h"
#include "../thirdparty/stb/stb_image_write.h"

Color Color::White = Color(255, 255, 255, 255);
Color Color::Black = Color(0, 0, 0, 255);

Color::Color() : 
    r(static_cast<char>(0)), g(static_cast<char>(0)), b(static_cast<char>(0)), a(static_cast<char>(255)) {     }

Color::Color(float grey) : 
    r(static_cast<char>(grey)), g(static_cast<char>(grey)), b(static_cast<char>(grey)), a(static_cast<char>(255)) {  }

Color::Color(float r, float g, float b, float a) :
    r(static_cast<char>(r)), g(static_cast<char>(g)), b(static_cast<char>(b)), a(static_cast<char>(a)) {  }

Color::Color(glm::vec4& v) : Color({ v.x, v.y, v.z, v.w }) {  }

Color::Color(glm::vec3& v) : Color({ v.x, v.y, v.z, 255 }) {    }

Color::Color(const Color& c) : r(c.r), g(c.g), b(c.b), a(c.a) {     }

Color& Color::operator= (const Color& c)
{
    this->r = c.r;
    this->g = c.g;
    this->b = c.b;
    this->a = c.a;
    return *this;
}

bool Color::operator==(const Color& c)
{
    return (c.r == this->r && c.g == this->g && c.b == this->b && c.a == this->a);
}

bool Color::operator!=(const Color& c)
{
    return !(*this == c);
}

const char Color::operator[] (size_t index) const
{
    if (index == 0)
        return r;
    else if (index == 1)
        return g;
    else if (index == 2)
        return b;
    else if (index == 3)
        return a;
    return a;
}

template<>
ImageBuffer<Color>::ImageBuffer(std::string filename)
{
    this->width = 100;
    this->height = 100;
    this->canvas = new Color[100 * 100];
    for (size_t i = 0; i != 100 * 100; ++i)
        this->canvas[i] = Color::Black;
    this->filename = filename;
}

template<>
ImageBuffer<Color>::ImageBuffer(unsigned int w, unsigned int h, std::string filename)
{
    if (w > 2000)
        w = 2000;
    if (h > 2000)
        h = 2000;
    this->width = w;
    this->height = h;
    this->canvas = new Color[static_cast<size_t>(w) * static_cast<size_t>(h)];
    for (size_t i = 0; i != this->width * this->height; ++i)
        this->canvas[i] = Color::Black;
    this->filename = filename;
}

template<typename T>
void ImageBuffer<T>::Write()
{
    std::cerr << "Writing files not of greyscale or color type is not supported.\n";
}

template<>
void ImageBuffer<Color>::Write()
{
    std::string resStr = std::to_string(this->width) + "x" + std::to_string(this->height);
    std::cout << "Writing to PNG with resolution " << resStr << " for colored images.\n";
    stbi_flip_vertically_on_write(true);
    
    int info;
    info = stbi_write_png((filename + ".png").c_str(), this->width, this->height, 4, this->canvas, 0);
    if (!info)
        std::cerr << "Writing to " << filename << ".png failed." << std::endl;
}

template<>
void ImageBuffer<float>::Write()
{
    std::string resStr = std::to_string(this->width) + "x" + std::to_string(this->height);
    std::cout << "Writing to PNG with resolution " << resStr << " for greyscale images.\n";
    stbi_flip_vertically_on_write(true);

    Color* colorCanvas = new Color[this->width * this->height];

    for (size_t index = 0; index != this->width * this->height; ++index)
    {
        float val = this->canvas[index];
        val = 127.5f - 127.5f * val;
        val = std::clamp(val, 0.f, 255.f);
        colorCanvas[index] = Color(val, val, val, 255);
    }
    
    int info;
    info = stbi_write_png((filename + ".png").c_str(), this->width, this->height, 4, colorCanvas, 0);
    if (!info)
        std::cerr << "Writing to " << filename << ".png failed." << std::endl;

    delete[] colorCanvas;
}
