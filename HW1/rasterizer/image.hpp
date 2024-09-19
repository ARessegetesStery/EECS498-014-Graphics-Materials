#ifndef ImageBuffer_H
#define ImageBuffer_H

#include <algorithm>
#include <string>
#include <optional>

#include "../thirdparty/glm/glm.hpp"

class Color
{
public:
    unsigned char r, g, b, a;

public:
    // Constructors
    Color();
    Color(float);
    Color(float, float, float, float);
    Color(glm::vec4&);
    Color(glm::vec3&);
    Color(const Color&);

    // Assignments and equality judgement
    Color& operator= (const Color&);
    bool operator== (const Color&);
    bool operator!= (const Color&);
    const char operator[] (size_t index) const;

    // Clearing colors
    static Color White;
    static Color Black;
};

inline Color operator+ (const Color& c1, const Color& c2)
{
    return Color(
        std::clamp(static_cast<float>(c1.r) + static_cast<float>(c2.r), 0.f, 255.f),
        std::clamp(static_cast<float>(c1.g) + static_cast<float>(c2.g), 0.f, 255.f),
        std::clamp(static_cast<float>(c1.b) + static_cast<float>(c2.b), 0.f, 255.f),
        static_cast<float>(c1.a)
    );
}

inline Color operator* (float coeff, const Color& c)
{
    return Color(
        std::clamp(static_cast<float>(c.r) * coeff, 0.f, 255.f),
        std::clamp(static_cast<float>(c.g) * coeff, 0.f, 255.f),
        std::clamp(static_cast<float>(c.b) * coeff, 0.f, 255.f),
        static_cast<float>(c.a)
    );
}

inline Color operator* (const Color& c, float coeff)
{
    return coeff * c;
}

template<typename T>
class ImageBuffer
{
private:
    uint32_t width, height;
    T* canvas;
    std::string filename;

public:
    // Constructors
    ImageBuffer(std::string = "output");
    ImageBuffer(uint32_t width, uint32_t height, std::string = "output");
    ImageBuffer(const ImageBuffer&);
    ~ImageBuffer();

    ImageBuffer& operator= (const ImageBuffer&);

    // Set/Get color for a specific pixel
    //     Attempting to set color to an invalid pixel will result in no change in the canvas
    //  Attempting to get color from an invalid pixel will give black
    void Set(uint32_t w, uint32_t h, T);
    std::optional<T> Get(uint32_t w, uint32_t h) const;

    // Write the canvas to a .png file with the designated filename
    void Write();

    inline uint32_t GetWidth() const { return width; }
    inline uint32_t GetHeight() const { return height; }
};

using Image = ImageBuffer<Color>;
using ImageGrey = ImageBuffer<float>;

template<typename T>
ImageBuffer<T>::ImageBuffer(std::string filename)
{
    this->width = 100;
    this->height = 100;
    this->canvas = new T[100 * 100];
    this->filename = filename;
}

template<typename T>
ImageBuffer<T>::ImageBuffer(unsigned int w, unsigned int h, std::string filename)
{
    if (w > 2000)
        w = 2000;
    if (h > 2000)
        h = 2000;
    this->width = w;
    this->height = h;
    this->canvas = new T[static_cast<size_t>(w) * static_cast<size_t>(h)];
    this->filename = filename;
}

template<>
ImageBuffer<Color>::ImageBuffer(std::string filename);

template<>
ImageBuffer<Color>::ImageBuffer(std::string filename);

template<>
ImageBuffer<Color>::ImageBuffer(unsigned int w, unsigned int h, std::string filename);

template<typename T>
ImageBuffer<T>::ImageBuffer(const ImageBuffer<T>& image)
{
    *this = image;
}

template<typename T>
ImageBuffer<T>::~ImageBuffer()
{
    if (canvas)
        delete[] canvas;
}

template<typename T>
ImageBuffer<T>& ImageBuffer<T>::operator= (const ImageBuffer<T>& image)
{
    if (this->canvas)
        delete[] canvas;

    this->width = image.width;
    this->height = image.height;
    this->canvas = new T[image.width * image.height];
    for (unsigned int i = 0; i != image.width * image.height; ++i)
        this->canvas[i] = image.canvas[i];
    this->filename = image.filename;

    return *this;
}

template<typename T>
void ImageBuffer<T>::Set(unsigned int w, unsigned int h, T c)
{
    if (!(!canvas || w >= width || h >= height))
        this->canvas[(size_t)(h * this->width + w)] = c;
}

template<typename T>
std::optional<T> ImageBuffer<T>::Get(unsigned int w, unsigned int h) const
{
    if (!(!canvas || w >= width || h >= height))
        return this->canvas[(size_t)(h * this->width + w)];
    return std::nullopt;
}

#endif
