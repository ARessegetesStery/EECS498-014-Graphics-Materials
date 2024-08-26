// Defines structures for passing around

#ifndef ENTITIES_H
#define ENTITIES_H

#include <array>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>

#include "image.hpp"

#include "../thirdparty/glm/glm.hpp"
#include "../thirdparty/glm/gtc/quaternion.hpp"

struct Triangle
{
    std::array<glm::vec4, 3> pos;
    std::array<glm::vec4, 3> normal;

    inline void Homogenize()
    {
        for (size_t i = 0; i < 3; ++i)
            pos[i] /= pos[i].w;
    }
};

template<typename T>
inline std::string ToStr(const T val, const int n = 3)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << val;
    return std::move(out).str();
}

inline std::string ToStr(glm::vec4 vec)
{
    return "[x: " + ToStr(vec.x) + "]" +
        "[y: " + ToStr(vec.y) + "]" +
        "[z: " + ToStr(vec.z) + "]" +
        "[w: " + ToStr(vec.w) + "]";
}

inline std::string ToStr(glm::vec3 vec)
{
    return "[x: " + ToStr(vec.x) + "]" +
        "[y: " + ToStr(vec.y) + "]" +
        "[z: " + ToStr(vec.z) + "]";
}

inline std::string ToStr(Color color)
{
    auto charToStr = [](char val) -> std::string
    {
        int ival = static_cast<int>(val);
        return ToStr(val < 0 ? ival + 256 : ival);
    };
    return "[r: " + charToStr(color.r) + "]" +
        "[g: " + charToStr(color.g) + "]" +
        "[b: " + charToStr(color.b) + "]" +
        "[a: " + charToStr(color.a) + "]";

}

inline std::string ToStr(glm::quat quat)
{
    return "[w: " + ToStr(quat.w) + "]" +
        "[x: " + ToStr(quat.x) + "]" +
        "[y: " + ToStr(quat.y) + "]" +
        "[z: " + ToStr(quat.z) + "]";

}

struct Camera
{
    // invalid camera has all members set to 0 (or 0 vector)
    glm::vec3 pos;
    glm::vec3 lookAt;
    glm::vec3 up;
    float_t width;
    float_t height;
    float_t nearClip;
    float_t farClip;

    inline std::string Info() const 
    {
        return std::string("Camera:\n") +
            "| pos: " + ToStr(this->pos) + "\n" +
            "| lookAt: " + ToStr(this->lookAt) + "\n" +
            "| up: " + ToStr(this->up) + "\n" +
            "| width: " + ToStr(width) + "\n" +
            "| height: " + ToStr(height) + "\n" +
            "| nearClip: " + ToStr(nearClip) + "\n" +
            "| farClip: " + ToStr(farClip);
    }

    Camera() : 
        pos(0, 0, 0), 
        lookAt(0, 0, 0),
        up(0, 0, 0),
        width(0),
        height(0),
        nearClip(0),
        farClip(0)
    {  }
};

struct MeshTransform
{
public:
    glm::quat rotation;         // quaternion representation of rotation/orientation
    glm::vec3 translation;
    glm::vec3 scale;

    MeshTransform(glm::quat rotation, glm::vec3 translation, glm::vec3 scale) : 
        rotation(rotation), translation(translation), scale(scale) {  }
};

struct Light
{
public:
    glm::vec3 pos;
    float intensity;
    Color color;

    Light(glm::vec3 pos, float intensity, Color color) : 
        pos(pos), intensity(intensity), color(color) {  }
};

#endif
