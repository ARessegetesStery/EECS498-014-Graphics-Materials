#ifndef LOADER_H
#define LOADER_H

#include <cstdint>
#include <string>
#include <optional>

#include "entities.hpp"
#include "../thirdparty/tinyobj/tiny_obj_fwd.h"

namespace tinyobj
{
    struct shape_t;
    struct attrib_t;
}

// Loads yaml config and obj models

enum class TestType
{
    TRIANGLE,
    TRANSFORM, TRANSFORM_TEST,
    SHADING_DEPTH, SHADING,
    ERROR
};

enum class AntiAliasConfig
{
    NONE, SSAA
};

std::string ToStr(glm::vec4 vec);
std::string ToStr(glm::vec3 vec);

class Loader
{
public:
    Loader() = default;
    Loader(std::string filename);

    bool Load();


    inline std::string Info() const
    {
        std::string typeStr = "";
        if (this->type == TestType::TRIANGLE)
            typeStr = "triangle";
        else if (this->type == TestType::TRANSFORM)
            typeStr = "transform";
        else if (this->type == TestType::SHADING)
            typeStr = "shading";
        else if (this->type == TestType::TRANSFORM_TEST)
            typeStr = "transform_test";
        else if (this->type == TestType::ERROR)
            typeStr = "error";

        std::string AAStr = "";
        if (this->AAConfig == AntiAliasConfig::NONE)
            AAStr = "none";
        else if (this->AAConfig == AntiAliasConfig::SSAA)
            AAStr = "SSAA";

        std::string transformStr = "<no transform needed>\n";
        if (this->type != TestType::TRIANGLE)
        {
            if (this->transforms.empty())
                transformStr = "[WARNING] <no transform specified>\n";
            else
            {
                transformStr = "Transforms:\n";
                for (auto& transform : this->transforms)
                {
                    transformStr += "| - rotation: " + ToStr(transform.rotation) + "\n";
                    transformStr += "|   translation: " + ToStr(transform.translation) + "\n";
                    transformStr += "|   scale: " + ToStr(transform.scale) + "\n";
                }
            }
            if (this->transforms.size() != this->shapes.size())
                transformStr += "[WARNING] number of transforms does not match number of shapes\n";
        }

        std::string lightStr = "<no light needed>\n";
        if (this->type == TestType::SHADING)
        {
            lightStr = "";
            lightStr += "Specular Exponent: " + ToStr(this->specularExponent) + "\n";
            lightStr += "Ambient Color: " + ToStr(this->ambientColor) + "\n";
            if (this->lights.empty())
                lightStr += "[WARNING] <no light specified>\n";
            else
            {
                lightStr += "Lights:\n";
                for (auto& light : this->lights)
                {
                    lightStr += "| - position: " + ToStr(light.pos) + "\n";
                    lightStr += "|   intensity: " + ToStr(light.intensity) + "\n";
                    lightStr += "|   color: " + ToStr(light.color) + "\n";
                }
            }
        }

        return "Type: " + typeStr + "\n" +
            "Anti-alias: " + AAStr + ((this->AAConfig == AntiAliasConfig::NONE) ? "" : " with spp " + ToStr(this->AASpp)) + "\n" +
            "Resolution: " + ToStr(this->width) + "x" + ToStr(this->height) + "\n" +
            "Model: " + this->modelName + "\n" +
            "Output: " + this->outputName + "\n" + 
            ((camera.width == 0) ? "<no camera specified>" : (this->camera.Info())) + "\n" +
            transformStr + lightStr;
    }

    inline const TestType GetType() const { return this->type; }
    inline const AntiAliasConfig GetAntiAliasConfig() const { return this->AAConfig; }
    inline const uint32_t GetSpp() const { return this->AASpp; }
    inline const uint32_t GetWidth() const { return this->width; }
    inline const uint32_t GetHeight() const { return this->height; }

    inline const glm::vec3 GetTestInput() const 
    {
        if (this->input.has_value())
            return this->input.value();
        else
            throw std::runtime_error("Transform test input not specified");
    }
    inline const glm::vec3 GetTestExpected() const
    {
        if (this->expected.has_value())
            return this->expected.value();
        else
            throw std::runtime_error("Transform test expected output not specified");
    }

    inline const Camera& GetCamera() const { return this->camera; }
    inline const std::vector<tinyobj::shape_t>& GetShapes() const { return this->shapes; }
    inline const std::vector<MeshTransform>& GetTransforms() const { return this->transforms; }
    inline const std::vector<Light>& GetLights() const { return this->lights; }
    inline const float GetSpecularExponent() const { return this->specularExponent; }
    inline const Color GetAmbientColor() const { return this->ambientColor; }
    inline const tinyobj::attrib_t& GetAttribs() const { return this->attribs; }

private:
    // configs
    std::string filename;

    TestType type;
    uint32_t width;
    uint32_t height;
    std::string modelName;
    std::string outputName;
    AntiAliasConfig AAConfig = AntiAliasConfig::NONE;
    uint32_t AASpp = 0;

    std::optional<glm::vec3> expected;
    std::optional<glm::vec3> input;

    Camera camera;

    tinyobj::attrib_t attribs;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<MeshTransform> transforms;

    std::vector<Light> lights;
    float specularExponent;
    Color ambientColor;

    // helpers
    bool LoadYaml();
    bool LoadObj();
};

#endif
