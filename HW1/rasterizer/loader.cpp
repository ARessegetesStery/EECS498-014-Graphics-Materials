#include "loader.hpp"

#include <cstdint>
#include <iostream>
#include <fstream>

#include "../thirdparty/fkyaml/node.hpp"

#define TINYOBJLOADER_IMPLEMENTATION 
#define TINYOBJLOADER_USE_MAPBOX_EARCUT                 // use robust triangulation
#define TINYOBJLOADER_DONOT_INCLUDE_MAPBOX_EARCUT
#include "../thirdparty/mapbox/earcut.hpp"              // included as a separate header
#include "../thirdparty/tinyobj/tiny_obj_loader.h"

void LoadVec3(const fkyaml::node& parent, std::string tag, glm::vec3& vec)
{
    if (parent.contains(tag))
    {
        auto node = parent[tag];
        vec.x = node[0].get_value<float>();
        vec.y = node[1].get_value<float>();
        vec.z = node[2].get_value<float>();
    }
    else 
    {
        std::string msg = "missing tag " + tag;
        throw fkyaml::exception(msg.c_str());
    }
}

void LoadColor(const fkyaml::node& parent, std::string tag, Color& color)
{
    if (parent.contains(tag))
    {
        auto node = parent[tag];
        uint32_t r, g, b, a;
        r = node[0].get_value<uint32_t>();
        g = node[1].get_value<uint32_t>();
        b = node[2].get_value<uint32_t>();
        a = 255;

        if (r > 255 || g > 255 || b > 255)
            throw fkyaml::exception("color value exceeding 255");
        
        color = Color(r, g, b, a);
    }
    else 
    {
        std::string msg = "missing tag " + tag;
        throw fkyaml::exception(msg.c_str());
    }
}

void LoadQuat(const fkyaml::node& parent, std::string tag, glm::quat& quat)
{
    if (parent.contains(tag))
    {
        auto node = parent[tag];
        quat.w = node[0].get_value<float>();
        quat.x = node[1].get_value<float>();
        quat.y = node[2].get_value<float>();
        quat.z = node[3].get_value<float>();
    }
    else 
    {
        std::string msg = "missing tag " + tag;
        throw fkyaml::exception(msg.c_str());
    }
}

#define LOAD_DATA_FROM_YAML(var, node, tag, TYPE) \
    if (node.contains(#tag)) \
        var = node[#tag].get_value<TYPE>(); \
    else \
        throw fkyaml::exception(("missing tag " + std::string(#tag)).c_str());

#define LOAD_DEF_DATA_FROM_YAML(var, node, tag, TYPE) \
    TYPE var; \
    LOAD_DATA_FROM_YAML(var, node, tag, TYPE)

#define LOAD_NODE_FROM_YAML_NOERROR(node, root, tag) \
    auto node = root; \
    if (root.contains(#tag)) \
        node = root[#tag]; \
    else \
        std::cout << "[WARNING] missing tag " << #tag << std::endl;

#define LOAD_NODE_FROM_YAML(node, root, tag) \
    auto node = root; \
    if (root.contains(#tag)) \
        node = root[#tag]; \
    else \
        throw fkyaml::exception(("missing tag " + std::string(#tag)).c_str());

#define LOAD_VEC3_FROM_YAML(node, tag, vec)     LoadVec3(node, #tag, vec);
#define LOAD_COLOR_FROM_YAML(node, tag, vec)    LoadColor(node, #tag, vec);
#define LOAD_QUAT_FROM_YAML(node, tag, vec)     LoadQuat(node, #tag, vec);

Loader::Loader(std::string filename) : Loader() 
    {
        this->filename = filename;
    }

bool Loader::Load()
{
    bool yamlSuccess = LoadYaml();
    if (!yamlSuccess)
    {
        std::cerr << "fail loading yaml. Quit.\n";
        return false;
    }
    else 
    {
        bool objSuccess = LoadObj();
        if (!objSuccess)
        {
            std::cerr << "fail loading obj. Quit.\n";
            return false;
        }
        else
            return true;    
    }
}

bool Loader::LoadYaml()
{
    // If the loader fails in any way, the resulting object must have TestType::ERROR

    // Parse the exact content of the config file here 
    // If there is any error in parsing, throw a fkyaml::exception instead of 
    //   directly setting type to TestType::ERROR so that the error can be 
    //   correctly printed out.
    try
    {
        std::ifstream ifs(filename);
        if (!ifs)
        {
            std::string msg = "error opening config file " + filename;
            throw fkyaml::exception(msg.c_str());
        }
        fkyaml::node root = fkyaml::node::deserialize(ifs);

        // type
        LOAD_DEF_DATA_FROM_YAML(task, root, task, std::string)
        if (task == "triangle")
            this->type = TestType::TRIANGLE;
        else if (task == "transform")
            this->type = TestType::TRANSFORM;
        else if (task == "transform-test")
            this->type = TestType::TRANSFORM_TEST;
        else if (task == "shading-depth")
            this->type = TestType::SHADING_DEPTH;
        else if (task == "shading")
            this->type = TestType::SHADING;
        else
        {
            std::string msg = "cannot recognize test type " + task;
            throw fkyaml::exception(msg.c_str());
        }

        // resolution
        LOAD_NODE_FROM_YAML(resNode, root, resolution)
        const uint32_t MAX_RES = 4096;\
        LOAD_DATA_FROM_YAML(this->width, resNode, width, uint32_t)
        LOAD_DATA_FROM_YAML(this->height, resNode, height, uint32_t)

        if (width > MAX_RES || height > MAX_RES)
            throw fkyaml::exception("invalid resolution: width/height exceeding 4096");

        // obj/output filename
        LOAD_DATA_FROM_YAML(this->modelName, root, obj, std::string)
        LOAD_DATA_FROM_YAML(this->outputName, root, output, std::string)

        // If the task is TRANSFORM or SHADING, then there must be a camera; load it
        if (this->type != TestType::TRIANGLE)
        {
            // Load Camera
            LOAD_NODE_FROM_YAML(cameraNode, root, camera)
            LOAD_VEC3_FROM_YAML(cameraNode, pos, camera.pos)
            LOAD_VEC3_FROM_YAML(cameraNode, lookAt, camera.lookAt)
            LOAD_VEC3_FROM_YAML(cameraNode, up, camera.up)
            LOAD_DATA_FROM_YAML(camera.width, cameraNode, width, float)
            LOAD_DATA_FROM_YAML(camera.height, cameraNode, height, float)
            LOAD_DATA_FROM_YAML(camera.nearClip, cameraNode, nearClip, float)
            LOAD_DATA_FROM_YAML(camera.farClip, cameraNode, farClip, float)

            // Load Transforms
            LOAD_NODE_FROM_YAML_NOERROR(transformNode, root, transforms)
            if (transformNode != root)
            {
                for (auto& subnode : transformNode)
                {
                    glm::quat rotation;
                    glm::vec3 translation, scale;
                    LOAD_QUAT_FROM_YAML(subnode, rotation, rotation)
                    LOAD_VEC3_FROM_YAML(subnode, translation, translation)
                    LOAD_VEC3_FROM_YAML(subnode, scale, scale)
                    glm::vec3 scale3(scale);
                    this->transforms.emplace_back(rotation, translation, scale3);
                }
            }

            // Load Light Infos
            LOAD_NODE_FROM_YAML_NOERROR(lightNode, root, lights)
            if (lightNode != root)
            {
                for (auto& light : lightNode)
                {
                    LOAD_DEF_DATA_FROM_YAML(intensity, light, intensity, float)
                    glm::vec3 pos;
                    LOAD_VEC3_FROM_YAML(light, pos, pos)

                    Color color;
                    LOAD_COLOR_FROM_YAML(light, color, color)
                    this->lights.emplace_back(pos, intensity, color);
                }
            }

            if (this->type == TestType::SHADING)
            {
                LOAD_DATA_FROM_YAML(this->specularExponent, root, exponent, float)
                LOAD_COLOR_FROM_YAML(root, ambient, this->ambientColor)
            }
        }
        else if (this->type == TestType::TRIANGLE)
        // if the task is TRIANGLE, then need to check whether it is SSAA
        {
            LOAD_DEF_DATA_FROM_YAML(AAName, root, antialias, std::string)
            if (AAName == "none")
            {
                this->AAConfig = AntiAliasConfig::NONE;
                this->AASpp = 0;
            }
            else if (AAName == "SSAA")
            {
                this->AAConfig = AntiAliasConfig::SSAA;
                LOAD_DATA_FROM_YAML(this->AASpp, root, samples, uint32_t)
            }
        }

        // If the task is TRANSFORM_TEST, then load the input/expected
        if (this->type == TestType::TRANSFORM_TEST)
        {
            glm::vec3 tempInput, tempExpected;
            LOAD_VEC3_FROM_YAML(root, input, tempInput)
            LOAD_VEC3_FROM_YAML(root, expected, tempExpected)
            this->input = glm::vec3(tempInput);
            this->expected = tempExpected;
        }
    }
    catch(const fkyaml::exception& e)
    {
        std::cerr << "error parsing config yaml file: " << filename << std::endl;
        std::cerr << "| msg: " << e.what() << '\n';
        this->type = TestType::ERROR;
    }

    // If type is error, init height and width with 0; otherwise all data must have been initialized
    //   where they are retrieved.
    if (this->type == TestType::ERROR)
    {
        this->width = 0;
        this->height = 0;
        this->outputName = "__error";
        return false;
    }
    return true;
}

bool Loader::LoadObj()
{
    std::string filename = this->modelName + ".obj";
    tinyobj::ObjReaderConfig readerConfig;
    readerConfig.mtl_search_path = "./";
    readerConfig.triangulate = true;
    readerConfig.triangulation_method = "earcut";
    readerConfig.vertex_color = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, readerConfig)) 
    {
        if (!reader.Error().empty()) 
            std::cerr << "TinyObjReader [ERROR]: " << reader.Error();
        return false;
    }

    if (!reader.Warning().empty()) 
        std::cout << "TinyObjReader [WARNING]: " << reader.Warning();

    this->attribs = reader.GetAttrib();
    this->shapes = reader.GetShapes();

    return true;
}
