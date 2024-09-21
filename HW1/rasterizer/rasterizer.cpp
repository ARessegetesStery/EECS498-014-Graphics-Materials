#include "rasterizer.hpp"

#include "loader.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

#include "../thirdparty/glm/gtx/quaternion.hpp"

// include standard libraries here if you need any

// @includealso 

// TODO implement @includealso copying

// This is the only single file where you should write your solution to
// If you have other implementations that you would like to include in for grading,
//  please add the files to the @includealso tag above. Otherwise, your files will
//  not be included in grading. 

Rasterizer::Rasterizer(Loader& loader) : 
    loader(loader),
    model(),
    view(glm::mat4(1.f)),  
    projection(glm::mat4(1.f)),  
    screenspace(glm::mat4(1.f)),
    ZBuffer(loader.GetWidth(), loader.GetHeight(), loader.GetOutputName())
{   
    for (size_t i = 0; i != loader.GetHeight(); ++i)
        for (size_t j = 0; j != loader.GetWidth(); ++j)
            ZBuffer.Set(j, i, -1.f);
}

void Rasterizer::DrawPrimitiveRaw(Image &image, Triangle trig, AntiAliasConfig config, uint32_t spp)
{
    uint32_t xmax = 0, xmin = UINT32_MAX;
    uint32_t ymax = 0, ymin = UINT32_MAX;
    const std::array<glm::vec4, 3>& vertices = trig.pos;

    std::array<glm::vec3, 3> trimmedPos;
    for (size_t ind = 0; ind != 3; ++ind)
    {
        const glm::vec4& v = vertices[ind];
        if (v.x > xmax)
            xmax = v.x;
        if (v.x < xmin)
            xmin = v.x;
        if (v.y > ymax)
            ymax = v.y;
        if (v.y < ymin)
            ymin = v.y;
        trimmedPos[ind] = glm::vec3(v.x, v.y, 0);
    }
    
    for (uint32_t x = xmin; x <= xmax; ++x)
        for (uint32_t y = ymin; y <= ymax; ++y)
            this->DrawPixel(x, y, trig, config, spp, image, Color::White);
}

void Rasterizer::AddModel(MeshTransform transform)
{
    glm::mat4 rotation = glm::toMat4(transform.rotation);
    this->AddModel(transform, rotation);
}

void Rasterizer::InitMSSAMask(ImageGrey& MSSAMask, uint32_t num_samples)
{
    for (size_t i = 0; i != this->loader.GetHeight(); ++i)
        for (size_t j = 0; j != this->loader.GetWidth(); ++j)
            MSSAMask.Set(j, i, Rasterizer::msaaMaskDefault);
    
    for(uint32_t i = 0; i < num_samples; ++i){
        msaaSamples.emplace_back(
            std::cos(num_samples/(2*std::numbers::pi)) + std::cos(std::numbers::pi/4),
            std::sin(num_samples/(2*std::numbers::pi)) + std::sin(std::numbers::pi/4)
        );
    }
    msaaSamples.emplace_back(0.5F,0.5F);
}

void Rasterizer::InitZBuffer(ImageGrey& ZBuffer)
{
    for (size_t i = 0; i != this->loader.GetHeight(); ++i)
        for (size_t j = 0; j != this->loader.GetWidth(); ++j)
            ZBuffer.Set(j, i, Rasterizer::zBufferDefault);
}

void Rasterizer::DrawPrimitiveDepth(Triangle transformed, Triangle original, ImageGrey& ZBuffer)
{
    uint32_t xmax = 0, xmin = UINT32_MAX;
    uint32_t ymax = 0, ymin = UINT32_MAX;
    const std::array<glm::vec4, 3>& vertices = transformed.pos;

    std::array<glm::vec3, 3> trimmedPos;
    for (size_t ind = 0; ind != 3; ++ind)
    {
        const glm::vec4& v = vertices[ind];
        if (v.x > xmax)
            xmax = v.x;
        if (v.x < xmin)
            xmin = v.x;
        if (v.y > ymax)
            ymax = v.y;
        if (v.y < ymin)
            ymin = v.y;
        trimmedPos[ind] = glm::vec3(v.x, v.y, 0);
    }

    for (uint32_t x = xmin; x <= xmax; ++x)
        for (uint32_t y = ymin; y <= ymax; ++y)
            this->UpdateDepthAtPixel(x, y, original, transformed, ZBuffer);
}

void Rasterizer::DrawPrimitiveShaded(Triangle transformed, Triangle original, Image& image)
{
    uint32_t xmax = 0, xmin = UINT32_MAX;
    uint32_t ymax = 0, ymin = UINT32_MAX;
    const std::array<glm::vec4, 3>& vertices = transformed.pos;

    std::array<glm::vec3, 3> trimmedPos;
    for (size_t ind = 0; ind != 3; ++ind)
    {
        const glm::vec4& v = vertices[ind];
        if (v.x > xmax)
            xmax = v.x;
        if (v.x < xmin)
            xmin = v.x;
        if (v.y > ymax)
            ymax = v.y;
        if (v.y < ymin)
            ymin = v.y;
        trimmedPos[ind] = glm::vec3(v.x, v.y, 0);
    }

    for (uint32_t x = xmin; x <= xmax; ++x)
        for (uint32_t y = ymin; y <= ymax; ++y)
            this->ShadeAtPixel(x, y, original, transformed, image);
}
