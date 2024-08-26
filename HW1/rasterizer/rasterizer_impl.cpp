#include <cstdint>

#include "image.hpp"
#include "loader.hpp"
#include "rasterizer.hpp"

// TODO
void Rasterizer::DrawPixel(uint32_t x, uint32_t y, Triangle trig, AntiAliasConfig config, uint32_t spp, Image& image, Color color)
{
    if (config == AntiAliasConfig::NONE)            // if anti-aliasing is off
    {

    }
    else if (config == AntiAliasConfig::SSAA)       // if anti-aliasing is on
    {

    }

    return;
}

// TODO
void Rasterizer::AddModel(MeshTransform transform, glm::mat4 rotation)
{
    /* model.push_back( model transformation constructed from translation, rotation and scale );*/
    return;
}

// TODO
void Rasterizer::SetView()
{
    const Camera& camera = this->loader.GetCamera();
    glm::vec3 cameraPos = camera.pos;
    glm::vec3 cameraLookAt = camera.lookAt;

    this->view = glm::mat4(1.);
    return;
}

// TODO
void Rasterizer::SetProjection()
{
    return;
}

// TODO
void Rasterizer::SetScreenSpace()
{
    float width = this->loader.GetWidth();
    float height = this->loader.GetHeight();

    return;
}

// TODO
glm::vec3 Rasterizer::BarycentricCoordinate(glm::vec2 pos, Triangle trig)
{
    return glm::vec3();
}

// TODO
float Rasterizer::zBufferDefault = float();

// TODO
void Rasterizer::UpdateDepthAtPixel(uint32_t x, uint32_t y, Triangle original, Triangle transformed, ImageGrey& ZBuffer)
{
    return;
}

// TODO
void Rasterizer::ShadeAtPixel(uint32_t x, uint32_t y, Triangle original, Triangle transformed, Image& image)
{
    return;
}
