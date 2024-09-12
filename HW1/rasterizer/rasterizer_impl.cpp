#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

#include "image.hpp"
#include "loader.hpp"
#include "rasterizer.hpp"

// TODO

void Rasterizer::DrawPixel(uint32_t x, uint32_t y, Triangle trig, AntiAliasConfig config, uint32_t spp, Image& image,
                           Color color) {
    glm::vec3 pixel(x, y, 0);
    if (config == AntiAliasConfig::NONE)   // if anti-aliasing is off
    {
        glm::vec3 X = pixel + glm::vec3(0.5, 0.5, 0);
        glm::vec3 a1(trig.pos[0]);
        glm::vec3 a2(trig.pos[1]);
        glm::vec3 a3(trig.pos[2]);
        glm::vec3 s1 = glm::cross(X - a1, a2 - a1);
        glm::vec3 s2 = glm::cross(X - a2, a3 - a2);
        glm::vec3 s3 = glm::cross(X - a3, a1 - a3);
        glm::vec3 screen_normal = glm::vec3(0, 0, 1);


        if ((glm::dot(s1, screen_normal) > 0 && glm::dot(s2, screen_normal) > 0 && glm::dot(s3, screen_normal) > 0)
            || (glm::dot(s1, screen_normal) < 0 && glm::dot(s2, screen_normal)) < 0
                 && glm::dot(s3, screen_normal) < 0) {
            // std::cout << "set pixel (" << x << ", " << y << ")" << std::endl;
            image.Set(x, y, color);
            return;
        }

    } else if (config == AntiAliasConfig::SSAA) {   // if anti-aliasing is on
        uint32_t num_inside = 0;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (uint32_t i = 0; i < spp; i++) {
            glm::vec3 X = pixel + glm::vec3(dis(gen), dis(gen), 0);
            glm::vec3 a1(trig.pos[0]);
            glm::vec3 a2(trig.pos[1]);
            glm::vec3 a3(trig.pos[2]);
            glm::vec3 s1 = glm::cross(X - a1, a2 - a1);
            glm::vec3 s2 = glm::cross(X - a2, a3 - a2);
            glm::vec3 s3 = glm::cross(X - a3, a1 - a3);
            glm::vec3 screen_normal = glm::vec3(0, 0, 1);


            if ((glm::dot(s1, screen_normal) > 0 && glm::dot(s2, screen_normal) > 0 && glm::dot(s3, screen_normal) > 0)
                || (glm::dot(s1, screen_normal) < 0 && glm::dot(s2, screen_normal) < 0
                    && glm::dot(s3, screen_normal) < 0)) {
                num_inside++;
            }
        }
        color = ((double) num_inside / (double) spp) * color;
        image.Set(x, y, color);
        return;
    }
    return;
}

// TODO
void Rasterizer::AddModel(MeshTransform transform, glm::mat4 rotation) {
    /* model.push_back( model transformation constructed from translation, rotation and scale );*/
    return;
}

// TODO
void Rasterizer::SetView() {
    const Camera& camera = this->loader.GetCamera();
    glm::vec3 cameraPos = camera.pos;
    glm::vec3 cameraLookAt = camera.lookAt;

    // TODO change this line to the correct view matrix
    this->view = glm::mat4(1.);

    return;
}

// TODO
void Rasterizer::SetProjection() {
    const Camera& camera = this->loader.GetCamera();

    float nearClip = camera.nearClip;   // near clipping distance, strictly positive
    float farClip = camera.farClip;     // far clipping distance, strictly positive

    float width = this->loader.GetWidth();
    float height = this->loader.GetHeight();

    // TODO change this line to the correct projection matrix
    this->projection = glm::mat4(1.);

    return;
}

// TODO
void Rasterizer::SetScreenSpace() {
    float width = this->loader.GetWidth();
    float height = this->loader.GetHeight();

    // TODO change this line to the correct screenspace matrix
    this->screenspace = glm::mat4(1.);

    return;
}

// TODO
glm::vec3 Rasterizer::BarycentricCoordinate(glm::vec2 pos, Triangle trig) {
    return glm::vec3();
}

// TODO
float Rasterizer::zBufferDefault = float();

// TODO
void Rasterizer::UpdateDepthAtPixel(uint32_t x, uint32_t y, Triangle original, Triangle transformed,
                                    ImageGrey& ZBuffer) {
    float result;
    ZBuffer.Set(x, y, result);

    return;
}

// TODO
void Rasterizer::ShadeAtPixel(uint32_t x, uint32_t y, Triangle original, Triangle transformed, Image& image) {
    Color result;
    image.Set(x, y, result);

    return;
}
