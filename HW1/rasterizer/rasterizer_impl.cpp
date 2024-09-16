#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

#include "image.hpp"
#include "loader.hpp"
#include "rasterizer.hpp"
#include "entities.hpp"

// TODO
const glm::vec3 pixel_center = glm::vec3(0.5F, 0.5F, 0.0F);

void Rasterizer::DrawPixel(uint32_t x, uint32_t y, Triangle trig, AntiAliasConfig config, uint32_t spp, Image& image,
                           Color color) {
    glm::vec3 pixel(x, y, 0.0F);
    if (config == AntiAliasConfig::NONE)   // if anti-aliasing is off
    {
        glm::vec3 sample_pos = pixel + pixel_center;
        glm::vec3 a1(trig.pos[0]);
        glm::vec3 a2(trig.pos[1]);
        glm::vec3 a3(trig.pos[2]);
        glm::vec3 s1 = glm::cross(sample_pos - a1, a2 - a1);
        glm::vec3 s2 = glm::cross(sample_pos - a2, a3 - a2);
        glm::vec3 s3 = glm::cross(sample_pos - a3, a1 - a3);
        glm::vec3 screen_normal = glm::vec3(0, 0, 1);

        if ((glm::dot(s1, screen_normal) > 0.0F && glm::dot(s2, screen_normal) > 0.0F
             && glm::dot(s3, screen_normal) > 0.0F)
            || (glm::dot(s1, screen_normal) < 0.0F && glm::dot(s2, screen_normal) < 0.0F
                && glm::dot(s3, screen_normal) < 0.0F)) {
            // std::cout << "set pixel (" << x << ", " << y << ")" << std::endl;
            image.Set(x, y, color);
            return;
        }

    } else if (config == AntiAliasConfig::SSAA) {   // if anti-aliasing is on
        uint32_t num_inside = 0;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0F, 1.0);
        for (uint32_t i = 0; i < spp; i++) {
            glm::vec3 x = pixel + glm::vec3(dis(gen), dis(gen), 0);
            glm::vec3 a1(trig.pos[0]);
            glm::vec3 a2(trig.pos[1]);
            glm::vec3 a3(trig.pos[2]);
            glm::vec3 s1 = glm::cross(x - a1, a2 - a1);
            glm::vec3 s2 = glm::cross(x - a2, a3 - a2);
            glm::vec3 s3 = glm::cross(x - a3, a1 - a3);
            glm::vec3 screen_normal = glm::vec3(0, 0, 1);


            if ((glm::dot(s1, screen_normal) > 0 && glm::dot(s2, screen_normal) > 0 && glm::dot(s3, screen_normal) > 0)
                || (glm::dot(s1, screen_normal) < 0 && glm::dot(s2, screen_normal) < 0
                    && glm::dot(s3, screen_normal) < 0)) {
                num_inside++;
            }
        }
        color = (static_cast<float>(num_inside) / static_cast<float>(spp)) * color;
        image.Set(x, y, color);
        return;
    }
}

// TODO
void Rasterizer::AddModel(MeshTransform transform, glm::mat4 rotation) {
    glm::mat4 scale(transform.scale.x, 0.0F, 0.0F, 0.0F,   // first column
                    0.0F, transform.scale.y, 0.0F, 0.0F,   // second column
                    0.0F, 0.0F, transform.scale.z, 0.0F,   // third column
                    0.0F, 0.0F, 0.0F, 1.0F                 // fourth column
    );

    glm::mat4 translation(1.0F, 0.0F, 0.0F, 0.0F,   // first column
                          0.0F, 1.0F, 0.0F, 0.0F,   // second column
                          0.0F, 0.0F, 1.0F, 0.0F,   // third colum
                          transform.translation.x, transform.translation.y, transform.translation.z,
                          1.0F   // fourth column
    );

    model.push_back(translation * rotation * scale);
}

// TODO
void Rasterizer::SetView() {
    const Camera& camera = this->loader.GetCamera();
    glm::vec3 camera_pos = camera.pos;
    glm::vec3 camera_look_at = camera.lookAt;

    glm::vec3 gaze = glm::normalize(camera_look_at - camera_pos);
    glm::vec3 camera_side = glm::normalize(glm::cross(gaze, camera.up));
    glm::vec3 up = glm::normalize(camera.up);

    // TODO change this line to the correct view matrix
    glm::mat4 translate_to_origin(1.0F, 0.0F, 0.0F, 0.0F,                          // first column
                                  0.0F, 1.0F, 0.0F, 0.0F,                          // second column
                                  0.0F, 0.0F, 1.0F, 0.0F,                          // third colum
                                  -camera_pos.x, -camera_pos.y, -camera_pos.z, 1.0F   // fourth column
    );

    glm::mat4 model_to_view( camera_side.x, up.x, -gaze.x, 0.0F, // first column
                            camera_side.y, up.y, -gaze.y, 0.0F, //second column
                            camera_side.z, up.z, -gaze.z, 0.0F, // third column
                            0.0F, 0.0F, 0.0F, 1.0F                               // fourth column
    );
    this->view = model_to_view * translate_to_origin;
}

// TODO
void Rasterizer::SetProjection() {
    const Camera& camera = this->loader.GetCamera();

    float near_clip = -camera.nearClip;   // near clipping distance, strictly positive
    float far_clip = -camera.farClip;     // far clipping distance, strictly positive

    float width = static_cast<float>(camera.width);
    float height = static_cast<float>(camera.height);

    // TODO change this line to the correct projection matrix
    glm::mat4 tranlsate_to_center(1.0F, 0.0F, 0.0F, 0.0F,                                 // first column
                                  0.0F, 1.0F, 0.0F, 0.0F,                                 // second column
                                  0.0F, 0.0F, 1.0F, 0.0F,                                 // third colum
                                  0.0F, 0.0F, -(near_clip + far_clip) / 2, 1.0F   // fourth column
    );

    glm::mat4 scale_to_bounding_box(2.0F / width, 0.0F, 0.0F, 0.0F,                    // first column
                                    0.0F, 2 / height, 0.0F, 0.0F,                   // second column
                                    0.0F, 0.0F, 2 / (near_clip - far_clip), 0.0F,   // third column
                                    0.0F, 0.0F, 0.0F, 1.0F                          // fourth column
    );

    glm::mat4 perspective(
        near_clip, 0.0F, 0.0F, 0.0F, // first column,
        0.0F, near_clip, 0.0F, 0.0F, // second column,
        0.0F, 0.0F, near_clip+far_clip, 1.0F, // third column,
        0.0F, 0.0F, -near_clip*far_clip, 0.0F // fourth column,
    );

    this->projection = scale_to_bounding_box * tranlsate_to_center * perspective;
}

// TODO
void Rasterizer::SetScreenSpace() {
    float width = static_cast<float>(this->loader.GetWidth());
    float height = static_cast<float>(this->loader.GetHeight());

    // TODO change this line to the correct screenspace matrix
    this->screenspace = glm::mat4(width/2, 0.0F, 0.0F, 0.0F, // first column
                                  0.0F, height/2, 0.0F, 0.0F, // second column
                                  0.0F, 0.0F, 1.0F, 0.0F, // third column
                                  width/2, height/2, 0.0F, 1.0F // fourth column
    );

    // this->screenspace = glm::mat4(1.0F);
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
