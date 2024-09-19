#include <cstdint>

#include "image.hpp"
#include "loader.hpp"
#include "rasterizer.hpp"

#include <iostream>
#include "../thirdparty/glm/gtx/string_cast.hpp"


static inline bool inside(uint32_t x, uint32_t y, Triangle trig){
    glm::vec3 X(x, y, 0);
    glm::vec3 A1(trig.pos[0].x, trig.pos[0].y, 0);
    glm::vec3 A2(trig.pos[1].x, trig.pos[1].y, 0);
    glm::vec3 A3(trig.pos[2].x, trig.pos[2].y, 0);

    auto S1 = glm::cross(X - A1, A2 - A1);
    auto S2 = glm::cross(X - A2, A3 - A2);
    auto S3 = glm::cross(X - A3, A1 - A3);

    bool all_pos = S1.z > 0 && S2.z > 0 && S3.z > 0;
    bool all_neg = S1.z < 0 && S2.z < 0 && S3.z < 0;

    return all_pos || all_neg;
}

// TODO
void Rasterizer::DrawPixel(uint32_t x, uint32_t y, Triangle trig, AntiAliasConfig config, uint32_t spp, Image& image, Color color)
{
    if (config == AntiAliasConfig::NONE)            // if anti-aliasing is off
    {
        if(inside(x, y, trig)){
            image.Set(x, y, color);
        }
    }
    else if (config == AntiAliasConfig::SSAA)       // if anti-aliasing is on
    {
        int count = 0;
        float s = glm::sqrt(spp);

        for(int i=0;i<s;i++){
            float sx = -0.5 + ((i+1) + 0.5)/s;
            for(int j=0;j<s;j++){
                float sy = -0.5 + ((j+1) + 0.5)/s;

                count += inside(x + sx, y + sy, trig);
            }
        }
        color = ((float)count/spp) * color;
        image.Set(x, y, color);
    }

    return;
}

// TODO
void Rasterizer::AddModel(MeshTransform transform, glm::mat4 rotation)
{
    /* model.push_back( model transformation constructed from translation, rotation and scale );*/
    glm::mat4 S(1.);
    S[0][0] = transform.scale.x;
    S[1][1] = transform.scale.y;
    S[2][2] = transform.scale.z;

    glm::mat4 matrix = S * rotation;
    matrix[3] = glm::vec4(transform.translation, 1.);

    this->model.push_back(matrix);
    return;
}

// TODO
void Rasterizer::SetView()
{
    const Camera& camera = this->loader.GetCamera();
    glm::vec3 cameraPos = camera.pos;
    glm::vec3 cameraLookAt = camera.lookAt;
    glm::vec3 cameraUp = camera.up;

    // TODO change this line to the correct view matrix
    glm::mat4 T(1.);
    T[3] = glm::vec4(-cameraPos, 1);

    glm::vec3 t = glm::normalize(camera.up);
    glm::vec3 g = glm::normalize(cameraLookAt - cameraPos);
    glm::vec3 r = glm::normalize(glm::cross(g, t));

    glm::mat4 R(1.);
    R[0] = glm::vec4(r, 0);
    R[1] = glm::vec4(t, 0);
    R[2] = glm::vec4(-g, 0);

    this->view = glm::transpose(R) * T;
    return;
}

// TODO
void Rasterizer::SetProjection()
{
    const Camera& camera = this->loader.GetCamera();

    float nearClip = camera.nearClip;                   // near clipping distance, strictly positive
    float farClip = camera.farClip;                     // far clipping distance, strictly positive
    
    float width = camera.width;
    float height = camera.height;
    
    // TODO change this line to the correct projection matrix
    this->projection = glm::mat4(1.);
    return;
}

// TODO
void Rasterizer::SetScreenSpace()
{
    float width = this->loader.GetWidth();
    float height = this->loader.GetHeight();

    // TODO change this line to the correct screenspace matrix
    this->screenspace = glm::mat4(1.);
    this->screenspace[0][0] = width/2.0;
    this->screenspace[1][1] = width/2.0;
    this->screenspace[3] = glm::vec4(width/2.0, height/2.0, 0, 1);
    return;
}

// TODO
glm::vec3 Rasterizer::BarycentricCoordinate(glm::vec2 pos, Triangle trig)
{
    float Xa = trig.pos[0].x, Ya = trig.pos[0].y;
    float Xb = trig.pos[1].x, Yb = trig.pos[1].y;
    float Xc = trig.pos[2].x, Yc = trig.pos[2].y;
    float X = pos.x, Y = pos.y;

    float alpha = (-(X-Xb)*(Yc-Yb) + (Y-Yb)*(Xc-Xb))/(-(Xa-Xb)*(Yc-Yb) + (Ya-Yb)*(Xc-Xb));
    float beta = (-(X-Xc)*(Ya-Yc) + (Y-Yc)*(Xa-Xc))/(-(Xb-Xc)*(Ya-Yc) + (Yb-Yc)*(Xa-Xc));

    return glm::vec3(
        alpha,
        beta,
        1 - alpha - beta
    );
}

// TODO
float Rasterizer::zBufferDefault = float();

// TODO
void Rasterizer::UpdateDepthAtPixel(uint32_t x, uint32_t y, Triangle original, Triangle transformed, ImageGrey& ZBuffer)
{

    float result;
    ZBuffer.Set(x, y, result);

    return;
}

// TODO
void Rasterizer::ShadeAtPixel(uint32_t x, uint32_t y, Triangle original, Triangle transformed, Image& image)
{

    Color result;
    image.Set(x, y, result);

    return;
}
