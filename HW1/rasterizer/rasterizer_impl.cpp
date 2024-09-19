#include <cstdint>

#include "image.hpp"
#include "loader.hpp"
#include "rasterizer.hpp"


static inline bool inside(glm::vec3 bc, Triangle trig){
    return (0 <= bc.x && bc.x <= 1) && (0 <= bc.y && bc.y <= 1) && (bc.z >= 0);
}

// TODO
void Rasterizer::DrawPixel(uint32_t x, uint32_t y, Triangle trig, AntiAliasConfig config, uint32_t spp, Image& image, Color color)
{
    glm::vec3 bc = this->BarycentricCoordinate(glm::vec2(x + .5, y + .5), trig);

    if (config == AntiAliasConfig::NONE)            // if anti-aliasing is off
    {
        color = inside(bc, trig) ? color : Color::Black;
    }
    else if (config == AntiAliasConfig::SSAA)       // if anti-aliasing is on
    {
        int count = 0;
        float s = glm::sqrt(spp);

        for(int i=0;i<s;i++){
            float sx = -0.5 + ((i+1) + 0.5)/s;
            for(int j=0;j<s;j++){
                float sy = -0.5 + ((j+1) + 0.5)/s;

                count += inside(this->BarycentricCoordinate(glm::vec2(x + sx, y+sy), trig), trig);
            }
        }
        color = ((float)count/spp) * color;
    }

    // if the pixel is inside the triangle
    image.Set(x, y, color);
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

    float width = this->loader.GetWidth();
    float height = this->loader.GetHeight();

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
