#include "Scene.h"
#include "Config.h"

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>

constexpr float GAMMA = 0.6f;

void UpdateProgress(float progress)
{
    static bool checkPoints[10] = {false};
    if constexpr(DEBUG) {
        int barWidth = 32;

        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    } else {
        int index = progress * 10;
        if (!checkPoints[index]) {
            std::cout << index * 10 << "%\n";
            checkPoints[index] = true;
        }
    }
};

inline float toneMap(float x) {
    return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
}

unsigned char toLinear(float sRGB) {
    return 255 * std::pow(std::clamp(sRGB, 0.0f, 1.0f), GAMMA);
}

int main() {
    using namespace std::chrono;
    auto startTime = high_resolution_clock::now();

    Scene scene;
    scene.addObjects(OBJ_PATH, MTL_SEARCH_DIR);
    scene.constructBVH();
    
    auto timeAfterVBVH = high_resolution_clock::now();
    std::cout << "BVH Construction time in seconds: " << duration_cast<seconds>(timeAfterVBVH - startTime).count() << '\n';
    int width = RESOLUTION, height = RESOLUTION;
    std::vector<std::vector<Vec3>> image(height, std::vector<Vec3>(width));
    Vec3 cameraPos = {
        0.0f, 1.0f, 4.0f
    };

    if constexpr(!DEBUG) {
        std::cout << "Debug mode disabled. Progress output will be in brief." <<  '\n';
    }

    // x: right
    // y: up
    // z: outwards
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            Vec3 worldPos = {
                (float)x / width - 0.5f, 
                1.5f - (float)y / height,
                (cameraPos.z + 1.0f) / 2
            };
            Ray ray {
                cameraPos,
                worldPos - cameraPos,
            };
            ray.dir.normalize();
            Vec3 value {};
            for (int i = 0; i < SPP; i++) {
                value += scene.trace(ray, MAX_DEPTH);
            }
            image[y][x] = value / SPP;
            UpdateProgress((float)(y * width + x) / (width * height));
        }
    }
    std::cout << std::endl;

    auto finishTime = high_resolution_clock::now();
    std::cout << "Rendering time in seconds: " << duration_cast<seconds>(finishTime - timeAfterVBVH).count() << '\n';

    std::filesystem::path outPath = std::filesystem::absolute(OUTPUT_PATH);

    FILE* fp = fopen(outPath.string().c_str(), "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            static unsigned char color[3];
            color[0] = toLinear(image[y][x].x);
            color[1] = toLinear(image[y][x].y);
            color[2] = toLinear(image[y][x].z);
            fwrite(color, 1, 3, fp);
        }
    }
    fclose(fp);
    std::cout << "Output image written to " << outPath << '\n';
}