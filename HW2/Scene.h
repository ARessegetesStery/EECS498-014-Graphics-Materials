#pragma once

#include "tiny_obj_loader.h"
#include "Accel.h"

#include <string>
#include <vector>

class Scene {
public:
    static tinyobj::ObjReader reader;
    std::vector<Object*> objects;
    std::vector<Object*> lights;
    BVH bvh;

    float lightArea = 0;

    void addObjects(const std::string& modelPath, const std::string& searchPath);
    void constructBVH();
    Intersection getIntersection(const Ray& ray);
    /**
     * @brief sample a point from the first object in the light vector
     * @todo add support for multiple light objects
    */
    Intersection sampleLight() const;
    Vec3 trace(const Ray& ray, int bounces = 2, bool discardEmission = false);
    ~Scene();
};