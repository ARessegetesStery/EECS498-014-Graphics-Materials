#pragma once

#include "Math.h"
#include "Ray.h"
#include <vector>

class Mesh {
public:
    Vec3 a;
    Vec3 b;
    Vec3 c;
    Vec3 normal;
    float area;
    Mesh(const Vec3& a, const Vec3& b, const Vec3& c);
    /**
     * returns the time that the ray travels before hitting this mesh
     * returns FLOAT_MAX if they don't intersect
    */
    float intersect(const Ray& ray);
    /**
     * @brief sample a random point on the mesh surface
    */
    Vec3 sample() const;
    bool isPointInsideMesh(const Vec3& point) const;
};

class BoundingBox {
public:
    Vec3 minCorner;
    Vec3 maxCorner;

    static BoundingBox boxUnion(const BoundingBox& b1, const BoundingBox& b2);
    static BoundingBox constructFromMesh(const Mesh&);
    void boxUnion(const BoundingBox& other);

    enum class Extent {
        x = 0,
        y, z
    };

    Vec3 centroid() const;
    Vec3 diagonal() const;
    Extent maxExtent() const;

    /**
     * return the shortest time that the ray travels before hitting the bounding box
     * return FLOAT_MAX if they don't intersect
     * a negative return value means the ray starts from inside the box
    */
    float intersect(const Ray& ray) const;
};

// forward declaration.
// We're going to use Intersection in Object class, 
// and use Object in the Intersection struct
struct Intersection;

class Object {
public:
    std::string name = ""; // for debug
    BoundingBox box;
    std::vector<Mesh> meshes;
    float area;
    // Vec3 ka; /* ambient color */
    Vec3 kd; /* albedo */
    Vec3 ke; /* emission */
    bool hasEmission = false;
    /**
     * @brief sample a surface point
    */
    Intersection sample() const;
    void constructBoundingBox();
};

struct Intersection {
    bool happened = false;
    float time = std::numeric_limits<float>::max();
    const Object* object = nullptr;
    Vec3 pos;
    const Mesh* mesh = nullptr;

    /* helper functions*/
    Vec3 getNormal() const;
    Vec3 getDiffuseColor() const;
    Vec3 getEmission() const;

    Vec3 calcBRDF(const Vec3& inDir, const Vec3& outDir) const;
};

class BVHNode {
public:
    BVHNode *left = nullptr, *right = nullptr;
    BoundingBox box;
    Object* object = nullptr;

    Intersection intersect(const Ray& ray);
    bool isLeaf() const;
};

class BVH {
public:
    BVHNode *root = nullptr;
    static BVHNode* build(const std::vector<Object*>& objects);
    /**
     * @param box the bounding box of all the objects
    */
    static std::pair<std::vector<Object*>, std::vector<Object*>> splitObjects(std::vector<Object*> objects, const BoundingBox& box);
};