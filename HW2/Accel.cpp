#include "Accel.h"
#include "Config.h"
#include <cassert>
#include <algorithm>

// Very important! Set it to 1E-9 and you'll likely see self-occlusion artifacts.
constexpr float MIN_TRAVEL_TIME = 1e-3;

BoundingBox BoundingBox::boxUnion(const BoundingBox& b1, const BoundingBox& b2) {
    return {
        Vec3::minOfTwo(b1.minCorner, b2.minCorner), 
        Vec3::maxOfTwo(b1.maxCorner, b2.maxCorner)
    };
}

BoundingBox BoundingBox::constructFromMesh(const Mesh &mesh) {
    return {
        Vec3::minOfTwo(Vec3::minOfTwo(mesh.a, mesh.b), mesh.c), 
        Vec3::maxOfTwo(Vec3::maxOfTwo(mesh.a, mesh.b), mesh.c)
    };
}

void BoundingBox::boxUnion(const BoundingBox &other) {
    this->minCorner = Vec3::minOfTwo(this->minCorner, other.minCorner);
    this->maxCorner = Vec3::maxOfTwo(this->maxCorner, other.maxCorner);
}

Vec3 BoundingBox::centroid() const {
    return (minCorner + maxCorner) / 2;
}

Vec3 BoundingBox::diagonal() const {
    return {maxCorner - minCorner};
}

BoundingBox::Extent BoundingBox::maxExtent() const {
    Vec3 d = diagonal();
    float maxDim = std::max({d.x, d.y, d.z});
    if (maxDim == d.x) return Extent::x;
    if (maxDim == d.y) return Extent::y;
    return Extent::z;
}

float BoundingBox::intersect(const Ray &ray) const {
    // assert (ray.isNormalized());

    float tmin = (minCorner.x - ray.pos.x) / ray.dir.x;
    float tmax = (maxCorner.x - ray.pos.x) / ray.dir.x;

    if (tmin > tmax) std::swap(tmin, tmax);

    float tymin = (minCorner.y - ray.pos.y) / ray.dir.y;
    float tymax = (maxCorner.y - ray.pos.y) / ray.dir.y;

    if (tymin > tymax) std::swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return std::numeric_limits<float>::max();

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (minCorner.z - ray.pos.z) / ray.dir.z;
    float tzmax = (maxCorner.z - ray.pos.z) / ray.dir.z;

    if (tzmin > tzmax) std::swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax)) {
        return std::numeric_limits<float>::max();
    }

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    if (tmax > 0.0f) {
        return tmin;
    }
    return std::numeric_limits<float>::max();
}

Intersection Object::sample() const {
    float currentArea = 0;
    size_t idx = 0;
    float sampleArea = Random::randUniformFloat() * area;
    while (true) {
        currentArea += meshes[idx].area;
        if (currentArea + std::numeric_limits<float>::epsilon() >= sampleArea) break;
        idx++;
    }
    Intersection inter;
    inter.happened = true;
    inter.mesh = &meshes[idx];
    inter.object = this;
    inter.pos = meshes[idx].sample();
    return inter;
}

void Object::constructBoundingBox()
{
    assert (!meshes.empty());

    box = BoundingBox::constructFromMesh(meshes[0]);
    for (size_t i = 1; i < meshes.size(); i++) {
        box.boxUnion(BoundingBox::constructFromMesh(meshes[i]));
    }
}

BVHNode* BVH::build(const std::vector<Object *> &objects) {
    assert (!objects.empty());
    BoundingBox box = objects[0]->box;
    for (size_t i = 1; i < objects.size(); i++) {
        box.boxUnion(objects[i]->box);
    }

    auto node = new BVHNode();
    node->box = box;
    auto numObjects = objects.size();

    if (numObjects == 1) {
        node->object = objects[0];
    } else {
        auto [firstHalf, secondHalf] = splitObjects(objects, box);
        node->left = build(firstHalf);
        node->right = build(secondHalf);
    }
    
    return node;
}

std::pair<std::vector<Object *>, std::vector<Object *>> BVH::splitObjects(std::vector<Object *> objects, const BoundingBox& box) {
    assert (objects.size() > 1);

    BoundingBox::Extent maxExt = box.maxExtent();
    if (maxExt == BoundingBox::Extent::x) {
        std::sort(objects.begin(), objects.end(), 
        [](const Object* lhs, const Object* rhs){
            return lhs->box.centroid().x < rhs->box.centroid().x;
        });
    } else if (maxExt == BoundingBox::Extent::y) {
        std::sort(objects.begin(), objects.end(), 
        [](const Object* lhs, const Object* rhs){
            return lhs->box.centroid().y < rhs->box.centroid().y;
        });
    } else {
        std::sort(objects.begin(), objects.end(), 
        [](const Object* lhs, const Object* rhs){
            return lhs->box.centroid().z < rhs->box.centroid().z;
        });
    }

    auto midIter = objects.begin() + objects.size() / 2;

    return {
        {objects.begin(), midIter},
        {midIter, objects.end()}
    };
}

Mesh::Mesh(const Vec3 &a, const Vec3 &b, const Vec3 &c) 
    : a(a), b(b), c(c) {
    Vec3 p = b-a, q = c-b;
    normal = (p).cross(q);
    area = normal.getLength() / 2;
    normal.normalize();
}

float Mesh::intersect(const Ray &ray) {
    if constexpr(DEBUG) {
        assert (ray.isNormalized());
    }
    float proj = ray.dir.dot(normal);

    // we ignore intersection with the backface
    if (proj > 0) return std::numeric_limits<float>::max();
    
    // find the plane defined by this triangle mesh
    // plane expression: n dot x + delta = 0
    float delta = -normal.dot(a);

    float time = - (delta + normal.dot(ray.pos)) / proj;
    if (time < 0) return std::numeric_limits<float>::max();
    
    Vec3 inter = ray.travel(time);
    if (isPointInsideMesh(inter)) return time;
    return std::numeric_limits<float>::max();
}

Vec3 Mesh::sample() const {
    float m = std::sqrt(Random::randUniformFloat()), n = Random::randUniformFloat();
    return a * (1.0f - m) + b * (m * (1.0f - n)) + c * (m * n);
}

bool Mesh::isPointInsideMesh(const Vec3& point) const {
    Vec3 i = point - a, j = point - b, k = point - c;
    auto isValid = [this](const Vec3& m, const Vec3& n) -> bool {
        return m.cross(n).dot(normal) > 0;
        };
    return isValid(i, j) && isValid(j, k) && isValid(k, i);
}

Intersection BVHNode::intersect(const Ray &ray) {
    if (box.intersect(ray) >= std::numeric_limits<float>::max()) {
        return {};
    }
    if (isLeaf()) {
        float shortestHitTime = std::numeric_limits<float>::max();
        Mesh* targetMesh;
        for (auto& mesh : object->meshes) {
            float hitTime = mesh.intersect(ray);
            if (hitTime > MIN_TRAVEL_TIME && hitTime < shortestHitTime) {
                shortestHitTime = hitTime;
                targetMesh = &mesh;
            }
        }

        if (shortestHitTime == std::numeric_limits<float>::max()) return {};

        Intersection result;
        result.happened = true;
        result.time = shortestHitTime;
        result.mesh = targetMesh;
        result.pos = ray.travel(shortestHitTime);
        result.object = object;
        return result;
    }
    Intersection leftIntersect = left->intersect(ray);
    Intersection rightIntersect = right->intersect(ray);
    if (leftIntersect.happened) {
        if (rightIntersect.happened) {
            return leftIntersect.time < rightIntersect.time ? leftIntersect : rightIntersect;
        }
        return leftIntersect;
    }
    return rightIntersect;
}

bool BVHNode::isLeaf() const {
    return !left && !right;
}

Vec3 Intersection::getNormal() const {
    return mesh->normal;
}

Vec3 Intersection::getDiffuseColor() const {
    return object->kd;
}

Vec3 Intersection::getEmission() const {
    return object->ke;
}

Vec3 Intersection::calcBRDF(const Vec3& inDir, const Vec3& outDir) const {
    assert (happened);
    const Vec3& normal = mesh->normal;
    if (inDir.dot(normal) > 0 || outDir.dot(normal) < 0) return {};
    // BRDF of a diffuse object
    /*-----------------------------------------------------------*/
    return getDiffuseColor() / PI;
    /*-----------------------------------------------------------*/
}
