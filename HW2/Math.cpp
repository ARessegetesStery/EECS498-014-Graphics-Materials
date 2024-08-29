#include "Math.h"
#include "Config.h"

#include <algorithm>
#include <cassert>

Vec3::Vec3(float x, float y, float z) : x(x), y(y), z(z)
{
}

Vec3 Vec3::minOfTwo(const Vec3 &v1, const Vec3 &v2)
{
    return {
        std::min(v1.x, v2.x),
        std::min(v1.y, v2.y),
        std::min(v1.z, v2.z)
    };
}

Vec3 Vec3::maxOfTwo(const Vec3& v1, const Vec3& v2) {
    return {
        std::max(v1.x, v2.x),
        std::max(v1.y, v2.y),
        std::max(v1.z, v2.z)
    };
}

Vec3 Vec3::reflect(const Vec3 &inDir, const Vec3 &normal) {
    return 2 * normal + inDir;
}

Vec3 Vec3::operator+(const Vec3 &v) const {
    return {
        x + v.x,
        y + v.y,
        z + v.z
    };
}

Vec3 Vec3::operator-(const Vec3 &v) const {
    return {
        x - v.x, y - v.y, z - v.z
    };
}

Vec3 Vec3::operator-() const {
    return {
        -x, -y, -z
    };
}

Vec3 Vec3::operator*(float f) const {
    return {
        x * f,
        y * f,
        z * f
    };
}

Vec3 Vec3::operator*(const Vec3 &v) const {
    return {
        x * v.x,
        y * v.y,
        z * v.z
    };
}

Vec3 Vec3::operator/(float f) const {
    assert (f > 0.0f);
    return {
        x / f,
        y / f, 
        z / f
    };
}

void Vec3::operator+=(const Vec3 &v){
    *this = *this + v;
}

std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
    return os;
}

Vec3 operator*(float f, const Vec3 &vec) {
    return vec * f;
}

Vec3 Vec3::cross(const Vec3& v) const {
        return {
            y * v.z - z * v.y, 
            z * v.x - x * v.z, 
            x * v.y - y * v.x
        };
    }

    float Vec3::getLength() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    void Vec3::normalize() {
        float len = getLength();
        x /= len;
        y /= len;
        z /= len;
    }

float Vec3::dot(const Vec3& v) const {
    return x * v.x + y * v.y + z * v.z;
}

std::mt19937 Random::generator(SEED);
std::uniform_real_distribution<float> Random::distribution {0.0f, 1.0f};

float Random::randUniformFloat() {
    return distribution(generator);
}

Vec3 localDirToWorld(const Vec3& direction, const Vec3& normal) {
    assert (std::abs(direction.getLength() - 1.0f) < 0.01f);
    // Orthonormal basis (tangent and bitangent) with respect to the normal
    Vec3 tangent, bitangent;

    // Construct the tangent and bitangent to form an orthogonal frame
    if (fabs(normal.x) > fabs(normal.y)) {
        tangent = { normal.z, 0, -normal.x };
    } else {
        tangent =  { 0, -normal.z, normal.y };
    }
    tangent.normalize();
    bitangent = tangent.cross(normal);
    bitangent.normalize();

    // Transform the direction from local coordinates to world coordinates
    Vec3 res = tangent * direction.x + bitangent * direction.y + normal * direction.z;
    res.normalize();
    return res;
}

Vec3 Random::randomHemisphereDirection(const Vec3 &normal) {
    /* 
        Uniformly generate a direction on the hemisphere oriented towards the positive y axis,
            represented by sphere coordinates
    */
    float azimuth = 0.0f;
    float elevation = 0.0f;

    // Convert spherical coordinates to Cartesian
    float x = cos(azimuth) * sin(elevation);
    float y = sin(azimuth) * sin(elevation);
    float z = cos(elevation);

    return localDirToWorld({x, y, z}, normal);
}

Vec3 Random::cosWeightedHemisphere(const Vec3 &normal) {
    /* 
        Generate a direction on the hemisphere oriented towards the positive y axis, 
            cosine-weighted by the elevation angle.
    */
    float azimuth = 0.0f;
    float elevation = 0.0f;

    // Convert spherical coordinates to Cartesian
    float x = cos(azimuth) * sin(elevation);
    float y = sin(azimuth) * sin(elevation);
    float z = cos(elevation);

    return localDirToWorld({x, y, z}, normal);
}
