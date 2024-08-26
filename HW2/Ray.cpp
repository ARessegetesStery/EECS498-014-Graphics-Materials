#include "Ray.h"

Ray::Ray(const Vec3 &p, const Vec3 &d) : pos(p), dir(d) {
}

Vec3 Ray::travel(float time) const {
    return pos + time * dir;
}

bool Ray::isNormalized() const {
    return std::abs(dir.getLength() - 1.0f) < 1E-3;
}
