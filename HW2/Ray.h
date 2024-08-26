#pragma once

#include "Math.h"

class Ray {
public:
    Vec3 pos;
    Vec3 dir;

    Ray() = default;
    Ray(const Vec3& p, const Vec3& d);

    Vec3 travel(float time) const;
    bool isNormalized() const;
};