#ifndef ALLOVOLUME_UTILS_H
#define ALLOVOLUME_UTILS_H

#include <cmath>

namespace allovolume {

#ifdef __device__
#define GPUEnable inline __device__ __host__
#else
#define GPUEnable inline
#endif

#define PI 3.141592653589793

struct Vector {
    float x, y, z;
    GPUEnable Vector() { }
    GPUEnable Vector(float v) { x = y = z = v; }
    GPUEnable Vector(float x_, float y_, float z_) : x(x_), y(y_), z(z_) { }
    GPUEnable Vector& operator += (const Vector& v) { x += v.x; y += v.y; z += v.z; return *this; }
    GPUEnable Vector& operator -= (const Vector& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    GPUEnable Vector& operator *= (float s) { x *= s; y *= s; z *= s; return *this; }
    GPUEnable Vector& operator /= (float s) { x /= s; y /= s; z /= s; return *this; }
    GPUEnable Vector operator + (const Vector& v) const { return Vector(x + v.x, y + v.y, z + v.z); }
    GPUEnable Vector operator - (const Vector& v) const { return Vector(x - v.x, y - v.y, z - v.z); }
    GPUEnable Vector operator - () const { return Vector(-x, -y, -z); }
    GPUEnable Vector operator * (float s) const { return Vector(x * s, y * s, z * s); }
    GPUEnable Vector operator / (float s) const { return Vector(x / s, y / s, z / s); }
    GPUEnable float operator * (const Vector& v) const { return x * v.x + y * v.y + z * v.z; }
    GPUEnable float len2() const { return x * x + y * y + z * z; }
    GPUEnable float len() const { return std::sqrt(x * x + y * y + z * z); }
    GPUEnable Vector normalize() const { return *this / len(); }
    GPUEnable bool operator <= (const Vector& v) const { return x <= v.x && y <= v.y && z <= v.z; }
    GPUEnable bool operator >= (const Vector& v) const { return x >= v.x && y >= v.y && z >= v.z; }
    GPUEnable Vector cross(const Vector& v) const {
        return Vector(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
};

GPUEnable float num_min(float a, float b) { return a < b ? a : b; }
GPUEnable float num_max(float a, float b) { return a > b ? a : b; }
GPUEnable float clamp01(float a) { return a < 0 ? 0 : (a > 1 ? 1 : a); }

struct Color {
    float r, g, b, a;
    Color() { }
    GPUEnable Color(float r_, float g_, float b_, float a_) : r(r_), g(g_), b(b_), a(a_) { }
    GPUEnable Color blendTo(Color c) {
        return Color(
            a * r + (1.0 - a) * c.r,
            a * g + (1.0 - a) * c.g,
            a * b + (1.0 - a) * c.b,
            a * a + (1.0 - a) * c.a
        );
    }
};

}

#endif
