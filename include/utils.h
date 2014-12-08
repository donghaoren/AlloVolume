#ifndef ALLOVOLUME_UTILS_H_INCLUDED
#define ALLOVOLUME_UTILS_H_INCLUDED

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
    GPUEnable Vector operator * (const Vector& v) const { return Vector(x * v.x, y * v.y, z * v.z); }
    GPUEnable Vector operator / (const Vector& v) const { return Vector(x / v.x, y / v.y, z / v.z); }
    GPUEnable Vector operator - () const { return Vector(-x, -y, -z); }
    GPUEnable Vector operator * (float s) const { return Vector(x * s, y * s, z * s); }
    GPUEnable Vector operator / (float s) const { return Vector(x / s, y / s, z / s); }
    GPUEnable float dot(const Vector& v) const { return x * v.x + y * v.y + z * v.z; }
    GPUEnable float len2() const { return x * x + y * y + z * z; }
    double len2_double() const { return (double)x * (double)x + (double)y * (double)y + (double)z * (double)z; }
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

struct Vector4 {
    float x, y, z, w;
};

struct Quaternion {
    Vector v;
    float w;

    GPUEnable Quaternion() { }
    GPUEnable Quaternion(float w_, float x_, float y_, float z_) : w(w_), v(x_, y_, z_) { }
    GPUEnable Quaternion(float w_, const Vector& v_) : w(w_), v(v_) { }

    GPUEnable static Quaternion Rotation(const Vector& v, float alpha) {
        return Quaternion(std::cos(alpha / 2.0f), v.normalize() * std::sin(alpha / 2.0f));
    }

    GPUEnable float len() const{
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + w * w);
    }
    GPUEnable Quaternion operator + (const Quaternion& q) const {
        return Quaternion(w + q.w, v + q.v);
    }
    GPUEnable Quaternion operator - (const Quaternion& q) const {
        return Quaternion(w - q.w, v - q.v);
    }
    GPUEnable Quaternion operator - () const {
        return Quaternion(-w, -v);
    }
    GPUEnable Quaternion operator * (float s) const {
        return Quaternion(w * s, v * s);
    }
    GPUEnable Quaternion operator / (float s) const {
        return Quaternion(w / s, v / s);
    }
    GPUEnable Quaternion normalize() const {
        float l = len();
        return *this * (1.0f / l);
    }
    GPUEnable Quaternion operator * (const Quaternion& q) const {
        return Quaternion(w * q.w - v.dot(q.v), v.cross(q.v) + q.v * w + v * q.w);
    }
    GPUEnable Quaternion operator * (const Vector& v) const {
        return *this * Quaternion(0.0f, v);
    }
    GPUEnable Quaternion inversion() const {
        return Quaternion(w, -v);
    }
    GPUEnable Vector rotate(const Vector& v) const {
        return ((*this) * v * inversion()).v;
    }
};

GPUEnable float num_min(float a, float b) { return a < b ? a : b; }
GPUEnable float num_max(float a, float b) { return a > b ? a : b; }
GPUEnable float clamp01(float a) { return a < 0 ? 0 : (a > 1 ? 1 : a); }
// integer division upper round.
inline int diviur(int a, int b) {
    if(a % b == 0) return a / b;
    return a / b + 1;
}

struct Color {
    float r, g, b, a;
    GPUEnable Color() { }
    GPUEnable Color(float r_, float g_, float b_, float a_) : r(r_), g(g_), b(b_), a(a_) { }
    GPUEnable Color(float r_, float g_, float b_) : r(r_), g(g_), b(b_), a(1) { }
    GPUEnable Color blendTo(Color c) {
        return Color(
            a * r + (1.0 - a) * c.r,
            a * g + (1.0 - a) * c.g,
            a * b + (1.0 - a) * c.b,
            a * a + (1.0 - a) * c.a
        );
    }
    GPUEnable Color blendToDifferential(Color c, float ratio) {
        float dt = pow(1 - a, ratio);
        return Color(
            (1.0 - dt) * r + dt * c.r,
            (1.0 - dt) * g + dt * c.g,
            (1.0 - dt) * b + dt * c.b,
            (1.0 - dt) * a + dt * c.a
        );
    }

    GPUEnable Color& operator += (const Color& v) { r += v.r; g += v.g; b += v.b; a += v.a; return *this; }
    GPUEnable Color& operator -= (const Color& v) { r -= v.r; g -= v.g; b -= v.b; a -= v.a; return *this; }
    GPUEnable Color& operator *= (const Color& v) { r *= v.r; g *= v.g; b *= v.b; a *= v.a; return *this; }
    GPUEnable Color& operator /= (const Color& v) { r /= v.r; g /= v.g; b /= v.b; a /= v.a; return *this; }
    GPUEnable Color& operator *= (float s) { r *= s; g *= s; b *= s; a *= s; return *this; }
    GPUEnable Color& operator /= (float s) { r /= s; g /= s; b /= s; a /= s; return *this; }
    GPUEnable Color operator + (const Color& v) const { return Color(r + v.r, g + v.g, b + v.b, a + v.a); }
    GPUEnable Color operator - (const Color& v) const { return Color(r - v.r, g - v.g, b - v.b, a - v.a); }
    GPUEnable Color operator * (const Color& v) const { return Color(r * v.r, g * v.g, b * v.b, a * v.a); }
    GPUEnable Color operator / (const Color& v) const { return Color(r / v.r, g / v.g, b / v.b, a / v.a); }
    GPUEnable Color operator * (float v) const { return Color(r * v, g * v, b * v, a * v); }
    GPUEnable Color operator / (float v) const { return Color(r / v, g / v, b / v, a / v); }
};

}

#endif
