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

template<typename FloatT>
struct Vector_ {
    FloatT x, y, z;
    GPUEnable Vector_() { }
    template<typename T2>
    GPUEnable Vector_(const Vector_<T2>& v) : x(v.x), y(v.y), z(v.z) { }
    GPUEnable Vector_(FloatT v) { x = y = z = v; }
    GPUEnable Vector_(FloatT x_, FloatT y_, FloatT z_) : x(x_), y(y_), z(z_) { }
    GPUEnable Vector_& operator += (const Vector_& v) { x += v.x; y += v.y; z += v.z; return *this; }
    GPUEnable Vector_& operator -= (const Vector_& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    GPUEnable Vector_& operator *= (FloatT s) { x *= s; y *= s; z *= s; return *this; }
    GPUEnable Vector_& operator /= (FloatT s) { x /= s; y /= s; z /= s; return *this; }
    GPUEnable Vector_ operator + (const Vector_& v) const { return Vector_(x + v.x, y + v.y, z + v.z); }
    GPUEnable Vector_ operator - (const Vector_& v) const { return Vector_(x - v.x, y - v.y, z - v.z); }
    GPUEnable Vector_ operator * (const Vector_& v) const { return Vector_(x * v.x, y * v.y, z * v.z); }
    GPUEnable Vector_ operator / (const Vector_& v) const { return Vector_(x / v.x, y / v.y, z / v.z); }
    GPUEnable Vector_ operator - () const { return Vector_(-x, -y, -z); }
    GPUEnable Vector_ operator * (FloatT s) const { return Vector_(x * s, y * s, z * s); }
    GPUEnable Vector_ operator / (FloatT s) const { return Vector_(x / s, y / s, z / s); }
    GPUEnable FloatT dot(const Vector_& v) const { return x * v.x + y * v.y + z * v.z; }
    GPUEnable FloatT len2() const { return x * x + y * y + z * z; }
    double len2_double() const { return (double)x * (double)x + (double)y * (double)y + (double)z * (double)z; }
    GPUEnable FloatT len() const { return std::sqrt(x * x + y * y + z * z); }
    GPUEnable Vector_ normalize() const { return *this / len(); }
    GPUEnable Vector_ safe_normalize() const { return (*this / (fabs(x) + fabs(y) + fabs(z))).normalize(); }
    GPUEnable bool operator <= (const Vector_& v) const { return x <= v.x && y <= v.y && z <= v.z; }
    GPUEnable bool operator >= (const Vector_& v) const { return x >= v.x && y >= v.y && z >= v.z; }
    GPUEnable Vector_ cross(const Vector_& v) const {
        return Vector_(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    GPUEnable FloatT& operator [] (int index) { return *((FloatT*)&x + index); }
    GPUEnable const FloatT& operator [] (int index) const { return *((FloatT*)&x + index); }
};

typedef Vector_<float> Vector;
typedef Vector_<double> Vector_d;

struct Vector4 {
    float x, y, z, w;
};

struct Vector4_d {
    double x, y, z, w;
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

template<typename FloatT>
struct Color_ {
    FloatT r, g, b, a;
    GPUEnable Color_() { }
    template<typename T2>
    GPUEnable Color_(const Color_<T2>& v) : r(v.r), g(v.g), b(v.b), a(v.a) { }
    GPUEnable Color_(FloatT r_, FloatT g_, FloatT b_, FloatT a_) : r(r_), g(g_), b(b_), a(a_) { }
    GPUEnable Color_(FloatT r_, FloatT g_, FloatT b_) : r(r_), g(g_), b(b_), a(1) { }
    GPUEnable Color_ blendTo(Color_ c) {
        return Color_(
            a * r + (1.0 - a) * c.r,
            a * g + (1.0 - a) * c.g,
            a * b + (1.0 - a) * c.b,
            a * a + (1.0 - a) * c.a
        );
    }
    GPUEnable Color_ blendToCorrected(Color_ c) {
        Color_ result(
            a * r + (1.0 - a) * c.r * c.a,
            a * g + (1.0 - a) * c.g * c.a,
            a * b + (1.0 - a) * c.b * c.a,
            a + (1.0 - a) * c.a
        );
        if(result.a != 0) {
            result.r /= result.a;
            result.g /= result.a;
            result.b /= result.a;
        } else result = Color_(0, 0, 0, 0);
        return result;
    }
    GPUEnable Color_ blendToDifferential(Color_ c, FloatT ratio) {
        FloatT dt = pow(1 - a, ratio);
        return Color_(
            (1.0 - dt) * r + dt * c.r,
            (1.0 - dt) * g + dt * c.g,
            (1.0 - dt) * b + dt * c.b,
            (1.0 - dt) * a + dt * c.a
        );
    }

    GPUEnable Color_& operator += (const Color_& v) { r += v.r; g += v.g; b += v.b; a += v.a; return *this; }
    GPUEnable Color_& operator -= (const Color_& v) { r -= v.r; g -= v.g; b -= v.b; a -= v.a; return *this; }
    GPUEnable Color_& operator *= (const Color_& v) { r *= v.r; g *= v.g; b *= v.b; a *= v.a; return *this; }
    GPUEnable Color_& operator /= (const Color_& v) { r /= v.r; g /= v.g; b /= v.b; a /= v.a; return *this; }
    GPUEnable Color_& operator *= (FloatT s) { r *= s; g *= s; b *= s; a *= s; return *this; }
    GPUEnable Color_& operator /= (FloatT s) { r /= s; g /= s; b /= s; a /= s; return *this; }
    GPUEnable Color_ operator + (const Color_& v) const { return Color_(r + v.r, g + v.g, b + v.b, a + v.a); }
    GPUEnable Color_ operator - (const Color_& v) const { return Color_(r - v.r, g - v.g, b - v.b, a - v.a); }
    GPUEnable Color_ operator * (const Color_& v) const { return Color_(r * v.r, g * v.g, b * v.b, a * v.a); }
    GPUEnable Color_ operator / (const Color_& v) const { return Color_(r / v.r, g / v.g, b / v.b, a / v.a); }
    GPUEnable Color_ operator * (FloatT v) const { return Color_(r * v, g * v, b * v, a * v); }
    GPUEnable Color_ operator / (FloatT v) const { return Color_(r / v, g / v, b / v, a / v); }
};

typedef Color_<float> Color;
typedef Color_<double> Color_d;

}

#endif
