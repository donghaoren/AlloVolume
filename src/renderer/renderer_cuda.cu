#include "allovolume/renderer.h"
#include <float.h>
#include <stdio.h>
#include <math_functions.h>
#include <algorithm>

#include "cuda_common.h"

#include "rkv.h"

#include <sys/time.h>

#include "../timeprofiler.h"

#define PREINT_MAX_P 100

using namespace std;

namespace allovolume {

// Internal data formats.
typedef float           DataType_Float32;
typedef unsigned short  DataType_UInt16;
typedef unsigned char   DataType_UInt8;

// Data format conversion from/to float.
template<typename DataType> inline __device__ DataType datatype_fromfloat(float t);
template<> inline __device__ DataType_Float32 datatype_fromfloat<DataType_Float32>(float t) { return t; }
template<> inline __device__ DataType_UInt16 datatype_fromfloat<DataType_UInt16>(float t) { return (unsigned short)(t * 65535.0f); }
template<> inline __device__ DataType_UInt8 datatype_fromfloat<DataType_UInt8>(float t) { return (unsigned char)(t * 255.0f); }

template<typename DataType> inline __device__ float datatype_rescale(float t);
template<> inline __device__ float datatype_rescale<DataType_Float32>(float t) { return t; }
template<> inline __device__ float datatype_rescale<DataType_UInt16>(float t) { return t / 65535.0f; }
template<> inline __device__ float datatype_rescale<DataType_UInt8>(float t) { return t / 255.0f; }

// Linear interpolation. out = a * (1 - t) + b * t
__device__ inline
float interp(float a, float b, float t) {
    return fmaf(t, b - a, a);
}

// Integer clamp to range [min, max]
__device__ __host__ inline
int clampi(int value, int min, int max) {
    if(value < min) return min;
    if(value > max) return max;
    return value;
}

// Float clamp to range [min, max]
__device__ __host__ inline
float clampf(float value, float min, float max) {
    return fmaxf(min, fminf(max, value));
}

// Float clamp to [0, 1].
__device__ inline
float clamp01f(float value) { return __saturatef(value); }

// Interleave 2 bits for 10-bit integers, for example:
// 01100 -> 0001001000000
//          0  1  1  0  0
__host__ __device__ inline
unsigned int interleave10bits2(unsigned short input) {
    unsigned int x = input;
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8) & 0x300f00f;
    x = (x | x << 4) & 0x30c30c3;
    x = (x | x << 2) & 0x9249249;
    return x;
}

// Constant memory seems slower than above.
// __constant__ unsigned int interleave10bits2_lookup[1024];
// __device__ inline
// unsigned int interleave10bits2_l(unsigned short input) {
//     return interleave10bits2_lookup[input];
// }

// Round the integer v up to the nearest power of 2.
__device__ __host__ inline
unsigned int next_power_of_2(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

// Morton Encode (limited to 10bit x, y, z indexes)
__device__ inline
unsigned int morton_encode_10bits(unsigned short x, unsigned short y, unsigned short z) {
    return interleave10bits2(x) | (interleave10bits2(y) << 1) | (interleave10bits2(z) << 2);
}

// Old Transfer function interpolation code.
// __device__ inline
// Color tf_interpolate(Color* tf, int tf_size, float t) {
//     float pos = clamp01f(t) * (tf_size - 1.0f);
//     int idx = floor(pos);
//     idx = clampi(idx, 0, tf_size - 2);
//     float diff = pos - idx;
//     Color t0 = tf[idx];
//     Color t1 = tf[idx + 1];
//     return t0 * (1.0 - diff) + t1 * diff;
// }
//
// struct transfer_function_t {
//     Color* data;
//     int size;

//     inline __device__ Color get(float t) {
//         return tf_interpolate(data, size, t);
//     }
// };

// KD-Tree data structure.
struct kd_tree_node_t {
    int split_axis; // 0, 1, 2 for x, y, z; -1 for leaf node, in leaf node, left = block index.
    float split_value;
    Vector bbox_min, bbox_max;
    int left, right;
};

// Ray marching kernel parameters.
struct ray_marching_parameters_t {
    const Lens::Ray* rays;
    Color* pixels;
    Color* pixels_back;
    VolumeRenderer::ClipRange* clip_ranges;

    Color bg_color;

    const BlockDescription* blocks;
    const float* data;
    int width, height;
    int block_count;
    float blend_coefficient;
    float step_size_multiplier;

    VolumeRenderer::RaycastingMethod raycasting_method;
    Vector bbox_min, bbox_max;

    const kd_tree_node_t* kd_tree;
    int kd_tree_root;

    int tf_size;

    Pose pose;
};

// Compute ray intersection with bounding box.
__device__ __host__
inline int intersectBox(Vector origin, Vector direction, Vector boxmin, Vector boxmax, float *tnear, float *tfar) {
    float tmin = FLT_MIN, tmax = FLT_MAX;
    float eps = 1e-8;
    if(fabs(direction.x) > eps) {
        float tx1 = (boxmin.x - origin.x) / direction.x;
        float tx2 = (boxmax.x - origin.x) / direction.x;
        tmin = fmaxf(tmin, fminf(tx1, tx2));
        tmax = fminf(tmax, fmaxf(tx1, tx2));
    } else {
        if(origin.x > boxmax.x || origin.x < boxmin.x) return false;
    }
    if(fabs(direction.y) > eps) {
        float ty1 = (boxmin.y - origin.y) / direction.y;
        float ty2 = (boxmax.y - origin.y) / direction.y;
        tmin = fmaxf(tmin, fminf(ty1, ty2));
        tmax = fminf(tmax, fmaxf(ty1, ty2));
    } else {
        if(origin.y > boxmax.y || origin.y < boxmin.y) return false;
    }
    if(fabs(direction.z) > eps) {
        float tz1 = (boxmin.z - origin.z) / direction.z;
        float tz2 = (boxmax.z - origin.z) / direction.z;
        tmin = fmaxf(tmin, fminf(tz1, tz2));
        tmax = fminf(tmax, fmaxf(tz1, tz2));
    } else {
        if(origin.z > boxmax.z || origin.z < boxmin.z) return false;
    }
    *tnear = tmin;
    *tfar = tmax;
    return tmax > tmin;
}

// CUDA Texture definition.
texture<float, 3, cudaReadModeElementType> volume_texture;
texture<float4, 1, cudaReadModeElementType> tf_texture;
texture<float4, 2, cudaReadModeElementType> tf_texture_preintergrated;

// Block interpolation.
template<typename DataType>
struct block_interpolate_t {
    const DataType* data;
    float sx, sy, sz, tx, ty, tz;
    int cxsize, cysize, czsize;
    int ystride, zstride;

    __device__
    inline block_interpolate_t(const BlockDescription& block, const DataType* data_) {
        data = data_;
        sx = (block.xsize - block.ghost_count * 2.0f) / (block.max.x - block.min.x);
        sy = (block.ysize - block.ghost_count * 2.0f) / (block.max.y - block.min.y);
        sz = (block.zsize - block.ghost_count * 2.0f) / (block.max.z - block.min.z);
        tx = (float)block.ghost_count - 0.5f - block.min.x * sx;
        ty = (float)block.ghost_count - 0.5f - block.min.y * sy;
        tz = (float)block.ghost_count - 0.5f - block.min.z * sz;
        cxsize = block.xsize - 2;
        cysize = block.ysize - 2;
        czsize = block.zsize - 2;
        ystride = block.xsize;
        zstride = block.xsize * block.ysize;
    }

    __device__
    inline float interpolate(Vector pos) const {
        float px = fmaf(pos.x, sx, tx);
        float py = fmaf(pos.y, sy, ty);
        float pz = fmaf(pos.z, sz, tz);

        int ix = clampi(floor(px), 0, cxsize);
        int iy = clampi(floor(py), 0, cysize);
        int iz = clampi(floor(pz), 0, czsize);

        float tx = px - ix;
        float ty = py - iy;
        float tz = pz - iz;

        int idx = ix + ystride * iy + zstride * iz;

        float t000 = data[idx];
        float t100 = data[idx + 1];
        float t001 = data[idx + zstride];
        float t101 = data[idx + 1 + zstride];
        float t010 = data[idx + ystride];
        float t110 = data[idx + 1 + ystride];
        float t011 = data[idx + ystride + zstride];
        float t111 = data[idx + 1 + ystride + zstride];

        float t00 = interp(t000, t001, tz);
        float t01 = interp(t010, t011, tz);
        float t0 = interp(t00, t01, ty);

        float t10 = interp(t100, t101, tz);
        float t11 = interp(t110, t111, tz);
        float t1 = interp(t10, t11, ty);

        return datatype_rescale<DataType>(interp(t0, t1, tx));
    }
};

// Block interpolation with Morton indexing.
template<typename DataType>
struct block_interpolate_morton_t {
    const DataType* data;
    float sx, sy, sz, tx, ty, tz;
    int cxsize, cysize, czsize;

    __device__
    inline block_interpolate_morton_t(const BlockDescription& block, const DataType* data_) {
        data = data_;
        sx = (block.xsize - block.ghost_count * 2.0f) / (block.max.x - block.min.x);
        sy = (block.ysize - block.ghost_count * 2.0f) / (block.max.y - block.min.y);
        sz = (block.zsize - block.ghost_count * 2.0f) / (block.max.z - block.min.z);
        tx = (float)block.ghost_count - 0.5f - block.min.x * sx;
        ty = (float)block.ghost_count - 0.5f - block.min.y * sy;
        tz = (float)block.ghost_count - 0.5f - block.min.z * sz;
        cxsize = block.xsize - 2;
        cysize = block.ysize - 2;
        czsize = block.zsize - 2;
    }

    __device__
    inline float interpolate(Vector pos) const {
        float px = fmaf(pos.x, sx, tx);
        float py = fmaf(pos.y, sy, ty);
        float pz = fmaf(pos.z, sz, tz);

        int ix = clampi(floor(px), 0, cxsize);
        int iy = clampi(floor(py), 0, cysize);
        int iz = clampi(floor(pz), 0, czsize);

        float tx = px - ix;
        float ty = py - iy;
        float tz = pz - iz;

        // Morton encode.
        unsigned int ix0_i = interleave10bits2(ix);
        unsigned int ix1_i = interleave10bits2(ix + 1);
        unsigned int iy0_i = interleave10bits2(iy) << 1;
        unsigned int iy1_i = interleave10bits2(iy + 1) << 1;
        unsigned int iz0_i = interleave10bits2(iz) << 2;
        unsigned int iz1_i = interleave10bits2(iz + 1) << 2;

        float t000 = data[ix0_i | iy0_i | iz0_i];
        float t001 = data[ix0_i | iy0_i | iz1_i];
        float t00 = interp(t000, t001, tz);

        float t010 = data[ix0_i | iy1_i | iz0_i];
        float t011 = data[ix0_i | iy1_i | iz1_i];
        float t01 = interp(t010, t011, tz);
        float t0 = interp(t00, t01, ty);

        float t100 = data[ix1_i | iy0_i | iz0_i];
        float t101 = data[ix1_i | iy0_i | iz1_i];
        float t10 = interp(t100, t101, tz);

        float t110 = data[ix1_i | iy1_i | iz0_i];
        float t111 = data[ix1_i | iy1_i | iz1_i];
        float t11 = interp(t110, t111, tz);

        float t1 = interp(t10, t11, ty);

        return datatype_rescale<DataType>(interp(t0, t1, tx));
    }
};

// Block interpolation with Morton indexing.
template<typename DataType>
struct block_interpolate_morton_traversal_t1 {
    const DataType* data;
    float sx, sy, sz, tx, ty, tz;
    int cxsize, cysize, czsize;

    float val_prev, val_this, step_size, kin, kout;
    int idx, steps;
    Vector pos, d;

    __device__
    inline block_interpolate_morton_traversal_t1(const BlockDescription& block, const DataType* data_, Vector pos_, Vector d_, float kin_, float kout_) {
        data = data_;
        sx = (block.xsize - block.ghost_count * 2.0f) / (block.max.x - block.min.x);
        sy = (block.ysize - block.ghost_count * 2.0f) / (block.max.y - block.min.y);
        sz = (block.zsize - block.ghost_count * 2.0f) / (block.max.z - block.min.z);
        tx = (float)block.ghost_count - 0.5f - block.min.x * sx;
        ty = (float)block.ghost_count - 0.5f - block.min.y * sy;
        tz = (float)block.ghost_count - 0.5f - block.min.z * sz;
        cxsize = block.xsize - 2;
        cysize = block.ysize - 2;
        czsize = block.zsize - 2;

        pos = pos_;
        d = d_;
        kin = kin_;
        kout = kout_;

        float distance = kout - kin;
        float voxel_size = (block.max.x - block.min.x) / block.xsize;
        steps = ceil(distance / voxel_size / 1);
        if(steps > block.xsize * 30) steps = block.xsize * 30;
        if(steps <= 2) steps = 2;
        steps *= 5;
        step_size = distance / steps;
        idx = steps;
        val_this = interpolate(pos + d * (kin + step_size * (float)steps));
    }

    __device__
    inline bool next() {
        if(idx > 0) {
            val_prev = val_this;
            idx -= 1;
            val_this = interpolate(pos + d * (kin + step_size * (float)idx));
            return true;
        } else {
            return false;
        }
    }

    __device__
    inline float interpolate(Vector pos) const {
        float px = fmaf(pos.x, sx, tx);
        float py = fmaf(pos.y, sy, ty);
        float pz = fmaf(pos.z, sz, tz);

        int ix = clampi(floor(px), 0, cxsize);
        int iy = clampi(floor(py), 0, cysize);
        int iz = clampi(floor(pz), 0, czsize);

        float tx = px - ix;
        float ty = py - iy;
        float tz = pz - iz;

        // Morton encode.
        unsigned int ix0_i = interleave10bits2(ix);
        unsigned int ix1_i = interleave10bits2(ix + 1);
        unsigned int iy0_i = interleave10bits2(iy) << 1;
        unsigned int iy1_i = interleave10bits2(iy + 1) << 1;
        unsigned int iz0_i = interleave10bits2(iz) << 2;
        unsigned int iz1_i = interleave10bits2(iz + 1) << 2;

        float t000 = data[ix0_i | iy0_i | iz0_i];
        float t001 = data[ix0_i | iy0_i | iz1_i];
        float t00 = interp(t000, t001, tz);

        float t010 = data[ix0_i | iy1_i | iz0_i];
        float t011 = data[ix0_i | iy1_i | iz1_i];
        float t01 = interp(t010, t011, tz);
        float t0 = interp(t00, t01, ty);

        float t100 = data[ix1_i | iy0_i | iz0_i];
        float t101 = data[ix1_i | iy0_i | iz1_i];
        float t10 = interp(t100, t101, tz);

        float t110 = data[ix1_i | iy1_i | iz0_i];
        float t111 = data[ix1_i | iy1_i | iz1_i];
        float t11 = interp(t110, t111, tz);

        float t1 = interp(t10, t11, ty);

        return datatype_rescale<DataType>(interp(t0, t1, tx));
    }
};

// Block interpolation with Morton indexing.
template<typename DataType>
struct block_interpolate_morton_traversal_t {
    const DataType* data;

    int xsize, ysize, zsize;

    // Stepping control.
    int X, Y, Z, step_x, step_y, step_z;
    float t_adv_x, t_adv_y, t_adv_z;
    float t_max_x, t_max_y, t_max_z;
    float tmax, t;

    // Shortcuts.
    unsigned int Xi0, Yi0, Zi0, Xi1, Yi1, Zi1;
    float v000, v001, v010, v011, v100, v101, v110, v111;

    float val_prev, val_this, step_size, kin, kout;
    int idx, steps;
    Vector pos, d;


    __device__ inline int get_x_part(int X) { return interleave10bits2(clampi(X, 0, xsize - 1)); }
    __device__ inline int get_y_part(int Y) { return interleave10bits2(clampi(Y, 0, ysize - 1)) << 1; }
    __device__ inline int get_z_part(int Z) { return interleave10bits2(clampi(Z, 0, zsize - 1)) << 2; }

    __device__
    inline block_interpolate_morton_traversal_t(const BlockDescription& block, const DataType* data_, Vector pos_, Vector d_, float kin_, float kout_) {
        data = data_;
        xsize = block.xsize;
        ysize = block.ysize;
        zsize = block.zsize;

        // Get rid of kin:
        tmax = kout_ - kin_;
        pos = pos_ + d_ * kin_;
        d = d_;
        // Now the ray is pos, pos + d * kmax.

        // Convert pos and d to integer voxel space.
        pos.x = (pos.x - block.min.x) / (block.max.x - block.min.x) * (block.xsize - block.ghost_count * 2) + block.ghost_count - 0.5f;
        pos.y = (pos.y - block.min.y) / (block.max.y - block.min.y) * (block.ysize - block.ghost_count * 2) + block.ghost_count - 0.5f;
        pos.z = (pos.z - block.min.z) / (block.max.z - block.min.z) * (block.zsize - block.ghost_count * 2) + block.ghost_count - 0.5f;
        d.x = d.x / (block.max.x - block.min.x) * (block.xsize - block.ghost_count * 2);
        d.y = d.y / (block.max.y - block.min.y) * (block.ysize - block.ghost_count * 2);
        d.z = d.z / (block.max.z - block.min.z) * (block.zsize - block.ghost_count * 2);

        X = pos.x; Y = pos.y; Z = pos.z;

        t = 0;

        Xi0 = get_x_part(X);
        Yi0 = get_y_part(Y);
        Zi0 = get_z_part(Z);
        Xi1 = get_x_part(X + 1);
        Yi1 = get_y_part(Y + 1);
        Zi1 = get_z_part(Z + 1);

        v000 = data[Xi0 | Yi0 | Zi0];
        v001 = data[Xi0 | Yi0 | Zi1];
        v010 = data[Xi0 | Yi1 | Zi0];
        v011 = data[Xi0 | Yi1 | Zi1];
        v100 = data[Xi1 | Yi0 | Zi0];
        v101 = data[Xi1 | Yi0 | Zi1];
        v110 = data[Xi1 | Yi1 | Zi0];
        v111 = data[Xi1 | Yi1 | Zi1];

        // Initialize t_max_x, t_max_y, t_max_z.
        if(d.x > 0) {
            t_max_x = (X + 1 - pos.x) / d.x; step_x = 1; t_adv_x = 1.0 / d.x;
        } else if(d.x < 0) {
            t_max_x = (X - pos.x) / d.x; t_adv_x = -1.0 / d.x; step_x = -1;
        } else {
            t_max_x = 1e20; t_adv_x = 0; step_x = 0;
        }

        if(d.y > 0) {
            t_max_y = (Y + 1 - pos.y) / d.y; step_y = 1; t_adv_y = 1.0 / d.y;
        } else if(d.y < 0) {
            t_max_y = (Y - pos.y) / d.y; t_adv_y = -1.0 / d.y; step_y = -1;
        } else {
            t_max_y = 1e20; t_adv_y = 0; step_y = 0;
        }

        if(d.z > 0) {
            t_max_z = (Z + 1 - pos.z) / d.z; step_z = 1; t_adv_z = 1.0 / d.z;
        } else if(d.z < 0) {
            t_max_z = (Z - pos.z) / d.z; t_adv_z = -1.0 / d.z; step_z = -1;
        } else {
            t_max_z = 1e20; t_adv_z = 0; step_z = 0;
        }

        // TODO.
        val_this = 0;
    }

    __device__
    inline bool next() {
        if(t >= tmax) return false;

        float t1 = fmin(tmax, fmin(t_max_z, fmin(t_max_x, t_max_y)));
        // if(t1 > t) {
        val_prev = val_this;
        Vector txyz = pos + d * t1 - Vector(X, Y, Z);
        float v00 = interp(v000, v001, txyz.z);
        float v01 = interp(v010, v011, txyz.z);
        float v10 = interp(v100, v101, txyz.z);
        float v11 = interp(v110, v111, txyz.z);
        float v0 = interp(v00, v01, txyz.y);
        float v1 = interp(v10, v11, txyz.y);
        float v = interp(v0, v1, txyz.x);
        val_this = datatype_rescale<DataType>(v);
        step_size = t1 - t;
        // }

        if(t_max_x < t_max_y) {
            if(t_max_x < t_max_z) {
                t = t_max_x;
                t_max_x = t_max_x + t_adv_x;
                if(step_x > 0) {
                    X += 1;
                    v000 = v100;
                    v001 = v101;
                    v010 = v110;
                    v011 = v111;
                    Xi0 = Xi1;
                    Xi1 = get_x_part(X + 1);
                    v100 = data[Xi1 | Yi0 | Zi0];
                    v101 = data[Xi1 | Yi0 | Zi1];
                    v110 = data[Xi1 | Yi1 | Zi0];
                    v111 = data[Xi1 | Yi1 | Zi1];
                } else {
                    X -= 1;
                    v100 = v000;
                    v101 = v001;
                    v110 = v010;
                    v111 = v011;
                    Xi1 = Xi0;
                    Xi0 = get_x_part(X);
                    v000 = data[Xi0 | Yi0 | Zi0];
                    v001 = data[Xi0 | Yi0 | Zi1];
                    v010 = data[Xi0 | Yi1 | Zi0];
                    v011 = data[Xi0 | Yi1 | Zi1];
                }
            } else {
                t = t_max_z;
                t_max_z = t_max_z + t_adv_z;
                if(step_z > 0) {
                    Z += 1;
                    v000 = v001;
                    v010 = v011;
                    v100 = v101;
                    v110 = v111;
                    Zi0 = Zi1;
                    Zi1 = get_z_part(Z + 1);
                    v001 = data[Xi0 | Yi0 | Zi1];
                    v011 = data[Xi0 | Yi1 | Zi1];
                    v101 = data[Xi1 | Yi0 | Zi1];
                    v111 = data[Xi1 | Yi1 | Zi1];
                } else {
                    Z -= 1;
                    v001 = v000;
                    v011 = v010;
                    v101 = v100;
                    v111 = v110;
                    Zi1 = Zi0;
                    Zi0 = get_z_part(Z);
                    v000 = data[Xi0 | Yi0 | Zi0];
                    v010 = data[Xi0 | Yi1 | Zi0];
                    v100 = data[Xi1 | Yi0 | Zi0];
                    v110 = data[Xi1 | Yi1 | Zi0];
                }
            }
        } else {
            if(t_max_y < t_max_z) {
                t = t_max_y;
                t_max_y = t_max_y + t_adv_y;
                if(step_y > 0) {
                    Y += 1;
                    v000 = v010;
                    v001 = v011;
                    v100 = v110;
                    v101 = v111;
                    Yi0 = Yi1;
                    Yi1 = get_y_part(Y + 1);
                    v010 = data[Xi0 | Yi1 | Zi0];
                    v011 = data[Xi0 | Yi1 | Zi1];
                    v110 = data[Xi1 | Yi1 | Zi0];
                    v111 = data[Xi1 | Yi1 | Zi1];
                } else {
                    Y -= 1;
                    v010 = v000;
                    v011 = v001;
                    v110 = v100;
                    v111 = v101;
                    Yi1 = Yi0;
                    Yi0 = get_y_part(Y);
                    v000 = data[Xi0 | Yi0 | Zi0];
                    v001 = data[Xi0 | Yi0 | Zi1];
                    v100 = data[Xi1 | Yi0 | Zi0];
                    v101 = data[Xi1 | Yi0 | Zi1];
                }
            } else {
                t = t_max_z;
                t_max_z = t_max_z + t_adv_z;
                if(step_z > 0) {
                    Z += 1;
                    v000 = v001;
                    v010 = v011;
                    v100 = v101;
                    v110 = v111;
                    Zi0 = Zi1;
                    Zi1 = get_z_part(Z + 1);
                    v001 = data[Xi0 | Yi0 | Zi1];
                    v011 = data[Xi0 | Yi1 | Zi1];
                    v101 = data[Xi1 | Yi0 | Zi1];
                    v111 = data[Xi1 | Yi1 | Zi1];
                } else {
                    Z -= 1;
                    v001 = v000;
                    v011 = v010;
                    v101 = v100;
                    v111 = v110;
                    Zi1 = Zi0;
                    Zi0 = get_z_part(Z);
                    v000 = data[Xi0 | Yi0 | Zi0];
                    v010 = data[Xi0 | Yi1 | Zi0];
                    v100 = data[Xi1 | Yi0 | Zi0];
                    v110 = data[Xi1 | Yi1 | Zi0];
                }
            }
        }

        return t < tmax;
    }
};

// Block interpolation with 3D texture (not used yet).
// struct block_interpolate_texture_t {
//     Vector scale, translate;

//     __device__
//     inline float interpolate(Vector pos) const {
//         float x = fmaf(pos.x, scale.x, translate.x);
//         float y = fmaf(pos.y, scale.y, translate.y);
//         float z = fmaf(pos.z, scale.z, translate.z);
//         return tex3D(volume_texture, x, y, z);
//     }
// };

struct ray_marching_kernel_blockinfo_t {
    float kin, kout;
    int index;
};


template<typename DataType>
__global__
void preprocess_data_kernel(float* data, DataType* data_processed, size_t data_size, TransferFunction::Scale scale, float min, float max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= data_size) return;
    float value = data[idx];

    if(scale == TransferFunction::kLogScale) {
        if(value > 0) value = log(value);
        else value = min;
    }

    value = (value - min) / (max - min);
    value = clamp01f(value);

    data_processed[idx] = datatype_fromfloat<DataType>(value);
}

template<typename DataType>
__global__
void preprocess_data_kernel_morton(float* data, DataType* data_processed, size_t data_size, int xsize, int ysize, int zsize, int block_size_morton, TransferFunction::Scale scale, float min, float max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= data_size) return;
    float value = data[idx];

    if(scale == TransferFunction::kLogScale) {
        if(value > 0) value = log(value);
        else value = min;
    }

    value = (value - min) / (max - min);
    value = clamp01f(value);

    int block_size = xsize * ysize * zsize;
    int offset = idx / block_size;

    idx = idx % block_size;
    int ix = idx % xsize;
    idx = idx / xsize;
    int iy = idx % ysize;
    int iz = idx / ysize;

    data_processed[offset * block_size_morton + morton_encode_10bits(ix, iy, iz)] = datatype_fromfloat<DataType>(value);
}

__device__
inline Color tf_tex_get(float pos) {
    float4 f4 = tex1D(tf_texture, pos);
    return Color(f4.x, f4.y, f4.z, f4.w);
}

__device__
inline Color tf_tex_get2d(float px, float py) {
    float4 f4 = tex2D(tf_texture_preintergrated, px, py);
    return Color(f4.x, f4.y, f4.z, f4.w);
}

struct traverse_stack_t {
    int node;
    float kmin, kmax;
    int stage;
};

__device__
inline int kd_tree_block_intersection(
    Vector pos, Vector direction,
    float kmin, float kmax,
    float g_kin, float g_kout,
    const kd_tree_node_t* kd_tree, int kd_tree_root,
    const BlockDescription* blocks, ray_marching_kernel_blockinfo_t* blockinfos,
    traverse_stack_t* stack
) {
    int stack_pointer = 0;
    stack[0].node = kd_tree_root;
    stack[0].kmin = kmin;
    stack[0].kmax = kmax;
    stack[0].stage = 0;
    int blockinfos_count = 0;
    while(stack_pointer >= 0) {
        traverse_stack_t& s = stack[stack_pointer];
        int axis = kd_tree[s.node].split_axis;
        if(axis < 0) {
            float kin, kout;
            if(intersectBox(pos, direction, blocks[kd_tree[s.node].left].min, blocks[kd_tree[s.node].left].max, &kin, &kout)) {
                if(kin < g_kin) kin = g_kin;
                if(kin < kout) {
                    blockinfos[blockinfos_count].kin = kin;
                    blockinfos[blockinfos_count].kout = kout;
                    blockinfos[blockinfos_count].index = kd_tree[s.node].left;
                    blockinfos_count += 1;
                }
            }
            stack_pointer -= 1;
        } else {
            float split_value = kd_tree[s.node].split_value;
            float pmina = pos[axis] + direction[axis] * s.kmin;
            float pmaxa = pos[axis] + direction[axis] * s.kmax;
            if(pmina <= split_value && pmaxa <= split_value) {
                stack[stack_pointer].node = kd_tree[s.node].left;
                stack[stack_pointer].kmin = s.kmin;
                stack[stack_pointer].kmax = s.kmax;
            } else if(pmina >= split_value && pmaxa >= split_value) {
                stack[stack_pointer].node = kd_tree[s.node].right;
                stack[stack_pointer].kmin = s.kmin;
                stack[stack_pointer].kmax = s.kmax;
            } else {
                float k_split = (split_value - pos[axis]) / direction[axis];
                if(pmina < split_value) {
                    stack_pointer += 1;
                    stack[stack_pointer].node = kd_tree[s.node].right;
                    stack[stack_pointer].kmin = k_split;
                    stack[stack_pointer].kmax = s.kmax;
                    s.node = kd_tree[s.node].left;
                    s.kmax = k_split;
                } else {
                    stack_pointer += 1;
                    stack[stack_pointer].node = kd_tree[s.node].left;
                    stack[stack_pointer].kmin = k_split;
                    stack[stack_pointer].kmax = s.kmax;
                    s.node = kd_tree[s.node].right;
                    s.kmax = k_split;
                }
            }
        }
    }
    return blockinfos_count;
}

template<typename DataType, typename BlockInterpolate>
__global__
void ray_marching_kernel_basic(ray_marching_parameters_t p) {
    // Pixel index.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= p.width || py >= p.height) return;
    register int idx = py * p.width + px;

    // Ray information.
    Lens::Ray ray = p.rays[idx];
    register Vector pos = p.pose.rotation.rotate(ray.origin) + p.pose.position;
    register Vector d = p.pose.rotation.rotate(ray.direction);

    // Initial color (background color).
    register Color color = p.bg_color;

    // Global ray information.
    float g_kin, g_kout;
    intersectBox(pos, d, p.bbox_min, p.bbox_max, &g_kin, &g_kout);
    if(g_kout < 0) {
        p.pixels[idx] = color;
        return;
    }
    if(g_kin < 0) g_kin = 0;

    // Block intersection.
    ray_marching_kernel_blockinfo_t blockinfos[128];
    traverse_stack_t stack[64];
    int blockinfos_count = kd_tree_block_intersection(pos, d, g_kin, g_kout, g_kin, g_kout, p.kd_tree, p.kd_tree_root, p.blocks, blockinfos, stack);

    // (Old O(n) Block intersection)
    // ray_marching_kernel_blockinfo_t blockinfos[512];
    // int blockinfos_count = 0;

    // for(int block_cursor = 0; block_cursor < p.block_count; block_cursor++) {
    //     BlockDescription block = p.blocks[block_cursor];
    //     float kin, kout;
    //     if(intersectBox(pos, d, block.min, block.max, &kin, &kout)) {
    //         if(kin < g_kin) kin = g_kin;
    //         if(kin < kout) {
    //             blockinfos[blockinfos_count].kin = kin;
    //             blockinfos[blockinfos_count].kout = kout;
    //             blockinfos[blockinfos_count].index = block_cursor;
    //             blockinfos_count += 1;
    //         }
    //     }
    // }

    // // Bubble-sort blocks according to distance.
    // for(;;) {
    //     bool swapped = false;
    //     int n = blockinfos_count;
    //     for(int c = 0; c < n - 1; c++) {
    //         if(blockinfos[c].kin < blockinfos[c + 1].kin) {
    //             ray_marching_kernel_blockinfo_t tmp = blockinfos[c + 1];
    //             blockinfos[c + 1] = blockinfos[c];
    //             blockinfos[c] = tmp;
    //             swapped = true;
    //         }
    //     }
    //     n -= 1;
    //     if(!swapped) break;
    // }

    // Simple solution: fixed step size.
    float kmax = g_kout;
    float L = p.blend_coefficient;

    // Render blocks.
    for(int cursor = 0; cursor < blockinfos_count; cursor++) {
        BlockDescription block = p.blocks[blockinfos[cursor].index];
        float kin = blockinfos[cursor].kin;
        float kout = blockinfos[cursor].kout;
        if(kout > kmax) kout = kmax;
        if(kin < kout) {
            // Render this block.
            float distance = kout - kin;
            float voxel_size = (block.max.x - block.min.x) / block.xsize; // assume voxels are cubes.
            int steps = ceil(distance / voxel_size / p.step_size_multiplier);
            if(steps > block.xsize * 30) steps = block.xsize * 30;
            if(steps < 2) steps = 2;
            float step_size = distance / steps;

            // Interpolate context.
            BlockInterpolate block_access(block, (DataType*)p.data + block.offset);

            // Blending with basic alpha compositing.
            for(int i = steps - 1; i >= 0; i--) {
                Color cm = tf_tex_get(block_access.interpolate(pos + d * (kin + step_size * ((float)i + 0.5f))));
                float k = expf(cm.a * step_size / L);
                color = Color(
                    cm.r * (1.0f - k) + color.r * k,
                    cm.g * (1.0f - k) + color.g * k,
                    cm.b * (1.0f - k) + color.b * k,
                    (1.0f - k) + color.a * k
                );
            }
            kmax = kin;
        }
    }

    // Un-premultiply alpha channel.
    if(color.a != 0) {
        color.r /= color.a;
        color.g /= color.a;
        color.b /= color.a;
    } else color = Color(0, 0, 0, 0);

    // Color output.
    p.pixels[idx] = color;
}

template<typename DataType, typename BlockInterpolate>
__global__
void ray_marching_kernel_preintegration(ray_marching_parameters_t p) {
    // Pixel index.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= p.width || py >= p.height) return;
    register int idx = py * p.width + px;

    // Ray information.
    Lens::Ray ray = p.rays[idx];
    register Vector pos = p.pose.rotation.rotate(ray.origin) + p.pose.position;
    register Vector d = p.pose.rotation.rotate(ray.direction);
    register float k_front = FLT_MAX;
    register float k_far = FLT_MAX;
    if(p.clip_ranges) {
        float k_near = p.clip_ranges[idx].t_near;
        k_front = p.clip_ranges[idx].t_front;
        k_far = p.clip_ranges[idx].t_far;
        pos += d * k_near;
        k_front -= k_near;
        k_far -= k_near;
    }

    // Initial color (background color).
    register Color color = p.bg_color;

    // Global ray information.
    float g_kin, g_kout;
    intersectBox(pos, d, p.bbox_min, p.bbox_max, &g_kin, &g_kout);
    if(g_kout < 0) {
        if(p.pixels) p.pixels[idx] = color;
        if(p.pixels_back) p.pixels_back[idx] = color;
        return;
    }
    if(g_kin < 0) g_kin = 0;

    // Block intersection.
    ray_marching_kernel_blockinfo_t blockinfos[128];
    traverse_stack_t stack[64];
    int blockinfos_count = kd_tree_block_intersection(pos, d, g_kin, g_kout, g_kin, g_kout, p.kd_tree, p.kd_tree_root, p.blocks, blockinfos, stack);

    // Some variables.
    float kmax = g_kout;
    float L = p.blend_coefficient;

    float tf_size = p.tf_size;

    // If rendering with front/back buffer, clamp to k_far, otherwise to k_front.
    bool is_rendering_back;
    if(p.pixels_back) {
        kmax = fminf(k_far, kmax);
        is_rendering_back = true;
    } else {
        kmax = fminf(k_front, kmax);
        is_rendering_back = false;
    }

    // Render blocks.
    for(int cursor = 0; cursor < blockinfos_count; cursor++) {
        BlockDescription block = p.blocks[blockinfos[cursor].index];
        float kin = blockinfos[cursor].kin;
        float kout = blockinfos[cursor].kout;
        if(kout > kmax) {
            kout = kmax;
        }
        bool is_back_finished = false;
        if(is_rendering_back) {
            if(kout <= k_front) {
                if(color.a != 0) {
                    color.r /= color.a;
                    color.g /= color.a;
                    color.b /= color.a;
                } else color = Color(0, 0, 0, 0);
                p.pixels_back[idx] = color;
                color = p.bg_color;
                is_rendering_back = false;
            } else if(kin <= k_front) {
                kin = k_front;
                is_back_finished = true;
            }
        }
        if(kin < kout) {
            // Render this block.
            float distance = kout - kin;
            float voxel_size = (block.max.x - block.min.x) / block.xsize; // assume voxels are cubes.
            int steps = ceil(distance / voxel_size / p.step_size_multiplier);
            if(steps > block.xsize * 30) steps = block.xsize * 30;
            if(steps <= 2) steps = 2;
            float step_size = distance / steps;

            // Blending with the pre-integrated lookup texture.
            // See "documents/allovolume-math.pdf" to see how we derived this.
            // Note that the formulas used in the code is a little bit more refined version,
            // They are essentially the same, although some terms are moved around for efficiency.
            // (We want to reduce the number of arithmetic operations in the rendering code).

            // The scaling factor of p.
            float pts_c = (step_size / L) / PREINT_MAX_P;
            // The minimum v0, v1 difference our pre-integration texture can tolerate.
            float mindiff = fmaxf(3.0 / tf_size, pts_c);

            // Interpolate context.
            BlockInterpolate block_access(block, (DataType*)p.data + block.offset);
            float val_prev = block_access.interpolate(pos + d * (kin + step_size * (float)steps));
            for(int i = steps - 1; i >= 0; i--) {
                // Access the volume.
                float val_this = block_access.interpolate(pos + d * (kin + step_size * (float)i));
                // Make sure val0 < val1 and val1 - val0 >= mindiff.
                float middle = (val_this + val_prev) / 2.0f;
                float diff = fmaxf(mindiff, fabs(val_this - val_prev)) / 2.0f;
                float val0 = middle - diff;
                float val1 = middle + diff;
                // Lookup the pre-integration table.
                float pts = pts_c / (val1 - val0);
                Color data0 = tf_tex_get2d(pts, val0);
                Color data1 = tf_tex_get2d(pts, val1);
                // Update the color.
                color.a = fmaf(data0.a, color.a, data1.a - data0.a) / data1.a;
                color.r = fmaf(data0.a, color.r, data1.r - data0.r) / data1.a;
                color.g = fmaf(data0.a, color.g, data1.g - data0.g) / data1.a;
                color.b = fmaf(data0.a, color.b, data1.b - data0.b) / data1.a;
                val_prev = val_this;
            }
            kmax = kin;
        }
        if(is_back_finished) {
            // Un-premultiply alpha channel.
            if(color.a != 0) {
                color.r /= color.a;
                color.g /= color.a;
                color.b /= color.a;
            } else color = Color(0, 0, 0, 0);
            p.pixels_back[idx] = color;
            // Reinitialize.
            is_rendering_back = false;
            color = p.bg_color;
            // Repeat the current block, since it's partially finished.
            cursor -= 1;
        }
    }

    // Un-premultiply alpha channel.
    if(color.a != 0) {
        color.r /= color.a;
        color.g /= color.a;
        color.b /= color.a;
    } else color = Color(0, 0, 0, 0);

    // Color output.
    if(is_rendering_back) {
        p.pixels_back[idx] = color;
        p.pixels[idx] = p.bg_color;
    } else {
        p.pixels[idx] = color;
    }
}

template<typename DataType, typename BlockInterpolate>
__global__
void ray_marching_kernel_preintegration_voxel_traversal(ray_marching_parameters_t p) {
    // Pixel index.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= p.width || py >= p.height) return;
    register int idx = py * p.width + px;

    // Ray information.
    Lens::Ray ray = p.rays[idx];
    register Vector pos = p.pose.rotation.rotate(ray.origin) + p.pose.position;
    register Vector d = p.pose.rotation.rotate(ray.direction);
    register float k_front = FLT_MAX;
    register float k_far = FLT_MAX;
    if(p.clip_ranges) {
        float k_near = p.clip_ranges[idx].t_near;
        k_front = p.clip_ranges[idx].t_front;
        k_far = p.clip_ranges[idx].t_far;
        pos += d * k_near;
        k_front -= k_near;
        k_far -= k_near;
    }

    // Initial color (background color).
    register Color color = p.bg_color;

    // Global ray information.
    float g_kin, g_kout;
    intersectBox(pos, d, p.bbox_min, p.bbox_max, &g_kin, &g_kout);
    if(g_kout < 0) {
        if(p.pixels) p.pixels[idx] = color;
        if(p.pixels_back) p.pixels_back[idx] = color;
        return;
    }
    if(g_kin < 0) g_kin = 0;

    // Block intersection.
    ray_marching_kernel_blockinfo_t blockinfos[128];
    traverse_stack_t stack[64];
    int blockinfos_count = kd_tree_block_intersection(pos, d, g_kin, g_kout, g_kin, g_kout, p.kd_tree, p.kd_tree_root, p.blocks, blockinfos, stack);

    // Some variables.
    float kmax = g_kout;
    float L = p.blend_coefficient;

    float tf_size = p.tf_size;

    // If rendering with front/back buffer, clamp to k_far, otherwise to k_front.
    bool is_rendering_back;
    if(p.pixels_back) {
        kmax = fminf(k_far, kmax);
        is_rendering_back = true;
    } else {
        kmax = fminf(k_front, kmax);
        is_rendering_back = false;
    }

    // Render blocks.
    for(int cursor = 0; cursor < blockinfos_count; cursor++) {
        BlockDescription block = p.blocks[blockinfos[cursor].index];
        float kin = blockinfos[cursor].kin;
        float kout = blockinfos[cursor].kout;
        if(kout > kmax) {
            kout = kmax;
        }
        bool is_back_finished = false;
        if(is_rendering_back) {
            if(kout <= k_front) {
                if(color.a != 0) {
                    color.r /= color.a;
                    color.g /= color.a;
                    color.b /= color.a;
                } else color = Color(0, 0, 0, 0);
                p.pixels_back[idx] = color;
                color = p.bg_color;
                is_rendering_back = false;
            } else if(kin <= k_front) {
                kin = k_front;
                is_back_finished = true;
            }
        }
        if(kin < kout) {
            // Render this block.

            // Blending with the pre-integrated lookup texture.
            // See "documents/allovolume-math.pdf" to see how we derived this.
            // Note that the formulas used in the code is a little bit more refined version,
            // They are essentially the same, although some terms are moved around for efficiency.
            // (We want to reduce the number of arithmetic operations in the rendering code).

            // Interpolate context.
            BlockInterpolate block_access(block, (DataType*)p.data + block.offset, pos, d, kin, kout);
            while(block_access.next()) {
                float val_prev = block_access.val_prev;
                float val_this = block_access.val_this;
                float step_size = block_access.step_size;

                // The scaling factor of p.
                float pts_c = (step_size / L) / PREINT_MAX_P;
                // The minimum v0, v1 difference our pre-integration texture can tolerate.
                float mindiff = fmaxf(3.0 / tf_size, pts_c);
                // Make sure val0 < val1 and val1 - val0 >= mindiff.
                float middle = (val_this + val_prev) / 2.0f;
                float diff = fmaxf(mindiff, fabs(val_this - val_prev)) / 2.0f;
                float val0 = middle - diff;
                float val1 = middle + diff;
                // Lookup the pre-integration table.
                float pts = pts_c / (val1 - val0);
                Color data0 = tf_tex_get2d(pts, val0);
                Color data1 = tf_tex_get2d(pts, val1);
                // Update the color.
                color.a = fmaf(data0.a, color.a, data1.a - data0.a) / data1.a;
                color.r = fmaf(data0.a, color.r, data1.r - data0.r) / data1.a;
                color.g = fmaf(data0.a, color.g, data1.g - data0.g) / data1.a;
                color.b = fmaf(data0.a, color.b, data1.b - data0.b) / data1.a;
                val_prev = val_this;
            }

            kmax = kin;
        }
        if(is_back_finished) {
            // Un-premultiply alpha channel.
            if(color.a != 0) {
                color.r /= color.a;
                color.g /= color.a;
                color.b /= color.a;
            } else color = Color(0, 0, 0, 0);
            p.pixels_back[idx] = color;
            // Reinitialize.
            is_rendering_back = false;
            color = p.bg_color;
            // Repeat the current block, since it's partially finished.
            cursor -= 1;
        }
    }

    // Un-premultiply alpha channel.
    if(color.a != 0) {
        color.r /= color.a;
        color.g /= color.a;
        color.b /= color.a;
    } else color = Color(0, 0, 0, 0);

    // Color output.
    if(is_rendering_back) {
        p.pixels_back[idx] = color;
        p.pixels[idx] = p.bg_color;
    } else {
        p.pixels[idx] = color;
    }
}

template<typename DataType, typename BlockInterpolate>
struct render_dxdt_t {
    BlockInterpolate& block;
    Vector pos, d;
    double kin, kout;
    double L;
    __device__ render_dxdt_t(BlockInterpolate& block_, Vector pos_, Vector d_, double kin_, double kout_, double L_)
    : block(block_), pos(pos_), d(d_), kin(kin_), kout(kout_), L(L_) { }

    __device__ void operator() (double x, Color y, Color& dy) {
        // y'(t, y) = (y - c(t)) * ln(1 - alpha(t)) / L
        Color c = tf_tex_get(block.interpolate(pos + d * (kout - x)));
        double s = c.a / L;
        c.a = 1.0f;
        dy = (y - c) * s;
    }
};

struct color_norm_t {
    __device__ inline double operator() (Color c) {
        return fmax(fmax(fabs(c.r), fabs(c.g)), fmax(fabs(c.b), fabs(c.a)));
    }
};

template<typename DataType, typename BlockInterpolate>
__global__
void ray_marching_kernel_rkf_double(ray_marching_parameters_t p) {
    // Pixel index.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= p.width || py >= p.height) return;
    register int idx = py * p.width + px;

    // Ray information.
    Lens::Ray ray = p.rays[idx];
    register Vector pos = p.pose.rotation.rotate(ray.origin) + p.pose.position;
    register Vector d = p.pose.rotation.rotate(ray.direction);

    // Initial color (background color).
    register Color color = p.bg_color;

    // Global ray information.
    float g_kin, g_kout;
    intersectBox(pos, d, p.bbox_min, p.bbox_max, &g_kin, &g_kout);
    if(g_kout < 0) {
        p.pixels[idx] = color;
        return;
    }
    if(g_kin < 0) g_kin = 0;

    // Block intersection.
    ray_marching_kernel_blockinfo_t blockinfos[128];
    traverse_stack_t stack[64];
    int blockinfos_count = kd_tree_block_intersection(pos, d, g_kin, g_kout, g_kin, g_kout, p.kd_tree, p.kd_tree_root, p.blocks, blockinfos, stack);

    // Simple solution: fixed step size.
    float kmax = g_kout;
    float L = p.blend_coefficient;

    // Render blocks.
    for(int cursor = 0; cursor < blockinfos_count; cursor++) {
        BlockDescription block = p.blocks[blockinfos[cursor].index];
        float kin = blockinfos[cursor].kin;
        float kout = blockinfos[cursor].kout;
        if(kout > kmax) kout = kmax;
        if(kin < kout) {
            // Render this block.
            float distance = kout - kin;
            float voxel_size = (block.max.x - block.min.x) / block.xsize; // assume voxels are cubes.
            BlockInterpolate block_access(block, (DataType*)p.data + block.offset);
            render_dxdt_t<DataType, BlockInterpolate> dxdt(block_access, pos, d, kin, kout, L);
            color_norm_t color_norm;
            Color new_color;
            RungeKuttaFehlberg(0.0f, distance, color, dxdt, color_norm, 1e-6f, voxel_size / 64.0f, voxel_size / 2.0f, new_color);
            color = new_color;
            kmax = kin;
        }
    }

    // Un-premultiply alpha channel.
    if(color.a != 0) {
        color.r /= color.a;
        color.g /= color.a;
        color.b /= color.a;
    } else color = Color(0, 0, 0, 0);

    // Color output.
    p.pixels[idx] = color;
}

__global__
void tf_preint_kernel(Color* table, Color* tf, float* Y, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size) return;
    float rsum = 0, gsum = 0, bsum = 0;
    float p = (i + 0.5) / size * PREINT_MAX_P;
    for(int j = 0; j < size; j++) {
        int idx = j * size + i;
        float Y_j = Y[j];
        Color TF_j = tf[j];
        float scaler = -p * exp(-p * Y_j) * TF_j.a / size / 2;
        float dr = scaler * TF_j.r;
        float dg = scaler * TF_j.g;
        float db = scaler * TF_j.b;
        rsum += dr;
        gsum += dg;
        bsum += db;
        table[idx].r = rsum;
        table[idx].g = gsum;
        table[idx].b = bsum;
        table[idx].a = exp(-Y_j * p);
        rsum += dr;
        gsum += dg;
        bsum += db;
        __syncthreads();
    }
}

class VolumeRendererImpl : public VolumeRenderer {
public:

    VolumeRendererImpl() :
        blend_coefficient(1.0),
        step_size_multiplier(1.0),
        raycasting_method(kBasicBlendingMethod),
        internal_format(kFloat32),
        enable_morton_ordering(false),
        bbox_min(-1e20, -1e20, -1e20),
        bbox_max(1e20, 1e20, 1e20),
        bg_color(0, 0, 0, 0)
    {
        tf_texture_data = NULL;
        tf_texture_data_size = 0;
        tf_preint_preprocessed = false;
        floatChannelDesc = cudaCreateChannelDesc<float>();
        image = NULL;
        image_back = NULL;
        clip_ranges_cpu = NULL;
    }

    struct BlockCompare {
        BlockCompare(Vector center_) {
            center = center_;
        }

        bool operator () (const BlockDescription& a, const BlockDescription& b) {
            double d1 = ((a.min + a.max) / 2.0f - center).len2_double();
            double d2 = ((b.min + b.max) / 2.0f - center).len2_double();
            return d1 > d2;
        }

        Vector center;
    };

    virtual void setBlendingCoefficient(float value) {
        blend_coefficient = value;
    }

    virtual void setVolume(VolumeBlocks* volume) {
        // Copy volume data.
        block_count = volume->getBlockCount();
        data.allocate(volume->getDataSize());
        data.upload(volume->getData());
        blocks.allocate(block_count);
        for(int i = 0; i < block_count; i++) {
            blocks[i] = *volume->getBlockDescription(i);
        }
        blocks.upload();
        buildKDTree();
        need_preprocess_volume = true;
    }

    virtual void setInternalFormat(InternalFormat format) {
        if(internal_format != format) {
            internal_format = format;
            need_preprocess_volume = true;
        }
    }

    virtual InternalFormat getInternalFormat() {
        return internal_format;
    }

    virtual void setEnableZIndex(bool enabled) {
        if(enable_morton_ordering != enabled) {
            enable_morton_ordering = enabled;
            need_preprocess_volume = true;
        }
    }
    virtual bool getEnableZIndex() {
        return enable_morton_ordering;
    }

    void preprocessVolume() {
        if(!need_preprocess_volume) return;

        int block_size = 128;

        float tf_min, tf_max;
        tf->getDomain(tf_min, tf_max);
        TransferFunction::Scale tf_scale = tf->getScale();
        // For non-linear scales, process the min, max values as well.
        if(tf_scale == TransferFunction::kLogScale) {
            tf_min = log(tf_min);
            tf_max = log(tf_max);
        }
        if(enable_morton_ordering) {
            blocks_morton.allocate(block_count);
            int block_size_morton = 0;
            int xsize_original = blocks[0].xsize;
            int ysize_original = blocks[0].ysize;
            int zsize_original = blocks[0].zsize;
            for(int i = 0; i < block_count; i++) {
                // Preprocess the volume.
                int xsize = blocks[i].xsize;
                int ysize = blocks[i].ysize;
                int zsize = blocks[i].zsize;
                int power_of_2_size = next_power_of_2(max(max(xsize, ysize), zsize));
                block_size_morton = max(block_size_morton, power_of_2_size * power_of_2_size * power_of_2_size);
                blocks_morton[i] = blocks[i];
            }
            for(int i = 0; i < block_count; i++) {
                blocks_morton[i].offset = i * block_size_morton;
            }

            blocks_morton.upload();

            data_processed.allocate(block_size_morton * block_count);

            switch(internal_format) {
                case kFloat32: {
                    data_processed.allocate_type<DataType_Float32>(block_size_morton * block_count);
                    preprocess_data_kernel_morton<DataType_Float32><<<diviur(data.size, block_size), block_size>>>(data.gpu, (DataType_Float32*)data_processed.gpu, data.size, xsize_original, ysize_original, zsize_original, block_size_morton, tf->getScale(), tf_min, tf_max);
                } break;
                case kUInt16: {
                    data_processed.allocate_type<DataType_UInt16>(block_size_morton * block_count);
                    preprocess_data_kernel_morton<DataType_UInt16><<<diviur(data.size, block_size), block_size>>>(data.gpu, (DataType_UInt16*)data_processed.gpu, data.size, xsize_original, ysize_original, zsize_original, block_size_morton, tf->getScale(), tf_min, tf_max);
                } break;
                case kUInt8: {
                    data_processed.allocate_type<DataType_UInt8>(block_size_morton * block_count);
                    preprocess_data_kernel_morton<DataType_UInt8><<<diviur(data.size, block_size), block_size>>>(data.gpu, (DataType_UInt8*)data_processed.gpu, data.size, xsize_original, ysize_original, zsize_original, block_size_morton, tf->getScale(), tf_min, tf_max);
                } break;
            }
        } else {
            switch(internal_format) {
                case kFloat32: {
                    data_processed.allocate_type<DataType_Float32>(data.size);
                    preprocess_data_kernel<DataType_Float32><<<diviur(data.size, block_size), block_size>>>(data.gpu, (DataType_Float32*)data_processed.gpu, data.size, tf->getScale(), tf_min, tf_max);
                } break;
                case kUInt16: {
                    data_processed.allocate_type<DataType_UInt16>(data.size);
                    preprocess_data_kernel<DataType_UInt16><<<diviur(data.size, block_size), block_size>>>(data.gpu, (DataType_UInt16*)data_processed.gpu, data.size, tf->getScale(), tf_min, tf_max);
                } break;
                case kUInt8: {
                    data_processed.allocate_type<DataType_UInt8>(data.size);
                    preprocess_data_kernel<DataType_UInt8><<<diviur(data.size, block_size), block_size>>>(data.gpu, (DataType_UInt8*)data_processed.gpu, data.size, tf->getScale(), tf_min, tf_max);
                } break;
            }
        }

        need_preprocess_volume = false;
    }

    virtual void setTransferFunction(TransferFunction* tf_) {
        tf = tf_;
        tf_preint_preprocessed = false;
        need_preprocess_volume = true;
    }

    virtual void setLens(Lens* lens_) {
        lens = lens_;
    }

    virtual void setImage(Image* image_) {
        image = image_;
    }

    virtual void setBackImage(Image* image_) {
        image_back = image_;
    }

    virtual void setPose(const Pose& pose_) {
        pose = pose_;
    }

    virtual void setStepSizeMultiplier(float value) {
        step_size_multiplier = value;
    }
    virtual float getStepSizeMultiplier() {
        return step_size_multiplier;
    }

    virtual void setBoundingBox(Vector min, Vector max) {
        bbox_min = min;
        bbox_max = max;
    }
    virtual void setRaycastingMethod(RaycastingMethod method) {
        raycasting_method = method;
    }

    virtual float getBlendingCoefficient() {
        return blend_coefficient;
    }
    virtual Pose getPose() {
        return pose;
    }
    virtual void getBoundingBox(Vector& min, Vector& max) {
        min = bbox_min;
        max = bbox_max;
    }
    virtual RaycastingMethod getRaycastingMethod() {
        return raycasting_method;
    }
    virtual void setBackgroundColor(Color color) {
        bg_color = color;
    }
    virtual Color getBackgroundColor() {
        return bg_color;
    }

    virtual void setClipRanges(ClipRange* ranges, size_t size) {
        clip_ranges_cpu = ranges;
        if(clip_ranges_cpu) {
            clip_ranges.allocate(size);
            clip_ranges.upload(clip_ranges_cpu);
        }
    }

    virtual void render() {
        render(0, 0, image->getWidth(), image->getHeight());
    }

    virtual void render(int x0, int y0, int total_width, int total_height) {
        // Prepare image.
        int pixel_count = image->getWidth() * image->getHeight();
        rays.allocate(pixel_count);

        // Generate rays.
        Lens::Viewport vp;
        vp.width = total_width;
        vp.height = total_height;
        vp.vp_x = x0; vp.vp_y = y0;
        vp.vp_width = image->getWidth(); vp.vp_height = image->getHeight();
        lens->getRaysGPU(vp, rays.gpu);

        ClipRange* clip_ranges_gpu = NULL;
        if(clip_ranges_cpu) {
            clip_ranges_gpu = clip_ranges.gpu;
        }

        // Proprocess the scale of the transfer function.
        preprocessVolume();

        // Upload the transfer function.
        if(raycasting_method == kPreIntegrationMethod) {
            if(!tf_preint_preprocessed) {
                uploadTransferFunctionPreintegratedGPU();
                tf_preint_preprocessed = true;
            }
        } else {
            uploadTransferFunctionTexture();
        }

        // Render kernel parameters.
        ray_marching_parameters_t pms;

        pms.rays = rays.gpu;
        pms.pixels = image ? image->getPixelsGPU() : NULL;
        pms.pixels_back = image_back ? image_back->getPixelsGPU() : NULL;
        pms.clip_ranges = clip_ranges_gpu;
        if(enable_morton_ordering) {
            pms.blocks = blocks_morton.gpu;
        } else {
            pms.blocks = blocks.gpu;
        }
        pms.kd_tree = kd_tree.gpu;
        pms.kd_tree_root = kd_tree_root;
        pms.data = data_processed.gpu;
        pms.width = image->getWidth();
        pms.height = image->getHeight();
        pms.block_count = block_count;

        pms.bbox_min = bbox_min;
        pms.bbox_max = bbox_max;
        pms.raycasting_method = raycasting_method;

        // Other parameters.
        pms.blend_coefficient = blend_coefficient;
        pms.step_size_multiplier = step_size_multiplier;
        pms.bg_color = bg_color;
        // Block range.
        pms.pose = pose;

        pms.tf_size = tf->getSize();

        int blockdim_x = 8; // 8x8 is the optimal block size.
        int blockdim_y = 8;
        if(enable_morton_ordering) {
            if(raycasting_method == kBasicBlendingMethod) {
                bindTransferFunctionTexture();
                switch(internal_format) {
                    case kFloat32: {
                        ray_marching_kernel_basic<DataType_Float32, block_interpolate_morton_t<DataType_Float32> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt16: {
                        ray_marching_kernel_basic<DataType_UInt16, block_interpolate_morton_t<DataType_UInt16> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt8: {
                        ray_marching_kernel_basic<DataType_UInt8, block_interpolate_morton_t<DataType_UInt8> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                }
                unbindTransferFunctionTexture();
            }
            if(raycasting_method == kPreIntegrationMethod) {
                bindTransferFunctionTexture2D();
                switch(internal_format) {
                    case kFloat32: {
                        ray_marching_kernel_preintegration<DataType_Float32, block_interpolate_morton_t<DataType_Float32> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt16: {
                        ray_marching_kernel_preintegration<DataType_UInt16, block_interpolate_morton_t<DataType_UInt16> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt8: {
                        ray_marching_kernel_preintegration<DataType_UInt8, block_interpolate_morton_t<DataType_UInt8> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                }
                unbindTransferFunctionTexture2D();
            }
            if(raycasting_method == kAdaptiveRKFMethod) {
                bindTransferFunctionTexture();
                switch(internal_format) {
                    case kFloat32: {
                        ray_marching_kernel_rkf_double<DataType_Float32, block_interpolate_morton_t<DataType_Float32> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt16: {
                        ray_marching_kernel_rkf_double<DataType_UInt16, block_interpolate_morton_t<DataType_UInt16> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt8: {
                        ray_marching_kernel_rkf_double<DataType_UInt8, block_interpolate_morton_t<DataType_UInt8> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                }
                unbindTransferFunctionTexture();
            }
        } else {
            if(raycasting_method == kBasicBlendingMethod) {
                bindTransferFunctionTexture();
                switch(internal_format) {
                    case kFloat32: {
                        ray_marching_kernel_basic<DataType_Float32, block_interpolate_t<DataType_Float32> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt16: {
                        ray_marching_kernel_basic<DataType_UInt16, block_interpolate_t<DataType_UInt16> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt8: {
                        ray_marching_kernel_basic<DataType_UInt8, block_interpolate_t<DataType_UInt8> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                }
                unbindTransferFunctionTexture();
            }
            if(raycasting_method == kPreIntegrationMethod) {
                bindTransferFunctionTexture2D();
                switch(internal_format) {
                    case kFloat32: {
                        ray_marching_kernel_preintegration<DataType_Float32, block_interpolate_t<DataType_Float32> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt16: {
                        ray_marching_kernel_preintegration<DataType_UInt16, block_interpolate_t<DataType_UInt16> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt8: {
                        ray_marching_kernel_preintegration<DataType_UInt8, block_interpolate_t<DataType_UInt8> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                }
                unbindTransferFunctionTexture2D();
            }
            if(raycasting_method == kAdaptiveRKFMethod) {
                bindTransferFunctionTexture();
                switch(internal_format) {
                    case kFloat32: {
                        ray_marching_kernel_rkf_double<DataType_Float32, block_interpolate_t<DataType_Float32> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt16: {
                        ray_marching_kernel_rkf_double<DataType_UInt16, block_interpolate_t<DataType_UInt16> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                    case kUInt8: {
                        ray_marching_kernel_rkf_double<DataType_UInt8, block_interpolate_t<DataType_UInt8> ><<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
                    } break;
                }
                unbindTransferFunctionTexture();
            }
        }
    }

    void uploadTransferFunctionTexture() {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

        if(tf_texture_data_size != tf->getSize()) {
            if(tf_texture_data) {
                cudaFreeArray(tf_texture_data);
            }
            cudaMallocArray(&tf_texture_data, &channel_desc, tf->getSize());
            tf_texture_data_size = tf->getSize();
        }

        Color* tf_color_logalpha = new Color[tf->getSize()];
        Color* tf_color = tf->getContent();
        for(int i = 0; i < tf->getSize(); i++) {
            tf_color_logalpha[i] = tf_color[i];
            tf_color_logalpha[i].a = log(1.0f - tf_color_logalpha[i].a);
        }

        cudaMemcpyToArray(tf_texture_data, 0, 0,
            tf_color_logalpha,
            sizeof(float4) * tf->getSize(),
            cudaMemcpyHostToDevice);

        delete [] tf_color_logalpha;

        tf_texture.normalized = 1;
        tf_texture.filterMode = cudaFilterModeLinear;
        tf_texture.addressMode[0] = cudaAddressModeClamp;
        tf_texture.addressMode[1] = cudaAddressModeClamp;
    }

    void uploadTransferFunctionPreintegratedGPU() {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        int size = tf->getSize();
        int size2d = size * size;

        if(tf_texture_data_size != size2d) {
            if(tf_texture_data) {
                cudaFreeArray(tf_texture_data);
            }
            tf_texture_data = NULL;
            cudaError_t err = cudaMallocArray(&tf_texture_data, &channel_desc, size, size);
            if(!tf_texture_data) {
                int size_bytes = size * size * sizeof(Color);
                fprintf(stderr, "cudaAllocate: cudaMalloc() of %d (%.2f MB): %s\n",
                    size_bytes, size_bytes / 1048576.0,
                    cudaGetErrorString(err));
                size_t memory_free, memory_total;
                cudaMemGetInfo(&memory_free, &memory_total);
                fprintf(stderr, "  Free: %.2f MB, Total: %.2f MB\n", (float)memory_free / 1048576.0, (float)memory_total / 1048576.0);
                throw bad_alloc();
            }
            tf_texture_data_size = size2d;
        }

        Color* tf_color = tf->getContent();

        preint_yt.allocate(size);
        preint_tf.allocate(size);
        preint_table.allocate(size2d);

        float csum = 0;
        for(int i = 0; i < size; i++) {
            float v = log(1.0f - tf_color[i].a);
            preint_tf[i] = tf_color[i];
            preint_tf[i].a = v;
            v /= size;
            csum += v / 2.0;
            preint_yt[i] = csum;
            csum += v / 2.0;
        }

        preint_yt.upload();
        preint_tf.upload();

        tf_preint_kernel<<<diviur(size, 64), 64>>>(preint_table.gpu, preint_tf.gpu, preint_yt.gpu, size);

        cudaMemcpy2DToArray(tf_texture_data, 0, 0,
            preint_table.gpu,
            sizeof(float4) * size,
            sizeof(float4) * size, size,
            cudaMemcpyDeviceToDevice);

        tf_texture_preintergrated.normalized = 1;
        tf_texture_preintergrated.filterMode = cudaFilterModeLinear;
        tf_texture_preintergrated.addressMode[0] = cudaAddressModeClamp;
        tf_texture_preintergrated.addressMode[1] = cudaAddressModeClamp;
    }

    void bindTransferFunctionTexture2D() {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        cudaBindTextureToArray(tf_texture_preintergrated, tf_texture_data, channel_desc);
    }
    void unbindTransferFunctionTexture2D() {
        cudaUnbindTexture(tf_texture_preintergrated);
    }
    void bindTransferFunctionTexture() {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        cudaBindTextureToArray(tf_texture, tf_texture_data, channel_desc);
    }
    void unbindTransferFunctionTexture() {
        cudaUnbindTexture(tf_texture);
    }
    cudaArray* tf_texture_data;
    size_t tf_texture_data_size;
    int tf_texture_preintergrated_steps;

    MirroredMemory<kd_tree_node_t> kd_tree;
    int kd_tree_root;
    int kd_tree_size;

    int buildKDTreeRecursive(kd_tree_node_t* nodes, int& nodes_count, int* blockids, int block_count, int axis) {
        if(block_count == 1) {
            kd_tree_node_t node;
            node.left = blockids[0];
            node.right = -1;
            node.split_value = 0;
            node.split_axis = -1;
            nodes[nodes_count++] = node;
            return nodes_count - 1;
        }
        float sp_min = FLT_MAX, sp_max = -FLT_MAX;
        for(int i = 0; i < block_count; i++) {
            sp_min = fminf(sp_min, blocks[blockids[i]].min[axis]);
            sp_max = fmaxf(sp_max, blocks[blockids[i]].max[axis]);
        }
        float split_value = (sp_min + sp_max) / 2.0f;

        int* blocks_left = new int[block_count];
        int blocks_left_count = 0;

        int* blocks_right = new int[block_count];
        int blocks_right_count = 0;

        for(int i = 0; i < block_count; i++) {
            if(blocks[blockids[i]].min[axis] + blocks[blockids[i]].max[axis] < 2.0f * split_value) {
                blocks_left[blocks_left_count++] = blockids[i];
            } else {
                blocks_right[blocks_right_count++] = blockids[i];
            }
        }

        // if(blocks_left_count == 0 || blocks_right_count == 0) {
        //     printf("Something wrong here.\n");
        //     printf("%f %f %f\n", sp_min, sp_max, split_value);
        //     for(int i = 0; i < block_count; i++) {
        //         printf("%f %f\n", blocks[blockids[i]].min[axis], blocks[blockids[i]].max[axis]);
        //     }
        //     exit(-1);
        // }

        int next_axis = (axis + 1) % 3;
        int left = buildKDTreeRecursive(nodes, nodes_count, blocks_left, blocks_left_count, next_axis);
        int right = buildKDTreeRecursive(nodes, nodes_count, blocks_right, blocks_right_count, next_axis);
        delete [] blocks_left;
        delete [] blocks_right;

        kd_tree_node_t node;
        node.left = left;
        node.right = right;
        node.split_value = split_value;
        node.split_axis = axis;
        nodes[nodes_count++] = node;
        return nodes_count - 1;
    }
    void buildKDTree() {
        kd_tree.allocate(blocks.size * 5);
        kd_tree_size = 0;

        int* blockids = new int[blocks.size];

        for(int i = 0; i < blocks.size; i++) blockids[i] = i;
        kd_tree_root = buildKDTreeRecursive(kd_tree.cpu, kd_tree_size, blockids, blocks.size, 0);
        kd_tree.size = kd_tree_size;
        kd_tree.upload();

        delete [] blockids;
    }

    // Memory regions:
    MirroredMemory<BlockDescription> blocks, blocks_morton;
    MirroredMemory<float> data, data_processed;
    MirroredMemory<Lens::Ray> rays;
    MirroredMemory<ClipRange> clip_ranges;
    ClipRange* clip_ranges_cpu;

    int block_count;
    TransferFunction* tf;
    Lens* lens;
    Image* image;
    Image* image_back;

    // Rendering parameters:
    Color bg_color;
    float blend_coefficient;
    float step_size_multiplier;
    InternalFormat internal_format;
    bool enable_morton_ordering;
    RaycastingMethod raycasting_method;

    // Global bounding box:
    Vector bbox_min, bbox_max;

    // Pose:
    Pose pose;

    cudaChannelFormatDesc floatChannelDesc;

    MirroredMemory<float> preint_yt;
    MirroredMemory<Color> preint_tf, preint_table;

    bool tf_preint_preprocessed;
    bool need_preprocess_volume;
};

VolumeRenderer* VolumeRenderer::CreateGPU() {
    return new VolumeRendererImpl();
}

}
