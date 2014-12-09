#include "renderer.h"
#include <float.h>
#include <stdio.h>
#include <math_functions.h>
#include <algorithm>

#include "cuda_common.h"

#include "rkv.h"

// #define RKV_SOLVER

using namespace std;

namespace allovolume {

__device__
inline float interp(float a, float b, float t) {
    return fmaf(t, b - a, a);
}

__device__ __host__
inline int clampi(int value, int min, int max) {
    if(value < min) return min;
    if(value > max) return max;
    return value;
}

__device__ __host__
inline float clampf(float value, float min, float max) {
    return fmaxf(min, fminf(max, value));
}

__device__
inline float clamp01f(float value) { return __saturatef(value); }

__device__
inline Color tf_interpolate(Color* tf, float tf_min, float tf_max, int tf_size, float t) {
    float pos = clamp01f((t - tf_min) / (tf_max - tf_min)) * tf_size - 0.5f;
    int idx = floor(pos);
    idx = clampi(idx, 0, tf_size - 2);
    float diff = pos - idx;
    Color t0 = tf[idx];
    Color t1 = tf[idx + 1];
    return t0 * (1.0 - diff) + t1 * diff;
}

struct transfer_function_t {
    Color* data;
    int size;
    float min, max;
    TransferFunction::Scale scale;

    inline __device__ Color get(float t) {
        if(scale == TransferFunction::kLogScale) {
            if(t > 0) t = log(t);
            else t = min;
        }
        return tf_interpolate(data, min, max, size, t);
    }
};

struct ray_marching_parameters_t {
    const Lens::Ray* rays;
    Color* pixels;
    transfer_function_t tf;

    const BlockDescription* blocks;
    const float* data;
    int width, height;
    int block_count;
    float blend_coefficient;

    VolumeRenderer::RaycastingMethod raycasting_method;
    Vector bbox_min, bbox_max;

    Pose pose;
};

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

__device__
inline float access_volume(const float* data, int xsize, int ysize, int zsize, int ix, int iy, int iz) {
    return data[iz * xsize * ysize + iy * xsize + ix];
}

struct block_interpolate_t {
    const float* data;
    float sx, sy, sz, tx, ty, tz;
    int cxsize, cysize, czsize;
    int ystride, zstride;

    __device__
    inline block_interpolate_t(const BlockDescription& block, const float* data_) {
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

        float t00 = interp(data[idx], data[idx + zstride], tz);
        float t01 = interp(data[idx + ystride], data[idx + ystride + zstride], tz);
        float t0 = interp(t00, t01, ty);

        float t10 = interp(data[idx + 1], data[idx + 1 + zstride], tz);
        float t11 = interp(data[idx + 1 + ystride], data[idx + 1 + ystride + zstride], tz);
        float t1 = interp(t10, t11, ty);

        return interp(t0, t1, tx);
    }
};

struct ray_marching_kernel_blockinfo_t {
    float kin, kout;
    int index;
};

struct render_dxdt_t {
    block_interpolate_t& block;
    transfer_function_t& tf;
    Vector pos, d;
    float kin, kout;
    float L;
    __device__ render_dxdt_t(block_interpolate_t& block_, transfer_function_t& tf_, Vector pos_, Vector d_, float kin_, float kout_, float L_)
    : block(block_), tf(tf_), pos(pos_), d(d_), kin(kin_), kout(kout_), L(L_) { }

    __device__ void operator() (float x, Color y, Color& dy) {
        // y'(t, y) = (y - c(t)) * ln(1 - alpha(t)) / L
        Color c = tf.get(block.interpolate(pos + d * (kout - x)));
        float s = log(1.0f - c.a) / L;
        c.a = 1.0f;
        dy = (y - c) * s;
    }
};

struct color_norm_t {
    __device__ inline float operator() (Color c) {
        return fmax(fmax(fabs(c.r), fabs(c.g)), fmax(fabs(c.b), fabs(c.a)));
    }
};

__global__
void ray_marching_kernel(ray_marching_parameters_t p) {
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
    register Color color(0, 0, 0, 0);

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
    int blockinfos_count = 0;

    for(int block_cursor = 0; block_cursor < p.block_count; block_cursor++) {
        BlockDescription block = p.blocks[block_cursor];
        float kin, kout;
        if(intersectBox(pos, d, block.min, block.max, &kin, &kout)) {
            if(kin < g_kin) kin = g_kin;
            if(kin < kout) {
                blockinfos[blockinfos_count].kin = kin;
                blockinfos[blockinfos_count].kout = kout;
                blockinfos[blockinfos_count].index = block_cursor;
                blockinfos_count += 1;
            }
        }
    }

    // Bubble-sort blocks according to distance.
    for(;;) {
        bool swapped = false;
        int n = blockinfos_count;
        for(int c = 0; c < n - 1; c++) {
            if(blockinfos[c].kin < blockinfos[c + 1].kin) {
                ray_marching_kernel_blockinfo_t tmp = blockinfos[c + 1];
                blockinfos[c + 1] = blockinfos[c];
                blockinfos[c] = tmp;
                swapped = true;
            }
        }
        n -= 1;
        if(!swapped) break;
    }

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
            if(p.raycasting_method == VolumeRenderer::kAdaptiveRKVMethod) {
                // Render this block.
                float distance = kout - kin;
                float voxel_size = (block.max.x - block.min.x) / block.xsize; // assume voxels are cubes.
                block_interpolate_t block_access(block, p.data + block.offset);
                render_dxdt_t dxdt(block_access, p.tf, pos, d, kin, kout, L);
                color_norm_t color_norm;
                Color new_color;
                RungeKuttaVerner(0.0f, distance, color, dxdt, color_norm, 1e-6f, voxel_size / 10.0f, voxel_size, new_color);
                color = new_color;
            } else {
                // Render this block.
                float distance = kout - kin;
                float voxel_size = (block.max.x - block.min.x) / block.xsize; // assume voxels are cubes.
                int steps = ceil(distance / voxel_size);
                if(steps > block.xsize * 10) steps = block.xsize * 10;
                float step_size = distance / steps;

                // Interpolate context.
                block_interpolate_t block_access(block, p.data + block.offset);

                // Blending with RK4.
                Color c0 = p.tf.get(block_access.interpolate(pos + d * kout));
                float c0s = log(1.0f - c0.a) / L;
                c0.a = 1.0f;
                for(int i = steps - 1; i >= 0; i--) {
                    Color cm = p.tf.get(block_access.interpolate(pos + d * (kin + step_size * ((float)i + 0.5f))));
                    float cms = log(1.0f - cm.a) / L;
                    cm.a = 1.0f;
                    Color c1 = p.tf.get(block_access.interpolate(pos + d * (kin + step_size * i)));
                    float c1s = log(1.0f - c1.a) / L;
                    c1.a = 1.0f;
                    // Runge Kutta Order 4 method.
                    // y'(t, y) = (y - c(t)) * ln(1 - alpha(t)) / L
                    //   y has premultiplied alpha.
                    //   c has non-premultiplied alpha.
                    Color k1 = (color - c0) * c0s;
                    Color k2 = (color + k1 * (step_size * 0.5f) - cm) * cms;
                    Color k3 = (color + k2 * (step_size * 0.5f) - cm) * cms;
                    Color k4 = (color + k3 * (step_size) - c1) * c1s;
                    color = color + (k1 + (k2 + k3) * 2.0f + k4) * (step_size / 6.0f);

                    c0 = c1;
                    c0s = c1s;
                }
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

class VolumeRendererImpl : public VolumeRenderer {
public:

    VolumeRendererImpl() :
        blocks(512),
        volume_blocks(512),
        data(512 * 32 * 32 * 32),
        volume_data(512 * 34 * 34 * 34),
        rays(1000 * 1000),
        blend_coefficient(1.0),
        raycasting_method(kRK4Method),
        bbox_min(-1e20, -1e20, -1e20),
        bbox_max(1e20, 1e20, 1e20) { }

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
    }

    virtual void setTransferFunction(TransferFunction* tf_) {
        tf = tf_;
    }

    virtual void setLens(Lens* lens_) {
        lens = lens_;
    }

    virtual void setImage(Image* image_) {
        image = image_;
    }

    virtual void setPose(const Pose& pose_) {
        pose = pose_;
    }

    virtual void setBoundingBox(Vector min, Vector max) {
        bbox_min = min;
        bbox_max = max;
    }
    virtual void setRaycastingMethod(RaycastingMethod method) {
        raycasting_method = method;
    }

    virtual void render() {
        // Sort blocks roughly.
        BlockCompare block_compare(pose.position);
        sort(blocks.cpu, blocks.cpu + block_count, block_compare);
        blocks.upload();

        // Prepare image.
        int pixel_count = image->getWidth() * image->getHeight();
        rays.allocate(pixel_count);

        // Generate rays.
        lens->getRaysGPU(image->getWidth(), image->getHeight(), rays.gpu);

        // Render kernel parameters.
        ray_marching_parameters_t pms;
        pms.rays = rays.gpu;
        pms.pixels = image->getPixelsGPU();
        pms.blocks = blocks.gpu;
        pms.data = data.gpu;
        pms.width = image->getWidth();
        pms.height = image->getHeight();
        pms.block_count = block_count;

        pms.bbox_min = bbox_min;
        pms.bbox_max = bbox_max;
        pms.raycasting_method = raycasting_method;

        // Set transfer function.
        pms.tf.data = tf->getContentGPU();
        tf->getDomain(pms.tf.min, pms.tf.max);
        pms.tf.scale = tf->getScale();
        pms.tf.size = tf->getSize();
        // For non-linear scales, process the min, max values as well.
        if(pms.tf.scale == TransferFunction::kLogScale) {
            pms.tf.min = log(pms.tf.min);
            pms.tf.max = log(pms.tf.max);
        }
        // Other parameters.
        pms.blend_coefficient = blend_coefficient;
        // Block range.
        pms.pose = pose;
        int blockdim_x = 8; // 8x8 is the optimal block size.
        int blockdim_y = 8;
        ray_marching_kernel<<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getWidth(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
        cudaThreadSynchronize();
    }

    MirroredMemory<BlockDescription> blocks;
    MirroredMemory<float> volume_data, volume_blocks, data;
    MirroredMemory<Lens::Ray> rays;
    int block_count;
    TransferFunction* tf;
    Lens* lens;
    Image* image;
    Pose pose;
    float blend_coefficient;
    RaycastingMethod raycasting_method;
    Vector bbox_min, bbox_max;
};

VolumeRenderer* VolumeRenderer::CreateGPU() {
    return new VolumeRendererImpl();
}

}
