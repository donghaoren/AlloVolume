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
inline Color tf_interpolate(Color* tf, int tf_size, float t) {
    float pos = clamp01f(t) * (tf_size - 1.0f);
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

    inline __device__ Color get(float t) {
        return tf_interpolate(data, size, t);
    }
};

struct kd_tree_node_t {
    int split_axis; // 0, 1, 2 for x, y, z; -1 for leaf node, in leaf node, left = block index.
    float split_value;
    Vector bbox_min, bbox_max;
    int left, right;
};

struct ray_marching_parameters_t {
    const Lens::Ray* rays;
    Color* pixels;

    Color bg_color;

    const BlockDescription* blocks;
    const float* data;
    int width, height;
    int block_count;
    float blend_coefficient;

    VolumeRenderer::RaycastingMethod raycasting_method;
    Vector bbox_min, bbox_max;

    const kd_tree_node_t* kd_tree;
    int kd_tree_root;

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

texture<float, 3, cudaReadModeElementType> volume_texture;
texture<float4, 1, cudaReadModeElementType> tf_texture;

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

struct block_interpolate_texture_t {
    Vector scale, translate;

    __device__
    inline float interpolate(Vector pos) const {
        float x = fmaf(pos.x, scale.x, translate.x);
        float y = fmaf(pos.y, scale.y, translate.y);
        float z = fmaf(pos.z, scale.z, translate.z);
        return tex3D(volume_texture, x, y, z);
    }
};

struct ray_marching_kernel_blockinfo_t {
    float kin, kout;
    int index;
};


__global__
void preprocess_data_kernel(float* data, float* data_processed, size_t data_size, TransferFunction::Scale scale, float min, float max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= data_size) return;
    float value = data[idx];

    if(scale == TransferFunction::kLogScale) {
        if(value > 0) value = log(value);
        else value = min;
    }

    value = (value - min) / (max - min);

    data_processed[idx] = value;
}

__device__
inline Color tf_tex_get(float pos) {
    float4 f4 = tex1D(tf_texture, pos);
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

__global__
void ray_marching_kernel_basic(ray_marching_parameters_t p) {
    // Pixel index.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= p.width || py >= p.height) return;
    register int idx = py * p.width + px;

    // __shared__ BlockDescription shared_blocks[512];
    // // Copy all blocks into shared_blocks.
    // for(int i = threadIdx.y * blockDim.x + threadIdx.x; i < p.block_count; i += blockDim.x * blockDim.y) {
    //     shared_blocks[i] = p.blocks[i];
    // }
    // __syncthreads();
    // p.blocks = shared_blocks;

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
            int steps = ceil(distance / voxel_size);
            if(steps > block.xsize * 10) steps = block.xsize * 10;
            float step_size = distance / steps;

            // Interpolate context.
            block_interpolate_t block_access(block, p.data + block.offset);

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

__global__
void ray_marching_kernel_rk4(ray_marching_parameters_t p) {
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
    ray_marching_kernel_blockinfo_t blockinfos[512];
    traverse_stack_t stack[24];
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
            int steps = ceil(distance / voxel_size);
            if(steps > block.xsize * 10) steps = block.xsize * 10;
            float step_size = distance / steps;

            // Interpolate context.
            block_interpolate_t block_access(block, p.data + block.offset);

            // Blending with RK4.
            Color c0 = tf_tex_get(block_access.interpolate(pos + d * kout));
            float c0s = c0.a / L;
            c0.a = 1.0f;
            for(int i = steps - 1; i >= 0; i--) {
                Color cm = tf_tex_get(block_access.interpolate(pos + d * (kin + step_size * ((float)i + 0.5f))));
                float cms = cm.a / L;
                cm.a = 1.0f;
                Color c1 = tf_tex_get(block_access.interpolate(pos + d * (kin + step_size * i)));
                float c1s = c1.a / L;
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

struct render_dxdt_t {
    block_interpolate_t& block;
    Vector_d pos, d;
    double kin, kout;
    double L;
    __device__ render_dxdt_t(block_interpolate_t& block_, Vector_d pos_, Vector_d d_, double kin_, double kout_, double L_)
    : block(block_), pos(pos_), d(d_), kin(kin_), kout(kout_), L(L_) { }

    __device__ void operator() (double x, Color_d y, Color_d& dy) {
        // y'(t, y) = (y - c(t)) * ln(1 - alpha(t)) / L
        Color_d c = tf_tex_get(block.interpolate(pos + d * (kout - x)));
        double s = c.a / L;
        c.a = 1.0;
        dy = (y - c) * s;
    }
};

struct color_norm_t {
    __device__ inline double operator() (Color_d c) {
        return fmax(fmax(fabs(c.r), fabs(c.g)), fmax(fabs(c.b), fabs(c.a)));
    }
};

__global__
void ray_marching_kernel_rkv_double(ray_marching_parameters_t p) {
    // Pixel index.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= p.width || py >= p.height) return;
    register int idx = py * p.width + px;

    // Ray information.
    Lens::Ray ray = p.rays[idx];
    register Vector_d pos = p.pose.rotation.rotate(ray.origin) + p.pose.position;
    register Vector_d d = p.pose.rotation.rotate(ray.direction);

    // Initial color (background color).
    register Color_d color = p.bg_color;

    // Global ray information.
    float g_kin, g_kout;
    intersectBox(pos, d, p.bbox_min, p.bbox_max, &g_kin, &g_kout);
    if(g_kout < 0) {
        p.pixels[idx] = color;
        return;
    }
    if(g_kin < 0) g_kin = 0;

    // Block intersection.
    ray_marching_kernel_blockinfo_t blockinfos[512];
    traverse_stack_t stack[24];
    int blockinfos_count = kd_tree_block_intersection(pos, d, g_kin, g_kout, g_kin, g_kout, p.kd_tree, p.kd_tree_root, p.blocks, blockinfos, stack);

    // Simple solution: fixed step size.
    double kmax = g_kout;
    double L = p.blend_coefficient;

    // Render blocks.
    for(int cursor = 0; cursor < blockinfos_count; cursor++) {
        BlockDescription block = p.blocks[blockinfos[cursor].index];
        double kin = blockinfos[cursor].kin;
        double kout = blockinfos[cursor].kout;
        if(kout > kmax) kout = kmax;
        if(kin < kout) {
            // Render this block.
            double distance = kout - kin;
            double voxel_size = (block.max.x - block.min.x) / block.xsize; // assume voxels are cubes.
            block_interpolate_t block_access(block, p.data + block.offset);
            render_dxdt_t dxdt(block_access, pos, d, kin, kout, L);
            color_norm_t color_norm;
            Color_d new_color;
            RungeKuttaVerner(0.0, distance, color, dxdt, color_norm, 1e-5, voxel_size / 16.0, voxel_size, new_color);
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

class VolumeRendererImpl : public VolumeRenderer {
public:

    VolumeRendererImpl() :
        blocks(512),
        data(512 * 34 * 34 * 34),
        data_processed(512 * 34 * 34 * 34),
        kd_tree(512 * 5),
        rays(1000 * 1000),
        blend_coefficient(1.0),
        raycasting_method(kRK4Method),
        bbox_min(-1e20, -1e20, -1e20),
        bbox_max(1e20, 1e20, 1e20),
        bg_color(0, 0, 0, 0)
    {
        tf_texture_data = NULL;
        tf_texture_data_size = 0;
        floatChannelDesc = cudaCreateChannelDesc<float>();
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
        data_processed.allocate(volume->getDataSize());
        data.upload(volume->getData());
        blocks.allocate(block_count);
        for(int i = 0; i < block_count; i++) {
            blocks[i] = *volume->getBlockDescription(i);
        }
        blocks.upload();
        buildKDTree();
    }

    void preprocessVolume() {
        float tf_min, tf_max;
        tf->getDomain(tf_min, tf_max);
        TransferFunction::Scale tf_scale = tf->getScale();
        // For non-linear scales, process the min, max values as well.
        if(tf_scale == TransferFunction::kLogScale) {
            tf_min = log(tf_min);
            tf_max = log(tf_max);
        }
        // Preprocess the volume.
        preprocess_data_kernel<<<diviur(data.size, 64), 64>>>(data.gpu, data_processed.gpu, data.size, tf->getScale(), tf_min, tf_max);
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

        // Proprocess the scale of the transfer function.
        preprocessVolume();
        // Upload the transfer function.
        uploadTransferFunctionTexture();

        // Render kernel parameters.
        ray_marching_parameters_t pms;

        pms.rays = rays.gpu;
        pms.pixels = image->getPixelsGPU();
        pms.blocks = blocks.gpu;
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
        pms.bg_color = bg_color;
        // Block range.
        pms.pose = pose;
        int blockdim_x = 8; // 8x8 is the optimal block size.
        int blockdim_y = 8;
        bindTransferFunctionTexture();
        if(raycasting_method == kBasicBlendingMethod) {
            ray_marching_kernel_basic<<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
        }
        if(raycasting_method == kRK4Method) {
            ray_marching_kernel_rk4<<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
        }
        if(raycasting_method == kAdaptiveRKVMethod) {
            ray_marching_kernel_rkv_double<<<dim3(diviur(image->getWidth(), blockdim_x), diviur(image->getHeight(), blockdim_y), 1), dim3(blockdim_x, blockdim_y, 1)>>>(pms);
        }
        unbindTransferFunctionTexture();
        cudaThreadSynchronize();
    }

    // Memory regions:
    MirroredMemory<BlockDescription> blocks;
    MirroredMemory<float> data, data_processed;
    MirroredMemory<Lens::Ray> rays;

    int block_count;
    TransferFunction* tf;
    Lens* lens;
    Image* image;

    // Rendering parameters:
    Color bg_color;
    float blend_coefficient;
    RaycastingMethod raycasting_method;
    // Global bounding box:
    Vector bbox_min, bbox_max;
    // Pose:
    Pose pose;

    cudaChannelFormatDesc floatChannelDesc;

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
    void bindTransferFunctionTexture() {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        cudaBindTextureToArray(tf_texture, tf_texture_data, channel_desc);
    }
    void unbindTransferFunctionTexture() {
        cudaUnbindTexture(tf_texture);
    }
    cudaArray* tf_texture_data;
    size_t tf_texture_data_size;

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
        float sp_min = FLT_MAX, sp_max = FLT_MIN;
        for(int i = 0; i < block_count; i++) {
            sp_min = fminf(sp_min, blocks[blockids[i]].min[axis]);
            sp_max = fmaxf(sp_max, blocks[blockids[i]].max[axis]);
        }
        float split_value = (sp_min + sp_max) / 2.0f;

        int* blocks_left = new int[block_count];
        int blocks_left_count = 0;
        int* blocks_right = new int[block_count];
        int blocks_right_count = 0;

        float eps = (sp_max - sp_min) * 1e-6;
        for(int i = 0; i < block_count; i++) {
            if(blocks[blockids[i]].max[axis] < split_value + eps) {
                blocks_left[blocks_left_count++] = blockids[i];
            } else {
                blocks_right[blocks_right_count++] = blockids[i];
            }
        }
        // for(int i = 0; i < blocks_left_count; i++) {
        //     printf("L: %f\n", split_value - blocks[blocks_left[i]].max[axis]);
        // }
        // for(int i = 0; i < blocks_right_count; i++) {
        //     printf("R: %f\n", blocks[blocks_right[i]].min[axis] - split_value);
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
    }


    // void traverseKDTreeNode(Vector pos, Vector direction, int node, float kmin, float kmax) {
    //     int axis = kd_tree[node].split_axis;
    //     if(axis < 0) {
    //         printf("traverseBlock: %d\n", kd_tree[node].left);
    //         float tkmin, tkmax;
    //         intersectBox(pos, direction, blocks[kd_tree[node].left].min, blocks[kd_tree[node].left].max, &tkmin, &tkmax);
    //         printf("  %g %g\n", tkmax, tkmin);
    //         return;
    //     }
    //     float split_value = kd_tree[node].split_value;
    //     Vector pmin = pos + direction * kmin;
    //     Vector pmax = pos + direction * kmax;
    //     if(pmina <= split_value && pmaxa <= split_value) {
    //         traverseKDTreeNode(pos, direction, kd_tree[node].left, kmin, kmax);
    //     } else if(pmina >= split_value && pmaxa >= split_value) {
    //         traverseKDTreeNode(pos, direction, kd_tree[node].right, kmin, kmax);
    //     } else {
    //         float k_split = (split_value - pos[axis]) / direction[axis];
    //         if(pmina < split_value) {
    //             traverseKDTreeNode(pos, direction, kd_tree[node].right, k_split, kmax);
    //             traverseKDTreeNode(pos, direction, kd_tree[node].left, kmin, k_split);
    //         } else {
    //             traverseKDTreeNode(pos, direction, kd_tree[node].left, k_split, kmax);
    //             traverseKDTreeNode(pos, direction, kd_tree[node].right, kmin, k_split);
    //         }
    //     }
    // }
    // void traverseKDTree(Lens::Ray ray) {
    //     Vector pos = ray.origin;
    //     Vector direction = ray.direction;
    //     float kmin, kmax;
    //     intersectBox(pos, direction, bbox_min, bbox_max, &kmin, &kmax);
    //     if(kmin < 0) kmin = 0;
    //     if(kmax < kmin) kmax = kmin;
    //     printf("Traverse\n");
    //     //traverseKDTreeNode(pos, direction, kd_tree_root, kmin, kmax);
    //     traverse_stack_t stack[100];
    //     int stack_pointer = 0;
    //     stack[0].node = kd_tree_root;
    //     stack[0].kmin = kmin;
    //     stack[0].kmax = kmax;
    //     stack[0].stage = 0;
    //     while(stack_pointer >= 0) {
    //         traverse_stack_t& s = stack[stack_pointer];
    //         int axis = kd_tree[s.node].split_axis;
    //         if(axis < 0) {
    //             printf("traverseBlock: %d\n", kd_tree[s.node].left);
    //             float tkmin, tkmax;
    //             intersectBox(pos, direction, blocks[kd_tree[s.node].left].min, blocks[kd_tree[s.node].left].max, &tkmin, &tkmax);
    //             printf("  %g %g\n", tkmax, tkmin);
    //             stack_pointer -= 1;
    //         } else {
    //             float split_value = kd_tree[s.node].split_value;
    //             Vector pmin = pos + direction * s.kmin;
    //             Vector pmax = pos + direction * s.kmax;
    //             if(pmin[axis] <= split_value && pmaxa <= split_value) {
    //                 stack[stack_pointer].node = kd_tree[s.node].left;
    //                 stack[stack_pointer].kmin = s.kmin;
    //                 stack[stack_pointer].kmax = s.kmax;
    //                 // traverseKDTreeNode(pos, direction, kd_tree[s.node].left, kmin, kmax);
    //             } else if(pmin[axis] >= split_value && pmax[axis] >= split_value) {
    //                 stack[stack_pointer].node = kd_tree[s.node].right;
    //                 stack[stack_pointer].kmin = s.kmin;
    //                 stack[stack_pointer].kmax = s.kmax;
    //                 // traverseKDTreeNode(pos, direction, kd_tree[s.node].right, kmin, kmax);
    //             } else {
    //                 float k_split = (split_value - pos[axis]) / direction[axis];
    //                 if(pmin[axis] < split_value) {
    //                     stack_pointer += 1;
    //                     stack[stack_pointer].node = kd_tree[s.node].right;
    //                     stack[stack_pointer].kmin = k_split;
    //                     stack[stack_pointer].kmax = s.kmax;
    //                     s.node = kd_tree[s.node].left;
    //                     s.kmax = k_split;
    //                     // traverseKDTreeNode(pos, direction, kd_tree[s.node].right, k_split, kmax);
    //                     // traverseKDTreeNode(pos, direction, kd_tree[s.node].left, kmin, k_split);
    //                 } else {
    //                     stack_pointer += 1;
    //                     stack[stack_pointer].node = kd_tree[s.node].left;
    //                     stack[stack_pointer].kmin = k_split;
    //                     stack[stack_pointer].kmax = s.kmax;
    //                     s.node = kd_tree[s.node].right;
    //                     s.kmax = k_split;
    //                     // traverseKDTreeNode(pos, direction, kd_tree[s.node].left, k_split, kmax);
    //                     // traverseKDTreeNode(pos, direction, kd_tree[s.node].right, kmin, k_split);
    //                 }
    //             }
    //         }
    //     }
    // }
};

VolumeRenderer* VolumeRenderer::CreateGPU() {
    return new VolumeRendererImpl();
}

}
