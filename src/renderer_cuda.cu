#include "renderer.h"
#include <math.h>
#include <algorithm>

#define CUDA_MAX_THREADS 1024

using namespace std;

namespace allovolume {

class ImageImpl : public Image {
public:
    ImageImpl(int width_, int height_) {
        width = width_; height = height_;
        data_cpu = new Color[width * height];
        cudaMalloc((void**)&data_gpu, sizeof(Color) * width * height);
        needs_upload = false;
        needs_download = false;
    }

    virtual Color* getPixels() {
        if(needs_download) {
            cudaMemcpy(data_cpu, data_gpu, sizeof(Color) * width * height, cudaMemcpyDeviceToHost);
            needs_download = false;
        }
        return data_cpu;
    }
    virtual Color* getPixelsGPU() {
        if(needs_upload) {
            cudaMemcpy(data_gpu, data_cpu, sizeof(Color) * width * height, cudaMemcpyHostToDevice);
            needs_upload = false;
        }
        return data_gpu;
    }
    virtual int getWidth() { return width; }
    virtual int getHeight() { return height; }
    virtual void setNeedsUpload() {
        needs_upload = true;
    }
    virtual void setNeedsDownload() {
        needs_download = true;
    }

    virtual ~ImageImpl() {
        delete [] data_cpu;
        cudaFree(data_gpu);
    }

    int width, height;
    Color *data_cpu, *data_gpu;
    bool needs_upload, needs_download;
};

Image* Image::Create(int width, int height) { return new ImageImpl(width, height); }

__global__ void get_rays_kernel(Vector ex, Vector ey, Vector ez, Vector origin, int width, int height, int pixel_count, Lens::Ray* rays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pixel_count) return;
    int x = idx % width;
    int y = idx / width;
    float theta = ((float)x / width - 0.5f) * PI * 2;
    float phi = -((float)y / height - 0.5f) * PI;
    rays[idx].origin = origin;
    Vector direction = ex * cos(theta) * cos(phi) + ey * sin(theta) * cos(phi) + ez * sin(phi);
    direction = direction.normalize();
    rays[idx].direction = direction;
}

class LensImpl : public Lens {
public:
    LensImpl(Vector origin_, Vector up_, Vector direction_) {
        origin = origin_;
        up = up_;
        direction = direction_;
        is_stereo = false;
    }
    LensImpl(Vector origin_, Vector up_, Vector direction_, float eye_separation_, float radius_) {
        origin = origin_;
        up = up_;
        direction = direction_;
        is_stereo = true;
        eye_separation = eye_separation_;
        radius = radius_;
    }
    virtual Vector getCenter() {
        return origin;
    }
    virtual void setParameter(const char* name, void* value) {
        if(strcmp(name, "origin") == 0) origin = *(Vector*)value;
        if(strcmp(name, "up") == 0) up = *(Vector*)value;
        if(strcmp(name, "direction") == 0) direction = *(Vector*)value;
        if(strcmp(name, "eye_separation") == 0) eye_separation = *(float*)value;
        if(strcmp(name, "radius") == 0) eye_separation = *(float*)value;
    }
    virtual void getRays(int width, int height, Ray* rays) {
        Vector ex = direction.normalize();
        Vector ey = ex.cross(up).normalize();
        Vector ez = ex.cross(ey).normalize();
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int p = y * width + x;
                float theta = ((float)x / width - 0.5f) * PI * 2;
                float phi = -((float)y / height - 0.5f) * PI;
                rays[p].origin = origin;
                rays[p].direction = ex * cos(theta) * cos(phi) + ey * sin(theta) * cos(phi) + ez * sin(phi);
                rays[p].direction = rays[p].direction.normalize();
            }
        }
    }
    virtual void getRaysGPU(int width, int height, Ray* rays) {
        Vector ex = direction.normalize();
        Vector ey = ex.cross(up).normalize();
        Vector ez = ex.cross(ey).normalize();
        int pixel_count = width * height;
        int cuda_blocks = pixel_count / CUDA_MAX_THREADS;
        if(pixel_count % CUDA_MAX_THREADS != 0) cuda_blocks += 1;
        get_rays_kernel<<<cuda_blocks, CUDA_MAX_THREADS>>>(ex, ey, ez, origin, width, height, pixel_count, rays);
    }

    Vector origin, up, direction;
    float eye_separation, radius;
    bool is_stereo;
};

Lens* Lens::CreateEquirectangular(Vector origin, Vector up, Vector direction) {
    return new LensImpl(origin, up, direction);
}
Lens* Lens::CreateEquirectangularStereo(Vector origin, Vector up, Vector direction, float eye_separation, float radius) {
    return new LensImpl(origin, up, direction, eye_separation, radius);
}

class TransferFunctionImpl : public TransferFunction {
public:
    virtual void setParameter(const char* name, void* value) {
    }

    virtual Metadata* getMetadata() {
        return &metadata;
    }
    virtual Color* getContent() {
    }
    virtual Color* getContentGPU() {
    }

    Metadata metadata;
};

TransferFunction* TransferFunction::CreateTest() {
    return new TransferFunctionImpl();
}

template<typename T>
struct MirroredMemory {
    T* cpu;
    T* gpu;
    size_t size, capacity;
    MirroredMemory(int size_) {
        size = size_;
        capacity = size_;
        cpu = new T[capacity];
        cudaMalloc((void**)&gpu, sizeof(T) * capacity);
    }
    T& operator [] (int index) { return cpu[index]; }
    const T& operator [] (int index) const { return cpu[index]; }
    void reserve(int count) {
        if(capacity < count) {
            capacity = count * 2;
            delete [] cpu;
            cudaFree(gpu);
            cpu = new T[capacity];
            cudaMalloc((void**)&gpu, sizeof(T) * capacity);
        }
    }
    void allocate(size_t size_) {
        reserve(size_);
        size = size_;
    }
    void upload(T* pointer) {
        cudaMemcpy(gpu, pointer, sizeof(T) * size, cudaMemcpyHostToDevice);
    }
    void upload() {
        cudaMemcpy(gpu, cpu, sizeof(T) * size, cudaMemcpyHostToDevice);
    }
    void download() {
        cudaMemcpy(cpu, gpu, sizeof(T) * size, cudaMemcpyDeviceToHost);
    }
    ~MirroredMemory() {
        delete [] cpu;
        cudaFree(gpu);
    }
};

struct ray_marching_parameters_t {
    Lens::Ray* rays;
    Color* pixels;
    BlockDescription* blocks;
    float* data;
    int pixel_count;
    int block_count;
    float step_size;
};

inline __device__ float fminf(float a, float b) { return min(a, b); }
inline __device__ float fmaxf(float a, float b) { return max(a, b); }
inline __device__ Vector fminf(Vector a, Vector b) { return Vector(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
inline __device__ Vector fmaxf(Vector a, Vector b) { return Vector(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
inline __device__ Vector vmul(Vector a, Vector b) { return Vector(a.x * b.x, a.y * b.y, a.z * b.z); }

inline __device__
int intersectBox(Vector origin, Vector direction, Vector boxmin, Vector boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    Vector invR = Vector(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
    Vector tbot = vmul(invR, boxmin - origin);
    Vector ttop = vmul(invR, boxmax - origin);

    // re-order intersections to find smallest and largest on each axis
    Vector tmin = fminf(ttop, tbot);
    Vector tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

inline __device__ float intersect_bbox(Vector p, Vector direction, Vector bmin, Vector bmax) {
    // if inside, no intersection.
    if(p <= bmax && p >= bmin) return 0;
    // Test for x.
    if(p.x < bmin.x && direction.x > 0) {
        float t = (bmin.x - p.x) / direction.x;
        Vector r = p + direction * t;
        if(r.y >= bmin.y && r.y <= bmax.y && r.z >= bmin.z && r.z <= bmax.z) {
            // there's only one possible, if we find it, return.
            return t;
        }
    }
    if(p.x > bmax.x && direction.x < 0) {
        float t = (bmax.x - p.x) / direction.x;
        Vector r = p + direction * t;
        if(r.y >= bmin.y && r.y <= bmax.y && r.z >= bmin.z && r.z <= bmax.z) {
            return t;
        }
    }
    if(p.y < bmin.y && direction.y > 0) {
        float t = (bmin.y - p.y) / direction.y;
        Vector r = p + direction * t;
        if(r.x >= bmin.x && r.x <= bmax.x && r.z >= bmin.z && r.z <= bmax.z) {
            return t;
        }
    }
    if(p.y > bmax.y && direction.y < 0) {
        float t = (bmax.y - p.y) / direction.y;
        Vector r = p + direction * t;
        if(r.x >= bmin.x && r.x <= bmax.x && r.z >= bmin.z && r.z <= bmax.z) {
            return t;
        }
    }
    if(p.z < bmin.z && direction.z > 0) {
        float t = (bmin.z - p.z) / direction.z;
        Vector r = p + direction * t;
        if(r.x >= bmin.x && r.x <= bmax.x && r.y >= bmin.y && r.y <= bmax.y) {
            return t;
        }
    }
    if(p.z > bmax.z && direction.z < 0) {
        float t = (bmax.z - p.z) / direction.z;
        Vector r = p + direction * t;
        if(r.x >= bmin.x && r.x <= bmax.x && r.y >= bmin.y && r.y <= bmax.y) {
            return t;
        }
    }
    // no intersection.
    return -1;
}

// inline __device__ float intersect_bbox_outpoint(Vector p, Vector direction, Vector bmin, Vector bmax) {

// }

inline __device__ int clamp(int value, int min, int max) {
    if(value < min) return min;
    if(value > max) return max;
    return value;
}

inline __device__ float access_volume(float* data, int xsize, int ysize, int zsize, int ix, int iy, int iz) {
    //return data[ix * ysize * zsize + iy * zsize + iz];
    return data[iz * xsize * ysize + iy * xsize + ix];
}

__device__ float block_interploate(Vector pos, Vector min, Vector max, float* data, int xsize, int ysize, int zsize) {
    // [ 0 | 1 | 2 | 3 ]
    Vector p(
        (pos.x - min.x) / (max.x - min.x) * xsize - 0.5f,
        (pos.y - min.y) / (max.y - min.y) * ysize - 0.5f,
        (pos.z - min.z) / (max.z - min.z) * zsize - 0.5f
    );
    int ix = p.x, iy = p.y, iz = p.z;
    ix = clamp(ix, 0, xsize - 2);
    iy = clamp(iy, 0, ysize - 2);
    iz = clamp(iz, 0, zsize - 2);
    float tx = p.x - ix, ty = p.y - iy, tz = p.z - iz;
    float t000 = access_volume(data, xsize, ysize, zsize, ix, iy, iz);
    float t001 = access_volume(data, xsize, ysize, zsize, ix, iy, iz + 1);
    float t010 = access_volume(data, xsize, ysize, zsize, ix, iy + 1, iz);
    float t011 = access_volume(data, xsize, ysize, zsize, ix, iy + 1, iz + 1);
    float t100 = access_volume(data, xsize, ysize, zsize, ix + 1, iy, iz);
    float t101 = access_volume(data, xsize, ysize, zsize, ix + 1, iy, iz + 1);
    float t110 = access_volume(data, xsize, ysize, zsize, ix + 1, iy + 1, iz);
    float t111 = access_volume(data, xsize, ysize, zsize, ix + 1, iy + 1, iz + 1);
    float t00 = t000 * (1.0f - tz) + t001 * tz;
    float t01 = t010 * (1.0f - tz) + t011 * tz;
    float t10 = t100 * (1.0f - tz) + t101 * tz;
    float t11 = t110 * (1.0f - tz) + t111 * tz;
    float t0 = t00 * (1.0f - ty) + t01 * ty;
    float t1 = t10 * (1.0f - ty) + t11 * ty;
    float t = t0 * (1.0 - tx) + t1 * tx;
    return t;
}

__global__ void ray_marching_kernel(ray_marching_parameters_t p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= p.pixel_count) return;
    Lens::Ray ray = p.rays[idx];
    Vector pos = ray.origin;
    Vector d = ray.direction;
    int block_cursor = 0;
    Color color(0, 0, 0, 0);
    float integral = 0;
    // Simple solution: fixed step size.
    while(block_cursor < p.block_count) {
        BlockDescription block = p.blocks[block_cursor];
        float kin, kout;
        if(intersectBox(pos, d, block.min, block.max, &kin, &kout) && kout >= 0) {
            if(kin < 0) kin = 0;
            // Render this block.
            float distance = kout - kin;

            int steps = 4;
            for(int i = 0; i < steps; i++) {
                float k = kin + distance * ((float)i + 0.5f) / steps;
                Vector pt = pos + d * k;
                float value = block_interploate(pt, block.min, block.max, p.data + block.offset, block.xsize, block.ysize, block.zsize);
                float v = log(1 + value);
                v = sin(v * 2) * sin(v * 2) * v / 10;
                // pt /= 1e9;
                // float v = sin(pt.x / 4) + sin(pt.y / 4) + sin(pt.z / 4);
                // v = v * v / 30;
                integral += v * distance / steps;
            }
        }
        block_cursor += 1;
    }
    color.r = color.g = color.b = integral / 1e10;
    color.a = 1;//integral / 1e10;
    p.pixels[idx] = color;
}

class VolumeRendererImpl : public VolumeRenderer {
public:

    VolumeRendererImpl() : blocks(500), data(500 * 32 * 32 * 32), rays(1000 * 1000) {
    }

    struct BlockCompare {
        BlockCompare(Vector center_) {
            center = center_;
        }
        bool operator () (const BlockDescription& a, const BlockDescription& b) {
            float d1 = ((a.min + a.max) / 2.0f - center).len2();
            float d2 = ((b.min + b.max) / 2.0f - center).len2();
            return d1 < d2;
        }

        Vector center;
    };

    virtual void setVolume(VolumeBlocks* volume_) {
        // Copy volume data.
        volume = volume_;
        data.allocate(volume->getDataSize());
        data.upload(volume->getData());

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
    virtual void render() {
        // Prepare blocks.
        int block_count = volume->getBlockCount();
        blocks.allocate(block_count);
        for(int i = 0; i < block_count; i++) {
            blocks[i] = *volume->getBlockDescription(i);
        }
        // Sort blocks.
        BlockCompare block_compare(lens->getCenter());
        //sort(blocks.cpu, blocks.cpu + block_count, block_compare);
        blocks.upload();

        // Prepare image.
        int pixel_count = image->getWidth() * image->getHeight();
        rays.allocate(pixel_count);
        lens->getRaysGPU(image->getWidth(), image->getHeight(), rays.gpu);

        int cuda_blocks = pixel_count / CUDA_MAX_THREADS;
        if(pixel_count % CUDA_MAX_THREADS != 0) cuda_blocks += 1;
        ray_marching_parameters_t pms;
        pms.rays = rays.gpu;
        pms.pixels = image->getPixelsGPU();
        pms.blocks = blocks.gpu;
        pms.data = data.gpu;
        pms.pixel_count = pixel_count;
        pms.block_count = block_count;
        pms.step_size = 0.01;
        ray_marching_kernel<<<cuda_blocks, CUDA_MAX_THREADS>>>(pms);
    }

    MirroredMemory<BlockDescription> blocks;
    MirroredMemory<float> data;
    MirroredMemory<Lens::Ray> rays;
    VolumeBlocks* volume;
    TransferFunction* tf;
    Lens* lens;
    Image* image;
};

VolumeRenderer* VolumeRenderer::CreateGPU() {
    return new VolumeRendererImpl();
}

}
