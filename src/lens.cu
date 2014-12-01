#include "renderer.h"
#include <float.h>
#include <stdio.h>
#include <math_functions.h>
#include <algorithm>

#include "cuda_common.h"

namespace allovolume {

__global__
void get_rays_kernel(Vector ex, Vector ey, Vector ez, Vector origin, int width, int height, int pixel_count, Lens::Ray* rays) {
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
        Vector ey = up.cross(ex).normalize();
        Vector ez = ex.cross(ey).normalize();
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int p = y * width + x;
                float theta = ((float)x / (float)width - 0.5f) * PI * 2;
                float phi = -((float)y / (float)height - 0.5f) * PI;
                rays[p].origin = origin;
                rays[p].direction = ex * cos(theta) * cos(phi) + ey * sin(theta) * cos(phi) + ez * sin(phi);
                rays[p].direction = rays[p].direction.normalize();
            }
        }
    }
    virtual void getRaysGPU(int width, int height, Ray* rays) {
        Vector ex = direction.normalize();
        Vector ey = up.cross(ex).normalize();
        Vector ez = ex.cross(ey).normalize();
        int pixel_count = width * height;
        int cuda_blocks = pixel_count / CUDA_DEFAULT_THREADS;
        if(pixel_count % CUDA_DEFAULT_THREADS != 0) cuda_blocks += 1;
        get_rays_kernel<<<cuda_blocks, CUDA_DEFAULT_THREADS>>>(ex, ey, ez, origin, width, height, pixel_count, rays);
    }

    Vector origin, up, direction;
    float eye_separation, radius;
    bool is_stereo;
};

Lens* Lens::CreateEquirectangular(Vector origin, Vector up, Vector direction, float eye_separation, float radius) {
    return new LensImpl(origin, up, direction);
}

}
