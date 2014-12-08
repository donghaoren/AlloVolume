#include "renderer.h"
#include <float.h>
#include <stdio.h>
#include <math_functions.h>
#include <algorithm>

#include "cuda_common.h"

namespace allovolume {

__global__
void get_rays_kernel(Lens::Ray* rays, int pixel_count, int width, int height, float focal_distance, float eye_separation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pixel_count) return;
    int x = idx % width;
    int y = idx / width;
    float theta = -((float)(x + 0.5) / width - 0.5f) * PI * 2;
    float phi = -((float)(y + 0.5) / height - 0.5f) * PI;
    Vector origin = Vector(0, 0, 0);
    Vector lookat = Vector(cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)) * focal_distance;
    Vector shift = Vector(-sin(theta) * cos(phi), cos(theta) * cos(phi), 0) * eye_separation;
    origin += shift;
    rays[idx].origin = origin;
    rays[idx].direction = (lookat - origin).normalize();
}

class LensImpl_StereoAware : public Lens {
public:
    LensImpl_StereoAware() {
        eye_separation = 0.0f;
        focal_distance = 1.0f;
    }

    virtual void setParameter(const char* name, const void* value) {
        if(strcmp(name, "eye_separation") == 0) eye_separation = *(float*)value;
        if(strcmp(name, "focal_distance") == 0) focal_distance = *(float*)value;
    }

    float eye_separation, focal_distance;
};

class LensImpl_Equirectangular : public LensImpl_StereoAware {
public:
    LensImpl_Equirectangular() {
    }
    virtual void getRays(int width, int height, Ray* rays) {
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int p = y * width + x;
                float theta = -((float)(x + 0.5) / width - 0.5f) * PI * 2;
                float phi = -((float)(y + 0.5) / height - 0.5f) * PI;
                Vector origin = Vector(0, 0, 0);
                Vector lookat = Vector(cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)) * focal_distance;
                Vector shift = Vector(-sin(theta) * cos(phi), cos(theta) * cos(phi), 0) * eye_separation;
                origin += shift;
                rays[p].origin = origin;
                rays[p].direction = (lookat - origin).normalize();
            }
        }
    }
    virtual void getRaysGPU(int width, int height, Ray* rays) {
        int number_of_threads = 64;
        int pixel_count = width * height;
        int n_blocks = diviur(pixel_count, number_of_threads);
        get_rays_kernel<<<n_blocks, number_of_threads>>>(rays, pixel_count, width, height, focal_distance, eye_separation);
    }
};

__global__
void get_rays_kernel_perspective(Lens::Ray* rays, int pixel_count, int width, int height, float focal_distance, float eye_separation, float screen_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pixel_count) return;
    int x = idx % width;
    int y = idx / width;
    float kx = -((float)(x + 0.5) / width - 0.5f);
    float ky = -((float)(y + 0.5) / height - 0.5f) * ((float)height / (float)width);
    Vector origin = Vector(0, 0, 0);
    Vector lookat = Vector(1.0f, kx * screen_width, ky * screen_width) * focal_distance;
    Vector shift = Vector(0, eye_separation, 0);
    origin += shift;
    rays[idx].origin = origin;
    rays[idx].direction = (lookat - origin).normalize();
}

class LensImpl_Perspective : public LensImpl_StereoAware {
public:
    LensImpl_Perspective(float fovx_ = PI / 4.0f) : fovx(fovx_) { }

    virtual void getRays(int width, int height, Ray* rays) {
        float screen_width = tan(fovx / 2.0f) * 2.0f;
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int p = y * width + x;
                float kx = -((float)(x + 0.5) / width - 0.5f);
                float ky = -((float)(y + 0.5) / height - 0.5f) * ((float)height / (float)width);
                Vector origin = Vector(0, 0, 0);
                Vector lookat = Vector(1.0f, kx * screen_width, ky * screen_width) * focal_distance;
                Vector shift = Vector(0, eye_separation, 0);
                origin += shift;
                rays[p].origin = origin;
                rays[p].direction = (lookat - origin).normalize();
            }
        }
    }
    virtual void getRaysGPU(int width, int height, Ray* rays) {
        int number_of_threads = 64;
        int pixel_count = width * height;
        int n_blocks = diviur(pixel_count, number_of_threads);
        float screen_width = tan(fovx / 2.0f) * 2.0f;
        get_rays_kernel_perspective<<<n_blocks, number_of_threads>>>(rays, pixel_count, width, height, focal_distance, eye_separation, screen_width);
    }

    float fovx;
};

Lens* Lens::CreateEquirectangular() {
    return new LensImpl_Equirectangular();
}

Lens* Lens::CreatePerspective(float fovx) {
    return new LensImpl_Perspective(fovx);
}

Lens* Lens::CreateOrthogonal(float spanx) {

}

}
