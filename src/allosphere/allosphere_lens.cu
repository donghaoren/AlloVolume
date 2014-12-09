#include "allosphere_calibration.h"

namespace allovolume {

texture<float4, 2, cudaReadModeElementType> warp_texture;
texture<float, 2, cudaReadModeElementType> blend_texture;

__global__ void allosphere_lens_get_rays_kernel(Lens::Ray* rays, int width, int height, float eye_separation, float focal_distance) {
    int px = threadIdx.x + blockDim.x * blockIdx.x;
    int py = threadIdx.y + blockDim.y * blockIdx.y;
    if(px < width && py < height) {
        float fx = ((float)px + 0.5f) / (float)width;
        float fy = ((float)py + 0.5f) / (float)height;
        float4 pos = tex2D(warp_texture, fx, 1.0f - fy);
        Lens::Ray r;
        Vector primary_direction = Vector(pos.z, pos.x, pos.y).normalize();
        Vector lookat = primary_direction * focal_distance;
        r.origin = Vector(-primary_direction.y, primary_direction.x, 0) * eye_separation / 2.0f;
        r.direction = (lookat - r.origin).normalize();
        rays[py * width + px] = r;
    }
}

__global__ void allosphere_lens_blend_kernel(Color* data, int width, int height) {
    int px = threadIdx.x + blockDim.x * blockIdx.x;
    int py = threadIdx.y + blockDim.y * blockIdx.y;
    if(px < width && py < height) {
        float fx = ((float)px + 0.5f) / (float)width;
        float fy = ((float)py + 0.5f) / (float)height;
        float blend = tex2D(blend_texture, fx, fy);
        Color r = data[py * width + px];
        r.a *= blend;
        r.r *= r.a;
        r.g *= r.a;
        r.b *= r.a;
        r.a = 1;
        data[py * width + px] = r;
    }
}

__global__ void allosphere_lens_noblend_kernel(Color* data, int width, int height) {
    int px = threadIdx.x + blockDim.x * blockIdx.x;
    int py = threadIdx.y + blockDim.y * blockIdx.y;
    if(px < width && py < height) {
        float fx = ((float)px + 0.5f) / (float)width;
        float fy = ((float)py + 0.5f) / (float)height;
        Color r = data[py * width + px];
        r.r *= r.a;
        r.g *= r.a;
        r.b *= r.a;
        r.a = 1;
        data[py * width + px] = r;
    }
}


class AllosphereLensImpl : public AllosphereLens {
public:
    AllosphereLensImpl(AllosphereCalibration::Projection* projection) {
        // Warp texture.
        channel_description = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&warp_data, &channel_description, projection->warpWidth, projection->warpHeight);
        cudaMemcpyToArray(warp_data, 0, 0,
            projection->warpData,
            sizeof(Vector4) * projection->warpWidth * projection->warpHeight,
            cudaMemcpyHostToDevice);
        warp_texture.normalized = 1;
        warp_texture.filterMode = cudaFilterModeLinear;
        warp_texture.addressMode[0] = cudaAddressModeClamp;
        warp_texture.addressMode[1] = cudaAddressModeClamp;


        // Blend texture.
        channel_description_blend = cudaCreateChannelDesc<float>();
        cudaMallocArray(&blend_data, &channel_description_blend, projection->blendWidth, projection->blendHeight);
        cudaMemcpyToArray(blend_data, 0, 0,
            projection->blendData,
            sizeof(float) * projection->blendWidth * projection->blendHeight,
            cudaMemcpyHostToDevice);
        blend_texture.normalized = 1;
        blend_texture.filterMode = cudaFilterModeLinear;
        blend_texture.addressMode[0] = cudaAddressModeClamp;
        blend_texture.addressMode[1] = cudaAddressModeClamp;

        eye_separation = 0;
        focal_distance = 1;
    }

    virtual void setParameter(const char* name, const void* value) {
        if(strcmp(name, "eye_separation") == 0) {
            eye_separation = *(float*)value;
        }
        if(strcmp(name, "focal_distance") == 0) {
            focal_distance = *(float*)value;
        }
    }
    virtual void getRays(int width, int height, Ray* rays) { }
    virtual void getRaysGPU(int width, int height, Ray* rays) {
        cudaBindTextureToArray(warp_texture, warp_data, channel_description);
        allosphere_lens_get_rays_kernel<<< dim3(diviur(width, 8), diviur(height, 8), 1), dim3(8, 8, 1) >>>(rays, width, height, eye_separation, focal_distance);
        cudaThreadSynchronize();
        cudaUnbindTexture(warp_texture);
    }

    virtual void performBlend(Image* img) {
        cudaBindTextureToArray(blend_texture, blend_data, channel_description_blend);
        int width = img->getWidth(), height = img->getHeight();
        Color* pixels = img->getPixelsGPU();
        allosphere_lens_blend_kernel<<< dim3(diviur(width, 8), diviur(height, 8), 1), dim3(8, 8, 1) >>>(pixels, width, height);
        cudaThreadSynchronize();
        cudaUnbindTexture(blend_texture);
    }

    virtual ~AllosphereLensImpl() {
    }

    float eye_separation, focal_distance;

    cudaArray* warp_data;
    cudaChannelFormatDesc channel_description;
    cudaArray* blend_data;
    cudaChannelFormatDesc channel_description_blend;
};

class AllosphereLensImpl_Wrapper : public AllosphereLens {
public:
    AllosphereLensImpl_Wrapper(Lens* lens_) : lens(lens_) { }

    virtual void setParameter(const char* name, const void* value) {
        lens->setParameter(name, value);
    }

    virtual void getRays(int width, int height, Ray* rays) {
        return lens->getRays(width, height, rays);
    }

    virtual void getRaysGPU(int width, int height, Ray* rays) {
        return lens->getRaysGPU(width, height, rays);
    }

    virtual void performBlend(Image* img) {
        int width = img->getWidth(), height = img->getHeight();
        Color* pixels = img->getPixelsGPU();
        allosphere_lens_noblend_kernel<<< dim3(diviur(width, 8), diviur(height, 8), 1), dim3(8, 8, 1) >>>(pixels, width, height);
        cudaThreadSynchronize();
    }

    ~AllosphereLensImpl_Wrapper() {
        delete lens;
    }

    Lens* lens;
};

AllosphereLens* AllosphereCalibration::CreateLens(AllosphereCalibration::Projection* projection) {
    if(projection->warpData && projection->blendData) {
        return new AllosphereLensImpl(projection);
    } else {
        return new AllosphereLensImpl_Wrapper(Lens::CreatePerspective(PI / 2));
    }
}

}
