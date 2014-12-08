#include "allosphere_calibration.h"

namespace allovolume {

texture<float4, 2, cudaReadModeElementType> warp_texture;
texture<float, 2, cudaReadModeElementType> blend_texture;

__global__ void allosphere_lens_get_rays_kernel(Lens::Ray* rays, int width, int height, Vector origin) {
    int px = threadIdx.x + blockDim.x * blockIdx.x;
    int py = threadIdx.y + blockDim.y * blockIdx.y;
    if(px < width && py < height) {
        float fx = ((float)px + 0.5f) / (float)width;
        float fy = ((float)py + 0.5f) / (float)height;
        float4 pos = tex2D(warp_texture, fx, 1.0f - fy);
        Lens::Ray r;
        r.origin = origin;
        r.direction = Vector(pos.z, pos.x, pos.y).normalize();
        rays[py * width + px] = r;
    }
}

__device__ float transform_color(float c) {
    c = (c - 20.0 / 255.0) / (128.0 / 255.0 - 20.0 / 255.0);
    if(c < 0) c = 0;
    if(c > 1) c = 1;
    c = c * c;
    return c;
}

__global__ void allosphere_lens_blend_kernel(Color* data, int width, int height) {
    int px = threadIdx.x + blockDim.x * blockIdx.x;
    int py = threadIdx.y + blockDim.y * blockIdx.y;
    if(px < width && py < height) {
        float fx = ((float)px + 0.5f) / (float)width;
        float fy = ((float)py + 0.5f) / (float)height;
        float blend = tex2D(blend_texture, fx, fy);
        Color r = data[py * width + px];
        r.r = transform_color(r.r);
        r.g = transform_color(r.g);
        r.b = transform_color(r.b);
        r *= blend;
        r.a = 1.0f;
        // data[py * width + px] = r;
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

        origin = Vector(0, 0, 0);
    }

    virtual void setParameter(const char* name, const void* value) {
        if(strcmp(name, "origin") == 0) {
            origin = *(Vector*)value;
        }
    }
    virtual void getRays(int width, int height, Ray* rays) { }
    virtual void getRaysGPU(int width, int height, Ray* rays) {
        cudaBindTextureToArray(warp_texture, warp_data, channel_description);
        allosphere_lens_get_rays_kernel<<< dim3(diviur(width, 8), diviur(height, 8), 1), dim3(8, 8, 1) >>>(rays, width, height, origin);
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

    Vector origin;

    cudaArray* warp_data;
    cudaChannelFormatDesc channel_description;
    cudaArray* blend_data;
    cudaChannelFormatDesc channel_description_blend;
};

AllosphereLens* AllosphereCalibration::CreateLens(AllosphereCalibration::Projection* projection) {
    return new AllosphereLensImpl(projection);
}

}
