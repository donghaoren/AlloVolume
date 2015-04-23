#include "allovolume/allosphere_calibration.h"
#include "../opengl_include.h"

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
        Vector primary_direction = Vector(pos.z, pos.x, pos.y).safe_normalize();
        Vector lookat = primary_direction * focal_distance;
        r.origin = Vector(-primary_direction.y, primary_direction.x, 0) * eye_separation / 2.0f;
        r.direction = (lookat - r.origin).safe_normalize();
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

        blend_gltexture = 0;
        warp_gltexture = 0;

        warpWidth = projection->warpWidth;
        warpHeight = projection->warpHeight;
        blendWidth = projection->blendWidth;
        blendHeight = projection->blendHeight;
        warpData = new Vector4[warpWidth * warpHeight];
        blendData = new float[blendWidth * blendHeight];
        memcpy(warpData, projection->warpData, sizeof(Vector4) * warpWidth * warpHeight);
        memcpy(blendData, projection->blendData, sizeof(float) * blendWidth * blendHeight);
    }

    virtual void setParameter(const char* name, const void* value) {
        if(strcmp(name, "eye_separation") == 0) {
            eye_separation = *(float*)value;
        }
        if(strcmp(name, "focal_distance") == 0) {
            focal_distance = *(float*)value;
        }
    }

    virtual void getParameter(const char* name, void* value) {
        if(strcmp(name, "eye_separation") == 0) {
             *(float*)value = eye_separation;
        }
        if(strcmp(name, "focal_distance") == 0) {
            *(float*)value = focal_distance;
        }
    }

    virtual void getRays(Viewport vp, Ray* rays) { }
    virtual void getRaysGPU(Viewport vp, Ray* rays) {
        int width = vp.vp_width;
        int height = vp.vp_height;
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
        if(blend_gltexture) glDeleteTextures(1, &blend_gltexture);
        if(warp_gltexture) glDeleteTextures(1, &warp_gltexture);
        cudaFreeArray(warp_data);
        cudaFreeArray(blend_data);
        delete [] warpData;
        delete [] blendData;
    }

    float eye_separation, focal_distance;

    cudaArray* warp_data;
    cudaChannelFormatDesc channel_description;
    cudaArray* blend_data;
    cudaChannelFormatDesc channel_description_blend;

    int warpWidth, warpHeight, blendWidth, blendHeight;
    Vector4* warpData;
    float* blendData;


    virtual unsigned int getBlendTexture() {
        if(blend_gltexture) return blend_gltexture;
        glGenTextures(1, &blend_gltexture);
        glBindTexture(GL_TEXTURE_2D, blend_gltexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, blendWidth, blendHeight, 0, GL_LUMINANCE, GL_FLOAT, blendData);

        return blend_gltexture;
    }

    virtual unsigned int getWrapTexture() {
        if(warp_gltexture) return warp_gltexture;
        glGenTextures(1, &warp_gltexture);
        glBindTexture(GL_TEXTURE_2D, warp_gltexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, warpWidth, warpHeight, 0, GL_RGBA, GL_FLOAT, warpData);

        return warp_gltexture;
    }

    unsigned int blend_gltexture, warp_gltexture;
};

class AllosphereLensImpl_Wrapper : public AllosphereLens {
public:
    AllosphereLensImpl_Wrapper(Lens* lens_) : lens(lens_) {
        blend_texture = 0;
        wrap_texture = 0;
    }

    virtual void setParameter(const char* name, const void* value) {
        lens->setParameter(name, value);
    }

    virtual void getParameter(const char* name, void* value) {
        lens->getParameter(name, value);
    }

    virtual void getRays(Viewport vp, Ray* rays) {
        return lens->getRays(vp, rays);
    }

    virtual void getRaysGPU(Viewport vp, Ray* rays) {
        return lens->getRaysGPU(vp, rays);
    }

    virtual void performBlend(Image* img) {
        int width = img->getWidth(), height = img->getHeight();
        Color* pixels = img->getPixelsGPU();
        allosphere_lens_noblend_kernel<<< dim3(diviur(width, 8), diviur(height, 8), 1), dim3(8, 8, 1) >>>(pixels, width, height);
        cudaThreadSynchronize();
    }

    virtual unsigned int getBlendTexture() {
        if(blend_texture) return blend_texture;
        glGenTextures(1, &blend_texture);
        glBindTexture(GL_TEXTURE_2D, blend_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        int wh = 1024;
        float* data = new float[wh * wh];
        for(int i = 0; i < wh * wh; i++) {
            data[i] = 1.0f;
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, wh, wh, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, data);

        delete [] data;

        return blend_texture;
    }
    virtual unsigned int getWrapTexture() {
        if(wrap_texture) return wrap_texture;
        glGenTextures(1, &wrap_texture);
        glBindTexture(GL_TEXTURE_2D, wrap_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        float es_cached = lens->getEyeSeparation();
        lens->setEyeSeparation(0);

        int width = 1024;
        int height = 512;
        Vector* data = new Vector[width * height];
        Ray* rays = new Ray[width * height];
        Lens::Viewport vp;
        vp.width = width;
        vp.height = height;
        vp.vp_x = 0; vp.vp_y = 0; vp.vp_width = width; vp.vp_height = height;
        lens->getRays(vp, rays);
        for(int i = 0; i < width * height; i++) {
            data[i] = rays[i].direction;
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, data);

        delete [] data;
        delete [] rays;

        lens->setEyeSeparation(es_cached);

        return wrap_texture;
    }

    ~AllosphereLensImpl_Wrapper() {
        delete lens;
        if(blend_texture) glDeleteTextures(1, &blend_texture);
        if(wrap_texture) glDeleteTextures(1, &wrap_texture);
    }

    unsigned int blend_texture, wrap_texture;

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
