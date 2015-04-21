#include "allovolume/renderer.h"
#include "cuda_common.h"

namespace allovolume {

class ImageImpl : public Image {
public:
    ImageImpl(int width_, int height_) {
        width = width_; height = height_;
        data_cpu = new Color[width * height];
        data_gpu = cudaAllocate<Color>(width * height);
        needs_upload = false;
        needs_download = false;
    }

    virtual Color* getPixels() {
        if(needs_download) {
            cudaDownload<Color>(data_cpu, data_gpu, width * height);
            needs_download = false;
        }
        return data_cpu;
    }
    virtual Color* getPixelsGPU() {
        if(needs_upload) {
            cudaUpload<Color>(data_gpu, data_cpu, width * height);
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

    virtual bool save(const char* path, const char* format) {
        Color* pixels = getPixels();
        return Image::WriteImageFile(path, format, width, height, pixels);
    }

    virtual ~ImageImpl() {
        delete [] data_cpu;
        cudaDeallocate(data_gpu);
    }

    int width, height;
    Color *data_cpu, *data_gpu;
    bool needs_upload, needs_download;
};

Image* Image::Create(int width, int height) { return new ImageImpl(width, height); }

__global__
void kernel_levels(Color* pixels, int width, int height, float levels_min, float levels_max, float levels_pow) {
    // Pixel index.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= width || py >= height) return;
    register int idx = py * width + px;
    Color c = pixels[idx];
    c.r = powf(__saturatef((c.r - levels_min) / (levels_max - levels_min)), levels_pow);
    c.g = powf(__saturatef((c.g - levels_min) / (levels_max - levels_min)), levels_pow);
    c.b = powf(__saturatef((c.b - levels_min) / (levels_max - levels_min)), levels_pow);
    pixels[idx] = c;
}

void Image::LevelsGPU(Image* img, float min, float max, float pow) {
    kernel_levels<<<dim3(diviur(img->getWidth(), 8), diviur(img->getHeight(), 8), 1), dim3(8, 8, 1)>>>(img->getPixelsGPU(), img->getWidth(), img->getHeight(), min, max, pow);
}

}
