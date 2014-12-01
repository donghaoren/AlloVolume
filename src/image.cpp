#include "renderer.h"
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

}
