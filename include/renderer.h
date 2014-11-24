#ifndef ALLOVOLUME_RENDERER_H
#define ALLOVOLUME_RENDERER_H

#include "dataset.h"
#include "utils.h"

namespace allovolume {

class Image {
public:
    virtual Color* getPixels() = 0;
    virtual Color* getPixelsGPU() = 0;
    virtual int getWidth() = 0;
    virtual int getHeight() = 0;
    virtual void setNeedsUpload() = 0;
    virtual void setNeedsDownload() = 0;
    virtual void save(const char* path, const char* format) = 0;

    virtual ~Image() { }

    static Image* Create(int width, int height);
};

class TransferFunction {
public:
    struct Metadata {
        float input_min, input_max;
        float blend_coefficient;
        int size;
        bool is_log_scale;
    };

    virtual void setParameter(const char* name, void* value) = 0;

    virtual Metadata* getMetadata() = 0;
    virtual Color* getContent() = 0;
    virtual Color* getContentGPU() = 0;

    virtual ~TransferFunction() { }

    static TransferFunction* CreateTest(float min, float max, int ticks, bool is_log);
};

class Lens {
public:
    struct Ray {
        Vector origin, direction;
    };

    virtual void setParameter(const char* name, void* value) = 0;
    virtual Vector getCenter() = 0;
    virtual void getRays(int width, int height, Ray* rays) = 0;
    virtual void getRaysGPU(int width, int height, Ray* rays) = 0;

    virtual ~Lens() { }

    static Lens* CreateEquirectangular(Vector origin, Vector up, Vector direction);
    static Lens* CreateEquirectangularStereo(Vector origin, Vector up, Vector direction, float eye_separation, float radius);
};

class VolumeRenderer {
public:
    virtual void setVolume(VolumeBlocks* volume) = 0;
    virtual void setTransferFunction(TransferFunction* tf) = 0;
    virtual void setLens(Lens* lens) = 0;
    virtual void setImage(Image* image) = 0;
    virtual void render() = 0;
    virtual ~VolumeRenderer() { }

    static VolumeRenderer* CreateGPU();
};

bool writeImageFile(const char* path, const char* format, int width, int height, Color* pixels);

}

#endif
