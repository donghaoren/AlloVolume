#ifndef ALLOVOLUME_RENDERER_H_INCLUDED
#define ALLOVOLUME_RENDERER_H_INCLUDED

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
    virtual bool save(const char* path, const char* format) = 0;

    virtual ~Image() { }

    static Image* Create(int width, int height);
    static bool WriteImageFile(const char* path, const char* format, int width, int height, Color* pixels);
};

class TransferFunction {
public:
    enum Scale {
        kLinearScale = 1,
        kLogScale = 2
    };

    // Get domain.
    virtual void getDomain(float& min, float& max) = 0;
    virtual Scale getScale() = 0;

    virtual Color* getContent() = 0;
    virtual Color* getContentGPU() = 0;
    virtual size_t getSize() = 0;

    virtual void setContent(const Color* color, size_t size) = 0;
    virtual void setDomain(float min, float max) = 0;
    virtual void setScale(Scale scale) = 0;

    virtual ~TransferFunction() { }

    static TransferFunction* CreateTransparent(float min, float max, Scale scale, size_t size);

    static TransferFunction* CreateGaussianTicks(float min, float max, Scale scale, int ticks);
    static TransferFunction* CreateLinearGradient(float min, float max, Scale scale);
};

class Lens {
public:
    struct Ray {
        Vector origin, direction;
    };

    virtual void setParameter(const char* name, const void* value) = 0;

    template<typename T>
    void set(const char* name, const T& value) { setParameter(name, &value); }

    // Common parameters.
    void setEyeSeparation(float value) { set<float>("eye_separation", value); }
    void setFocalDistance(float value) { set<float>("focal_distance", value); }

    virtual void getRays(int width, int height, Ray* rays) = 0;
    virtual void getRaysGPU(int width, int height, Ray* rays) = 0;

    virtual ~Lens() { }

    static Lens* CreateEquirectangular();
    static Lens* CreatePerspective(float fovx);
    static Lens* CreateOrthogonal(float spanx);
};

struct Pose {
    Vector position;
    Quaternion rotation;
    Pose() : position(0, 0, 0), rotation(1, 0, 0, 0) { }
};

class VolumeRenderer {
public:
    // Set volume.
    virtual void setVolume(VolumeBlocks* volume) = 0;
    // This should be approximately the size of your volume.
    virtual void setBlendingCoefficient(float value) = 0;
    // Transfer function.
    virtual void setTransferFunction(TransferFunction* tf) = 0;
    // Set lens.
    virtual void setLens(Lens* lens) = 0;
    // Set pose.
    virtual void setPose(const Pose& pose) = 0;
    // Set output image.
    virtual void setImage(Image* image) = 0;
    // Render!
    virtual void render() = 0;
    virtual ~VolumeRenderer() { }

    static VolumeRenderer* CreateGPU();
};

}

#endif
