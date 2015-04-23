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

    static void LevelsGPU(Image* img, float min, float max, float pow);
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
    virtual size_t getSize() = 0;

    virtual void setContent(const Color* color, size_t size) = 0;
    virtual void setDomain(float min, float max) = 0;
    virtual void setScale(Scale scale) = 0;

    virtual ~TransferFunction() { }

    static TransferFunction* CreateTransparent(float min, float max, Scale scale, size_t size);

    static TransferFunction* CreateGaussianTicks(float min, float max, Scale scale, int ticks);
    static TransferFunction* CreateLinearGradient(float min, float max, Scale scale);

    static void ParseLayers(TransferFunction* target, size_t size, const char* layers);
};

class Lens {
public:
    struct Ray {
        Vector origin, direction;
    };

    struct Viewport {
        // Size of the big image.
        int width, height;
        // Size of this viewport.
        int vp_x, vp_y, vp_width, vp_height;
    };

    virtual void setParameter(const char* name, const void* value) = 0;
    virtual void getParameter(const char* name, void* value) = 0;

    template<typename T>
    void set(const char* name, const T& value) { setParameter(name, &value); }

    template<typename T>
    T get(const char* name) { T value; getParameter(name, &value); return value; }

    // Common parameters.
    void setEyeSeparation(float value) { set<float>("eye_separation", value); }
    void setFocalDistance(float value) { set<float>("focal_distance", value); }
    float getEyeSeparation() { return get<float>("eye_separation"); }
    float getFocalDistance() { return get<float>("focal_distance"); }

    virtual void getRays(Viewport vp, Ray* rays) = 0;
    virtual void getRaysGPU(Viewport vp, Ray* rays) = 0;

    void getRays(int width, int height, Ray* rays) {
        Viewport vp;
        vp.width = width; vp.height = height;
        vp.vp_x = 0; vp.vp_y = 0; vp.vp_width = width; vp.vp_height = height;
        return getRays(vp, rays);
    }

    void getRaysGPU(int width, int height, Ray* rays) {
        Viewport vp;
        vp.width = width; vp.height = height;
        vp.vp_x = 0; vp.vp_y = 0; vp.vp_width = width; vp.vp_height = height;
        return getRaysGPU(vp, rays);
    }

    virtual ~Lens() { }

    static Lens* CreateEquirectangular();
    static Lens* CreatePerspective(float fovx);
};

struct Pose {
    Vector position;
    Quaternion rotation;
    Pose() : position(0, 0, 0), rotation(1, 0, 0, 0) { }
};

class VolumeRenderer {
public:
    enum RaycastingMethod {
        kBasicBlendingMethod = 0,
        kRK4Method = 1,
        kAdaptiveRKFMethod = 2,
        kPreIntegrationMethod = 3
    };

    struct ClipRange {
        float t_front, t_far;
    };

    // Set volume.
    virtual void setVolume(VolumeBlocks* volume) = 0;
    // This should be approximately the size of your volume.
    virtual void setBlendingCoefficient(float value) = 0;
    virtual float getBlendingCoefficient() = 0;
    virtual void setStepSizeMultiplier(float value) = 0;
    virtual float getStepSizeMultiplier() = 0;
    // Transfer function.
    virtual void setTransferFunction(TransferFunction* tf) = 0;
    // Set lens.
    virtual void setLens(Lens* lens) = 0;
    // Set pose.
    virtual void setPose(const Pose& pose) = 0;
    virtual Pose getPose() = 0;
    // Background color.
    virtual void setBackgroundColor(Color color) = 0;
    virtual Color getBackgroundColor() = 0;
    // Set clip range, which should be the same size as image and back_image.
    virtual void setClipRanges(ClipRange* ranges, size_t size) = 0;
    // Set output image.
    virtual void setImage(Image* image) = 0;
    virtual void setBackImage(Image* image) = 0;

    virtual void setBoundingBox(Vector min, Vector max) = 0;
    virtual void getBoundingBox(Vector& min, Vector& max) = 0;

    virtual void setRaycastingMethod(RaycastingMethod method) = 0;
    virtual RaycastingMethod getRaycastingMethod() = 0;

    // Render!
    virtual void render(int x0, int y0, int total_width, int total_height) = 0;
    virtual void render() = 0;
    virtual ~VolumeRenderer() { }

    static VolumeRenderer* CreateGPU();
};

}

#endif
