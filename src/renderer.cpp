#include "renderer.h"

class Image {
public:
    virtual Color* getPixels() = 0;
    virtual Color* getPixelsGPU() = 0;
    virtual int getWidth() = 0;
    virtual int getHeight() = 0;
    virtual void setNeedsUpload() = 0;
    virtual void setNeedsDownload() = 0;

    virtual ~Image() { }

    static Image* Create(int width, int height);
};
