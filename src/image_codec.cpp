#include "renderer.h"

#define cimg_display 0
#include "CImg.h"

using namespace cimg_library;

namespace allovolume {
    bool writeImageFile(const char* path, const char* format, int width, int height, Color* pixels) {
        CImg<float> image(width, height, 1, 4);
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int idx = x + y * width;
                image(x, y, 0) = pixels[idx].r * 255;
                image(x, y, 1) = pixels[idx].g * 255;
                image(x, y, 2) = pixels[idx].b * 255;
                image(x, y, 3) = pixels[idx].a * 255;
            }
        }
        image.save(path);
    }
}
