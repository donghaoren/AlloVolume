#include "renderer.h"
#include "FreeImage.h"
#include <string.h>

namespace allovolume {
    void freeimage_initialize() {
        static bool initialized = false;
        if(!initialized) {
            FreeImage_Initialise();
            initialized = true;
        }
    }

    struct rgba64_t { unsigned short r, g, b, a; };
    struct rgba32_t { unsigned char r, g, b, a; };

    int clamp_pixel_8(float v) {
        int x = v * 255;
        if(x < 0) return 0;
        if(x > 255) return 255;
        return x;
    }
    int clamp_pixel_16(float v) {
        int x = v * 65535;
        if(x < 0) return 0;
        if(x > 65535) return 65535;
        return x;
    }

    bool writeImageFile(const char* path, const char* format, int width, int height, Color* pixels) {
        freeimage_initialize();
        if(strcmp(format, "png16") == 0) {
            FIBITMAP* bitmap = FreeImage_AllocateT(FIT_RGBA16, width, height, 64);
            for(int y = 0; y < height; y++) {
                rgba64_t* scanline = (rgba64_t*)FreeImage_GetScanLine(bitmap, y);
                for(int x = 0; x < width; x++) {
                    Color pixel = pixels[y * width + x];
                    scanline[x].r = clamp_pixel_16(pixel.r);
                    scanline[x].g = clamp_pixel_16(pixel.g);
                    scanline[x].b = clamp_pixel_16(pixel.b);
                    scanline[x].a = clamp_pixel_16(1);
                }
            }
            FreeImage_Save(FIF_PNG, bitmap, path);
        }
        if(strcmp(format, "png") == 0) {
            FIBITMAP* bitmap = FreeImage_AllocateT(FIT_BITMAP, width, height, 32);
            for(int y = 0; y < height; y++) {
                rgba32_t* scanline = (rgba32_t*)FreeImage_GetScanLine(bitmap, y);
                for(int x = 0; x < width; x++) {
                    Color pixel = pixels[y * width + x];
                    scanline[x].r = clamp_pixel_8(pixel.r);
                    scanline[x].g = clamp_pixel_8(pixel.g);
                    scanline[x].b = clamp_pixel_8(pixel.b);
                    scanline[x].a = clamp_pixel_8(1);
                }
            }
            FreeImage_Save(FIF_PNG, bitmap, path);
        }
    }
}
