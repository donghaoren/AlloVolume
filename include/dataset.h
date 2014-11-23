#ifndef ALLOVOLUME_DATASET_H
#define ALLOVOLUME_DATASET_H

#include "utils.h"
#include <stdlib.h>

namespace allovolume {

struct BlockDescription {
    Vector min, max;
    // size in each axis.
    // pixel at (x, y, z) = offset + x + y * xsize + z * xsize * ysize
    int xsize, ysize, zsize;
    int ghost_count;
    size_t offset;
};

class VolumeBlocks {
public:
    virtual float* getData() = 0;
    virtual size_t getDataSize() = 0;
    virtual int getBlockCount() = 0;
    virtual BlockDescription* getBlockDescription(int index) = 0;

    static void WriteToFile(VolumeBlocks* dataset, const char* path);
    static VolumeBlocks* LoadFromFile(const char* path);

    virtual ~VolumeBlocks() { }
};

VolumeBlocks* Dataset_FLASH_Create(const char* filename, const char* fieldname);

}

#endif
