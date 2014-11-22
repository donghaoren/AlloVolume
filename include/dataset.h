#ifndef ALLOVOLUME_DATASET_H
#define ALLOVOLUME_DATASET_H

#include "utils.h"
#include <stdlib.h>

namespace allovolume {

struct BlockDescription {
    Vector min, max;
    int xsize, ysize, zsize; // size in each axis.
    size_t offset;
};

class VolumeBlocks {
public:
    virtual float* getData() = 0;
    virtual size_t getDataSize() = 0;
    virtual int getBlockCount() = 0;
    virtual BlockDescription* getBlockDescription(int index) = 0;

    virtual ~VolumeBlocks() { }
};

VolumeBlocks* Dataset_FLASH_Create(const char* filename, const char* fieldname);

}

#endif
