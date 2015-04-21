#include "allovolume/dataset.h"
#include "allovolume/renderer.h"
#include <stdio.h>

using namespace allovolume;

int main(int argc, char* argv[]) {
    VolumeBlocks* volume = Dataset_FLASH_Create(argv[1], argv[2]);
    VolumeBlocks::WriteToFile(volume, argv[3]);
    delete volume;
}
