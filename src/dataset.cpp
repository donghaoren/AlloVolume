#include "dataset.h"
#include "stdio.h"

namespace allovolume {

void VolumeBlocks::WriteToFile(VolumeBlocks* dataset, const char* path) {
    FILE* fout = fopen(path, "wb");
    size_t block_count = dataset->getBlockCount();
    size_t data_size = dataset->getDataSize();
    fwrite(&block_count, sizeof(size_t), 1, fout);
    fwrite(&data_size, sizeof(size_t), 1, fout);
    fwrite(dataset->getData(), sizeof(float), data_size, fout);
    for(int i = 0; i < block_count; i++) {
        fwrite(dataset->getBlockDescription(i), sizeof(BlockDescription), 1, fout);
    }
    fclose(fout);
}

class Dataset_File : public VolumeBlocks {
public:

    Dataset_File(const char* path) {
        FILE* fin = fopen(path, "rb");
        fread(&o_block_count, sizeof(size_t), 1, fin);
        fread(&o_data_size, sizeof(size_t), 1, fin);
        o_data = new float[o_data_size];
        o_blocks = new BlockDescription[o_block_count];
        fread(o_data, sizeof(float), o_data_size, fin);
        fread(o_blocks, sizeof(BlockDescription), o_block_count, fin);
        fclose(fin);
    }

    virtual float* getData() {
        return o_data;
    }

    virtual size_t getDataSize() {
        return o_data_size;
    }

    virtual int getBlockCount() {
        return o_block_count;
    }

    virtual BlockDescription* getBlockDescription(int index) {
        return &o_blocks[index];
    }

    virtual ~Dataset_File() {
        delete [] o_data;
        delete [] o_blocks;
    }

    float* o_data;
    size_t o_data_size;
    size_t o_block_count;
    BlockDescription* o_blocks;

};

VolumeBlocks* VolumeBlocks::LoadFromFile(const char* path) {
    return new Dataset_File(path);
}

}
