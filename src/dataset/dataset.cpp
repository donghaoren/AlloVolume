#include "dataset.h"
#include "stdio.h"
#include <string.h>

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
    for(int i = 0; i < block_count; i++) {
        fwrite(dataset->getBlockTreeInfo(i), sizeof(BlockTreeInfo), 1, fout);
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
        o_treeinfos = new BlockTreeInfo[o_block_count];
        fread(o_data, sizeof(float), o_data_size, fin);
        fread(o_blocks, sizeof(BlockDescription), o_block_count, fin);
        //fread(o_treeinfos, sizeof(BlockTreeInfo), o_block_count, fin);
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

    virtual BlockTreeInfo* getBlockTreeInfo(int index) {
        return &o_treeinfos[index];
    }

    virtual ~Dataset_File() {
        delete [] o_data;
        delete [] o_blocks;
    }

    float* o_data;
    size_t o_data_size;
    size_t o_block_count;
    BlockDescription* o_blocks;
    BlockTreeInfo* o_treeinfos;
};

VolumeBlocks* VolumeBlocks::LoadFromFile(const char* path) {
    return new Dataset_File(path);
}

class Dataset_Buffer : public VolumeBlocks {
public:

    struct Header {
        size_t block_count;
        size_t data_size;
    };

    Dataset_Buffer(const void* data, size_t length) {
        buffer = (unsigned char*)malloc(length);
        unsigned char* ptr = buffer;
        memcpy(buffer, data, length);

        Header* header = (Header*)ptr;
        ptr += sizeof(Header);

        o_block_count = header->block_count;
        o_data_size = header->data_size;


        o_data = (float*)ptr;
        ptr += o_data_size * sizeof(float);

        o_blocks = (BlockDescription*)ptr;
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

    virtual BlockTreeInfo* getBlockTreeInfo(int index) {
        return NULL;
    }

    virtual ~Dataset_Buffer() {
        free(buffer);
    }

    float* o_data;
    size_t o_data_size;
    size_t o_block_count;
    BlockDescription* o_blocks;
    unsigned char* buffer;
};

VolumeBlocks* VolumeBlocks::LoadFromBuffer(const void* buffer, size_t length) {
    return new Dataset_Buffer(buffer, length);
}
size_t VolumeBlocks::WriteToBufferSize(VolumeBlocks* dataset) {
    return sizeof(Dataset_Buffer::Header) + dataset->getBlockCount() * sizeof(BlockDescription) + dataset->getDataSize() * sizeof(float);
}
void VolumeBlocks::WriteToBuffer(VolumeBlocks* dataset, void* buffer, size_t length) {
    unsigned char* ptr = (unsigned char*)buffer;
    Dataset_Buffer::Header* hdr = (Dataset_Buffer::Header*)ptr;
    ptr += sizeof(Dataset_Buffer::Header);
    hdr->block_count = dataset->getBlockCount();
    hdr->data_size = dataset->getDataSize();
    float* data = (float*)ptr;
    ptr += sizeof(float) * dataset->getDataSize();
    memcpy(data, dataset->getData(), sizeof(float) * dataset->getDataSize());
    BlockDescription* blocks = (BlockDescription*)ptr;
    for(int i = 0; i < dataset->getBlockCount(); i++) {
        memcpy(&blocks[i], dataset->getBlockDescription(i), sizeof(BlockDescription));
    }
}

}
