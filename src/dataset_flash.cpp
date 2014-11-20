#include "dataset.h"
#include "hdf5.h"

#include "stdio.h"

namespace allovolume {

class Dataset_FLASH : public VolumeBlocks {
public:

    Dataset_FLASH(const char* filename, const char* fieldname) {
        file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

        dataset_id = H5Dopen2(file_id, fieldname, H5P_DEFAULT);
        hid_t dataspace = H5Dget_space(dataset_id);
        int rank = H5Sget_simple_extent_ndims(dataspace);
        hsize_t dims[4];
        H5Sget_simple_extent_dims(dataspace, dims, NULL);

        // Read data.
        data = new float[dims[0] * dims[1] * dims[2] * dims[3]];
        data_size = dims[0] * dims[1] * dims[2] * dims[3];
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, NULL, NULL, H5P_DEFAULT, data);

        // Read block bounding boxes.
        float* bboxes = new float[dims[0] * 3 * 2];
        hid_t dataset_bbox = H5Dopen2(file_id, "/bounding box", H5P_DEFAULT);
        H5Dread(dataset_bbox, H5T_NATIVE_FLOAT, NULL, NULL, H5P_DEFAULT, bboxes);

        // Read block node types.
        int* node_types = new int[dims[0]];
        hid_t dataset_node_type = H5Dopen2(file_id, "/node type", H5P_DEFAULT);
        H5Dread(dataset_node_type, H5T_NATIVE_INT, NULL, NULL, H5P_DEFAULT, node_types);

        // Compute block count.
        block_count = 0;
        for(int i = 0; i < dims[0]; i++) {
            if(node_types[i] == 1) block_count += 1;
        }
        blocks = new BlockDescription[block_count];
        int idx = 0;
        for(size_t i = 0; i < dims[0]; i++) {
            if(node_types[i] == 1) {
                blocks[idx].min.x = bboxes[i * 6 + 0];
                blocks[idx].max.x = bboxes[i * 6 + 1];
                blocks[idx].min.y = bboxes[i * 6 + 2];
                blocks[idx].max.y = bboxes[i * 6 + 3];
                blocks[idx].min.z = bboxes[i * 6 + 4];
                blocks[idx].max.z = bboxes[i * 6 + 5];
                blocks[idx].xsize = dims[1];
                blocks[idx].ysize = dims[2];
                blocks[idx].zsize = dims[3];
                blocks[idx].offset = i * dims[1] * dims[2] * dims[3];
                idx += 1;
            }
        }

        H5Sclose(dataspace);
        H5Dclose(dataset_id);
        H5Dclose(dataset_bbox);
        H5Fclose(file_id);

    }

    virtual float* getData() {
        return data;
    }

    virtual size_t getDataSize() {
        return data_size;
    }

    virtual int getBlockCount() {
        return block_count;
    }

    virtual BlockDescription* getBlockDescription(int index) {
        return &blocks[index];
    }

    virtual ~Dataset_FLASH() {
        delete [] data;
        delete [] blocks;
    }

    hid_t file_id, dataset_id;
    float* data;
    size_t data_size;
    int block_count;
    BlockDescription* blocks;

};

VolumeBlocks* Dataset_FLASH_Create(const char* filename, const char* fieldname) {
    return new Dataset_FLASH(filename, fieldname);
}

}
