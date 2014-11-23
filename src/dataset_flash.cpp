#include "dataset.h"
#include "hdf5.h"

#include "stdio.h"

namespace allovolume {

class Dataset_FLASH : public VolumeBlocks {
public:

    static int clampi(int x, int min, int max) {
        if(x < min) return min;
        if(x > max) return max;
        return x;
    }

    Dataset_FLASH(const char* filename, const char* fieldname) {
        file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

        dataset_id = H5Dopen2(file_id, fieldname, H5P_DEFAULT);
        hid_t dataspace = H5Dget_space(dataset_id);
        int rank = H5Sget_simple_extent_ndims(dataspace);
        hsize_t dims[4];
        H5Sget_simple_extent_dims(dataspace, dims, NULL);

        // Read data.
        int block_count = dims[0];
        int xsize = dims[1], ysize = dims[2], zsize = dims[3];
        int block_size = xsize * ysize * zsize;
        int data_size = block_count * block_size;
        float* data = new float[data_size];
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, NULL, NULL, H5P_DEFAULT, data);

        // Read block bounding boxes.
        float* bboxes = new float[block_count * 3 * 2];
        hid_t dataset_bbox = H5Dopen2(file_id, "/bounding box", H5P_DEFAULT);
        H5Dread(dataset_bbox, H5T_NATIVE_FLOAT, NULL, NULL, H5P_DEFAULT, bboxes);

        // Read block node types.
        int* node_types = new int[block_count];
        hid_t dataset_node_type = H5Dopen2(file_id, "/node type", H5P_DEFAULT);
        H5Dread(dataset_node_type, H5T_NATIVE_INT, NULL, NULL, H5P_DEFAULT, node_types);

        int* gid = new int[block_count * 15];
        hid_t dataset_gid = H5Dopen2(file_id, "/gid", H5P_DEFAULT);
        H5Dread(dataset_gid, H5T_NATIVE_INT, NULL, NULL, H5P_DEFAULT, gid);

        int* refine_level = new int[block_count];
        hid_t dataset_refine_level = H5Dopen2(file_id, "/refine level", H5P_DEFAULT);
        H5Dread(dataset_refine_level, H5T_NATIVE_INT, NULL, NULL, H5P_DEFAULT, refine_level);

        int ghost_count = 1;

        int o_xsize = xsize + ghost_count * 2;
        int o_ysize = ysize + ghost_count * 2;
        int o_zsize = zsize + ghost_count * 2;
        int o_block_size = o_xsize * o_ysize * o_zsize;

        o_data_size = block_count * o_block_size;
        o_data = new float[o_data_size];
        for(int i = 0; i < o_data_size; i++) o_data[i] = 0;
        for(size_t i = 0; i < block_count; i++) {
            float* volume = data + i * block_size;
            float* o_volume = o_data + i * o_block_size;
            for(int x = 0; x < o_xsize; x++) {
                for(int y = 0; y < o_ysize; y++) {
                    for(int z = 0; z < o_zsize; z++) {
                        int ix = clampi(x - ghost_count, 0, xsize - 1);
                        int iy = clampi(y - ghost_count, 0, ysize - 1);
                        int iz = clampi(z - ghost_count, 0, zsize - 1);
                        int idx = ix + iy * xsize + iz * xsize * ysize;
                        int o_idx = x + y * o_xsize + z * o_xsize * o_ysize;
                        o_volume[o_idx] = volume[idx];
                    }
                }
            }
        }

        o_block_count = 0;
        bool* is_leaf = new bool[block_count];
        int max_refine_level = 65536;
        for(size_t i = 0; i < block_count; i++) {
            is_leaf[i] = refine_level[i] == max_refine_level || (refine_level[i] < max_refine_level && node_types[i] == 1);
            if(is_leaf[i]) {
                o_block_count += 1;
            }
        }

        o_blocks = new BlockDescription[o_block_count];
        int idx = 0;
        for(size_t i = 0; i < block_count; i++) {
            if(!is_leaf[i]) continue;
            o_blocks[idx].min.x = bboxes[i * 6 + 0];
            o_blocks[idx].max.x = bboxes[i * 6 + 1];
            o_blocks[idx].min.y = bboxes[i * 6 + 2];
            o_blocks[idx].max.y = bboxes[i * 6 + 3];
            o_blocks[idx].min.z = bboxes[i * 6 + 4];
            o_blocks[idx].max.z = bboxes[i * 6 + 5];
            o_blocks[idx].xsize = xsize + ghost_count * 2;
            o_blocks[idx].ysize = ysize + ghost_count * 2;
            o_blocks[idx].zsize = zsize + ghost_count * 2;
            o_blocks[idx].ghost_count = 1;
            o_blocks[idx].offset = i * o_block_size;
            idx += 1;
            // blocks[i].is_leaf = node_types[i] == 1;
            // for(int k = 0; k < 6; k++) {
            //     int p = gid[i * 15 + k] - 1;
            //     blocks[i].neighbors[k] = p >= 0 ? p : -1;
            // }
            // {
            //     int p = gid[i * 15 + 6] - 1;
            //     blocks[i].parent = p >= 0 ? p : -1;
            // }
            // for(int k = 0; k < 8; k++) {
            //     int p = gid[i * 15 + 7 + k] - 1;
            //     blocks[i].children[k] = p >= 0 ? p : -1;
            // }
        }
        // // Fixup neighbors.
        // for(size_t i = 0; i < block_count; i++) {
        //     for(int k = 0; k < 6; k++) {
        //         if(blocks[i].neighbors[k] < 0) {
        //             int p = blocks[i].parent;
        //             while(p >= 0) {
        //                 if(blocks[p].neighbors[k] >= 0) {
        //                     blocks[i].neighbors[k] = blocks[p].neighbors[k];
        //                     break;
        //                 }
        //                 p = blocks[p].parent;
        //             }
        //         }
        //     }
        // }
        delete [] data;
        delete [] bboxes;
        delete [] node_types;
        delete [] gid;
        H5Sclose(dataspace);
        H5Dclose(dataset_id);
        H5Dclose(dataset_bbox);
        H5Fclose(file_id);

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

    virtual ~Dataset_FLASH() {
        delete [] o_data;
        delete [] o_blocks;
    }

    hid_t file_id, dataset_id;
    float* o_data;
    size_t o_data_size;
    int o_block_count;
    BlockDescription* o_blocks;

};

VolumeBlocks* Dataset_FLASH_Create(const char* filename, const char* fieldname) {
    return new Dataset_FLASH(filename, fieldname);
}

}
