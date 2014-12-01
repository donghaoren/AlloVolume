#include "cuda_common.h"
#include <stdio.h>

namespace allovolume {

    void* gpuAllocate(size_t size) {
        void* result = 0;
        cudaError_t err = cudaMalloc(&result, size);
        if(!result) {
            fprintf(stderr, "cudaAllocate: cudaMalloc() of %llu (%.2f MB): %s\n",
                size, size / 1048576.0,
                cudaGetErrorString(err));
            size_t memory_free, memory_total;
            cudaMemGetInfo(&memory_free, &memory_total);
            fprintf(stderr, "  Free: %.2f MB, Total: %.2f MB\n", (float)memory_free / 1048576.0, (float)memory_total / 1048576.0);
            throw bad_alloc();
        }
        return result;
    }

    void gpuDeallocate(void* pointer) {
        cudaFree(pointer);
    }

    void gpuUpload(void* dest, const void* src, size_t size) {
        cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
        if(err != cudaSuccess) {
            fprintf(stderr, "cudaUpload: cudaMemcpy(): %s\n", cudaGetErrorString(err));
            throw runtime_error();
        }
    }

    void gpuDownload(void* dest, const void* src, size_t size) {
        cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
            fprintf(stderr, "cudaUpload: cudaMemcpy(): %s\n", cudaGetErrorString(err));
            throw runtime_error();
        }
    }

}
