#ifndef ALLOVOLUME_CUDA_COMMON_H_INCLUDED
#define ALLOVOLUME_CUDA_COMMON_H_INCLUDED

namespace allovolume {

    class runtime_error { };
    class bad_alloc { };

    void* gpuAllocate(size_t size);
    void gpuDeallocate(void* pointer);
    void gpuUpload(void* dest, const void* src, size_t size);
    void gpuDownload(void* dest, const void* src, size_t size);

    template<typename T>
    inline T* cudaAllocate(size_t size) {
        return (T*)gpuAllocate(sizeof(T) * size);
    }

    template<typename T>
    inline void cudaDeallocate(T* pointer) {
        return gpuDeallocate(pointer);
    }

    template<typename T>
    void cudaUpload(T* dest, const T* src, size_t count) {
        gpuUpload(dest, src, sizeof(T) * count);
    }

    template<typename T>
    void cudaDownload(T* dest, T* src, size_t count) {
        gpuDownload(dest, src, sizeof(T) * count);
    }

    template<typename T>
    struct MirroredMemory {
        T* cpu;
        T* gpu;
        size_t size, capacity;
        MirroredMemory() {
            size = 0;
            capacity = 0;
            cpu = NULL;
            gpu = NULL;
        }
        MirroredMemory(int size_) {
            size = size_;
            capacity = size_;
            cpu = new T[capacity];
            gpu = cudaAllocate<T>(capacity);
        }
        T& operator [] (int index) { return cpu[index]; }
        const T& operator [] (int index) const { return cpu[index]; }
        void reserve(int count) {
            if(capacity < count) {
                capacity = count;
                if(cpu) delete [] cpu;
                if(gpu) cudaDeallocate<T>(gpu);
                cpu = new T[capacity];
                gpu = cudaAllocate<T>(capacity);
            }
        }
        template<typename T1>
        void allocate_type(size_t size_) {
            size_t T1_size = sizeof(T1) * size_;
            size_t Tcount = T1_size / sizeof(T);
            if(T1_size % sizeof(T) != 0) Tcount += 1;
            reserve(Tcount);
            size = Tcount;
        }
        void allocate(size_t size_) {
            reserve(size_);
            size = size_;
        }
        void upload(T* pointer) {
            cudaUpload(gpu, pointer, size);
        }
        void upload() {
            cudaUpload(gpu, cpu, size);
        }
        void download() {
            cudaDownload(cpu, gpu, size);
        }
        ~MirroredMemory() {
            if(cpu) delete [] cpu;
            if(gpu) cudaDeallocate<T>(gpu);
        }
    };
}

#endif
