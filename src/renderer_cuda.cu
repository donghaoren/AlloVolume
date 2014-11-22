#include "renderer.h"
#include <math.h>
#include <algorithm>

#define CUDA_MAX_THREADS 1024

//#define CPU_EMULATE

using namespace std;

namespace allovolume {

#ifndef CPU_EMULATE

template<typename T>
T* cudaAllocate(size_t size) {
    T* result;
    cudaMalloc((void**)&result, sizeof(T) * size);
    return result;
}

template<typename T>
void cudaDeallocate(T* pointer) {
    cudaFree(pointer);
}

template<typename T>
void cudaUpload(T* dest, T* src, size_t count) {
    cudaMemcpy(dest, src, sizeof(T) * count, cudaMemcpyHostToDevice);
}

template<typename T>
void cudaDownload(T* dest, T* src, size_t count) {
    cudaMemcpy(dest, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
}

#define CUDA_KERNEL __global__
#define CUDA_DEVICE __device__

#else

template<typename T>
T* cudaAllocate(size_t size) {
    return new T[size];
}

template<typename T>
void cudaDeallocate(T* pointer) {
    delete [] pointer;
}

template<typename T>
void cudaUpload(T* dest, T* src, size_t count) {
    memcpy(dest, src, sizeof(T) * count);
}

template<typename T>
void cudaDownload(T* dest, T* src, size_t count) {
    memcpy(dest, src, sizeof(T) * count);
}

struct int3 { int x, y, z; };
int3 blockDim, blockIdx, threadIdx;

#define CUDA_KERNEL
#define CUDA_DEVICE

#endif

class ImageImpl : public Image {
public:
    ImageImpl(int width_, int height_) {
        width = width_; height = height_;
        data_cpu = new Color[width * height];
        data_gpu = cudaAllocate<Color>(width * height);
        needs_upload = false;
        needs_download = false;
    }

    virtual Color* getPixels() {
        if(needs_download) {
            cudaDownload<Color>(data_cpu, data_gpu, width * height);
            needs_download = false;
        }
        return data_cpu;
    }
    virtual Color* getPixelsGPU() {
        if(needs_upload) {
            cudaUpload<Color>(data_gpu, data_cpu, width * height);
            needs_upload = false;
        }
        return data_gpu;
    }
    virtual int getWidth() { return width; }
    virtual int getHeight() { return height; }
    virtual void setNeedsUpload() {
        needs_upload = true;
    }
    virtual void setNeedsDownload() {
        needs_download = true;
    }

    virtual ~ImageImpl() {
        delete [] data_cpu;
        cudaDeallocate(data_gpu);
    }

    int width, height;
    Color *data_cpu, *data_gpu;
    bool needs_upload, needs_download;
};

Image* Image::Create(int width, int height) { return new ImageImpl(width, height); }

CUDA_KERNEL void get_rays_kernel(Vector ex, Vector ey, Vector ez, Vector origin, int width, int height, int pixel_count, Lens::Ray* rays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pixel_count) return;
    int x = idx % width;
    int y = idx / width;
    float theta = ((float)x / width - 0.5f) * PI * 2;
    float phi = -((float)y / height - 0.5f) * PI;
    rays[idx].origin = origin;
    Vector direction = ex * cos(theta) * cos(phi) + ey * sin(theta) * cos(phi) + ez * sin(phi);
    direction = direction.normalize();
    rays[idx].direction = direction;
}

class LensImpl : public Lens {
public:
    LensImpl(Vector origin_, Vector up_, Vector direction_) {
        origin = origin_;
        up = up_;
        direction = direction_;
        is_stereo = false;
    }
    LensImpl(Vector origin_, Vector up_, Vector direction_, float eye_separation_, float radius_) {
        origin = origin_;
        up = up_;
        direction = direction_;
        is_stereo = true;
        eye_separation = eye_separation_;
        radius = radius_;
    }
    virtual Vector getCenter() {
        return origin;
    }
    virtual void setParameter(const char* name, void* value) {
        if(strcmp(name, "origin") == 0) origin = *(Vector*)value;
        if(strcmp(name, "up") == 0) up = *(Vector*)value;
        if(strcmp(name, "direction") == 0) direction = *(Vector*)value;
        if(strcmp(name, "eye_separation") == 0) eye_separation = *(float*)value;
        if(strcmp(name, "radius") == 0) eye_separation = *(float*)value;
    }
    virtual void getRays(int width, int height, Ray* rays) {
        Vector ex = direction.normalize();
        Vector ey = ex.cross(up).normalize();
        Vector ez = ey.cross(ex).normalize();
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int p = y * width + x;
                float theta = ((float)x / width - 0.5f) * PI * 2;
                float phi = -((float)y / height - 0.5f) * PI;
                rays[p].origin = origin;
                rays[p].direction = ex * cos(theta) * cos(phi) + ey * sin(theta) * cos(phi) + ez * sin(phi);
                rays[p].direction = rays[p].direction.normalize();
            }
        }
    }
    virtual void getRaysGPU(int width, int height, Ray* rays) {
        Vector ex = direction.normalize();
        Vector ey = ex.cross(up).normalize();
        Vector ez = ey.cross(ex).normalize();
        int pixel_count = width * height;
        int cuda_blocks = pixel_count / CUDA_MAX_THREADS;
        if(pixel_count % CUDA_MAX_THREADS != 0) cuda_blocks += 1;
    #ifndef CPU_EMULATE
        get_rays_kernel<<<cuda_blocks, CUDA_MAX_THREADS>>>(ex, ey, ez, origin, width, height, pixel_count, rays);
    #else
        blockDim.x = CUDA_MAX_THREADS;
        for(int i = 0; i < cuda_blocks; i++) {
            for(int j = 0; j < CUDA_MAX_THREADS; j++) {
                blockIdx.x = i;
                threadIdx.x = j;
                get_rays_kernel(ex, ey, ez, origin, width, height, pixel_count, rays);
            }
        }
    #endif
    }

    Vector origin, up, direction;
    float eye_separation, radius;
    bool is_stereo;
};

Lens* Lens::CreateEquirectangular(Vector origin, Vector up, Vector direction) {
    return new LensImpl(origin, up, direction);
}
Lens* Lens::CreateEquirectangularStereo(Vector origin, Vector up, Vector direction, float eye_separation, float radius) {
    return new LensImpl(origin, up, direction, eye_separation, radius);
}

Color rainbow_colormap[] = { Color(0.471412, 0.108766, 0.527016), Color(0.445756, 0.110176, 0.549008), Color(0.420099, 0.111586, 0.571001), Color(0.394443, 0.112997, 0.592993), Color(0.368787, 0.114407, 0.614986), Color(0.34313, 0.115817, 0.636978), Color(0.317474, 0.117227, 0.658971), Color(0.30382, 0.130517, 0.677031), Color(0.294167, 0.147766, 0.69378), Color(0.284514, 0.165015, 0.71053), Color(0.274861, 0.182264, 0.727279), Color(0.265208, 0.199513, 0.744028), Color(0.255555, 0.216762, 0.760777), Color(0.250196, 0.236254, 0.772907), Color(0.249132, 0.257991, 0.780416), Color(0.248069, 0.279728, 0.787925), Color(0.247005, 0.301465, 0.795434), Color(0.245941, 0.323202, 0.802943), Color(0.244878, 0.344939, 0.810452), Color(0.244962, 0.366259, 0.815542), Color(0.248488, 0.386326, 0.813373), Color(0.252015, 0.406394, 0.811204), Color(0.255542, 0.426461, 0.809035), Color(0.259069, 0.446529, 0.806867), Color(0.262595, 0.466596, 0.804698), Color(0.266122, 0.486664, 0.802529), Color(0.27249, 0.50249, 0.792471), Color(0.278857, 0.518316, 0.782413), Color(0.285225, 0.534141, 0.772355), Color(0.291592, 0.549967, 0.762297), Color(0.29796, 0.565793, 0.752239), Color(0.304327, 0.581619, 0.742181), Color(0.312466, 0.593997, 0.728389), Color(0.321196, 0.605227, 0.713353), Color(0.329926, 0.616456, 0.698317), Color(0.338656, 0.627685, 0.683282), Color(0.347385, 0.638915, 0.668246), Color(0.356115, 0.650144, 0.65321), Color(0.366029, 0.659446, 0.637262), Color(0.377127, 0.666821, 0.620403), Color(0.388225, 0.674195, 0.603544), Color(0.399323, 0.681569, 0.586684), Color(0.410421, 0.688944, 0.569825), Color(0.421519, 0.696318, 0.552966), Color(0.433185, 0.702972, 0.536335), Color(0.446557, 0.707463, 0.520393), Color(0.459929, 0.711955, 0.504451), Color(0.473301, 0.716446, 0.488509), Color(0.486673, 0.720937, 0.472566), Color(0.500045, 0.725429, 0.456624), Color(0.513417, 0.72992, 0.440682), Color(0.528494, 0.732128, 0.427547), Color(0.543572, 0.734335, 0.414412), Color(0.558649, 0.736543, 0.401277), Color(0.573727, 0.738751, 0.388142), Color(0.588804, 0.740958, 0.375007), Color(0.603882, 0.743166, 0.361872), Color(0.619337, 0.743583, 0.351457), Color(0.634919, 0.743402, 0.34195), Color(0.650501, 0.743222, 0.332443), Color(0.666083, 0.743042, 0.322935), Color(0.681665, 0.742861, 0.313428), Color(0.697247, 0.742681, 0.303921), Color(0.712199, 0.740876, 0.2961), Color(0.726521, 0.737447, 0.289966), Color(0.740842, 0.734018, 0.283832), Color(0.755164, 0.730589, 0.277698), Color(0.769486, 0.727159, 0.271564), Color(0.783808, 0.72373, 0.26543), Color(0.797308, 0.719143, 0.259858), Color(0.808342, 0.711081, 0.255976), Color(0.819376, 0.703019, 0.252094), Color(0.83041, 0.694957, 0.248211), Color(0.841444, 0.686895, 0.244329), Color(0.852478, 0.678833, 0.240446), Color(0.863512, 0.670771, 0.236564), Color(0.869512, 0.6567, 0.23336), Color(0.875513, 0.642629, 0.230157), Color(0.881513, 0.628557, 0.226953), Color(0.887513, 0.614486, 0.22375), Color(0.893514, 0.600415, 0.220546), Color(0.899514, 0.586344, 0.217343), Color(0.901235, 0.567363, 0.213599), Color(0.901529, 0.546745, 0.209674), Color(0.901823, 0.526127, 0.20575), Color(0.902117, 0.505509, 0.201825), Color(0.902412, 0.484891, 0.197901), Color(0.902706, 0.464273, 0.193976), Color(0.900873, 0.441104, 0.189491), Color(0.896914, 0.415383, 0.184446), Color(0.892955, 0.389662, 0.179401), Color(0.888995, 0.363941, 0.174356), Color(0.885036, 0.33822, 0.16931), Color(0.881077, 0.312499, 0.164265), Color(0.877277, 0.286724, 0.159347), Color(0.873957, 0.260788, 0.15481), Color(0.870638, 0.234851, 0.150274), Color(0.867318, 0.208915, 0.145737), Color(0.863998, 0.182979, 0.141201), Color(0.860679, 0.157042, 0.136664), Color(0.857359, 0.131106, 0.132128) };
int rainbow_colormap_length = 101;

Color temperature_colormap[] = { Color(0.178927, 0.305394, 0.933501), Color(0.194505, 0.321768, 0.934388), Color(0.210084, 0.338142, 0.935275), Color(0.225662, 0.354515, 0.936162), Color(0.24124, 0.370889, 0.93705), Color(0.256818, 0.387263, 0.937937), Color(0.272397, 0.403637, 0.938824), Color(0.287975, 0.42001, 0.939711), Color(0.303553, 0.436384, 0.940598), Color(0.320312, 0.45186, 0.941631), Color(0.33766, 0.466886, 0.942736), Color(0.355009, 0.481913, 0.943842), Color(0.372358, 0.496939, 0.944947), Color(0.389706, 0.511966, 0.946053), Color(0.407055, 0.526992, 0.947158), Color(0.424404, 0.542019, 0.948264), Color(0.441752, 0.557045, 0.949369), Color(0.46088, 0.573202, 0.950701), Color(0.483565, 0.591619, 0.952487), Color(0.506249, 0.610035, 0.954273), Color(0.528934, 0.628452, 0.956059), Color(0.551619, 0.646868, 0.957845), Color(0.574304, 0.665285, 0.95963), Color(0.596989, 0.683702, 0.961416), Color(0.619674, 0.702118, 0.963202), Color(0.642359, 0.720535, 0.964988), Color(0.663674, 0.737186, 0.967112), Color(0.684989, 0.753838, 0.969237), Color(0.706304, 0.770489, 0.971361), Color(0.727619, 0.787141, 0.973486), Color(0.748934, 0.803792, 0.97561), Color(0.770249, 0.820444, 0.977735), Color(0.791564, 0.837095, 0.979859), Color(0.812879, 0.853747, 0.981984), Color(0.829241, 0.866678, 0.983575), Color(0.843127, 0.877751, 0.984899), Color(0.857013, 0.888823, 0.986224), Color(0.870899, 0.899895, 0.987548), Color(0.884784, 0.910967, 0.988873), Color(0.89867, 0.922039, 0.990197), Color(0.912556, 0.933111, 0.991522), Color(0.926442, 0.944184, 0.992846), Color(0.937639, 0.953012, 0.990446), Color(0.943458, 0.957352, 0.980595), Color(0.949277, 0.961691, 0.970745), Color(0.955096, 0.966031, 0.960895), Color(0.960915, 0.970371, 0.951044), Color(0.966735, 0.974711, 0.941194), Color(0.972554, 0.979051, 0.931344), Color(0.978373, 0.983391, 0.921493), Color(0.984192, 0.987731, 0.911643), Color(0.985523, 0.988281, 0.889588), Color(0.986854, 0.988832, 0.867533), Color(0.988184, 0.989382, 0.845479), Color(0.989515, 0.989932, 0.823424), Color(0.990846, 0.990483, 0.801369), Color(0.992177, 0.991033, 0.779314), Color(0.993508, 0.991583, 0.757259), Color(0.994838, 0.992134, 0.735205), Color(0.99506, 0.991841, 0.703655), Color(0.994726, 0.991128, 0.667358), Color(0.994393, 0.990415, 0.63106), Color(0.994059, 0.989702, 0.594763), Color(0.993726, 0.988988, 0.558466), Color(0.993392, 0.988275, 0.522169), Color(0.993059, 0.987562, 0.485871), Color(0.992725, 0.986849, 0.449574), Color(0.991041, 0.981443, 0.419698), Color(0.986657, 0.966652, 0.402664), Color(0.982272, 0.951861, 0.38563), Color(0.977887, 0.93707, 0.368596), Color(0.973502, 0.922279, 0.351561), Color(0.969117, 0.907488, 0.334527), Color(0.964733, 0.892697, 0.317493), Color(0.960348, 0.877906, 0.300459), Color(0.955963, 0.863115, 0.283425), Color(0.949755, 0.838501, 0.27843), Color(0.943546, 0.813887, 0.273434), Color(0.937338, 0.789273, 0.268439), Color(0.93113, 0.764659, 0.263444), Color(0.924921, 0.740045, 0.258448), Color(0.918713, 0.715431, 0.253453), Color(0.912505, 0.690818, 0.248457), Color(0.906296, 0.666204, 0.243462), Color(0.900561, 0.641354, 0.238738), Color(0.895063, 0.616386, 0.23415), Color(0.889564, 0.591418, 0.229562), Color(0.884065, 0.56645, 0.224974), Color(0.878567, 0.541481, 0.220385), Color(0.873068, 0.516513, 0.215797), Color(0.867569, 0.491545, 0.211209), Color(0.862071, 0.466577, 0.206621), Color(0.856762, 0.4373, 0.201988), Color(0.851831, 0.399403, 0.197267), Color(0.846901, 0.361507, 0.192546), Color(0.841971, 0.32361, 0.187824), Color(0.83704, 0.285713, 0.183103), Color(0.83211, 0.247817, 0.178382), Color(0.82718, 0.20992, 0.173661), Color(0.822249, 0.172024, 0.168939), Color(0.817319, 0.134127, 0.164218) };
int temperature_colormap_length = 101;

Color gist_ncer_colormap[] = { Color(0.041600, 0.000000, 0.000000, 1.000000), Color(0.062190, 0.000000, 0.000000, 1.000000), Color(0.093074, 0.000000, 0.000000, 1.000000), Color(0.113664, 0.000000, 0.000000, 1.000000), Color(0.144548, 0.000000, 0.000000, 1.000000), Color(0.165138, 0.000000, 0.000000, 1.000000), Color(0.196023, 0.000000, 0.000000, 1.000000), Color(0.216612, 0.000000, 0.000000, 1.000000), Color(0.247497, 0.000000, 0.000000, 1.000000), Color(0.268087, 0.000000, 0.000000, 1.000000), Color(0.298971, 0.000000, 0.000000, 1.000000), Color(0.319561, 0.000000, 0.000000, 1.000000), Color(0.350445, 0.000000, 0.000000, 1.000000), Color(0.371035, 0.000000, 0.000000, 1.000000), Color(0.401920, 0.000000, 0.000000, 1.000000), Color(0.432804, 0.000000, 0.000000, 1.000000), Color(0.453394, 0.000000, 0.000000, 1.000000), Color(0.484278, 0.000000, 0.000000, 1.000000), Color(0.504868, 0.000000, 0.000000, 1.000000), Color(0.535753, 0.000000, 0.000000, 1.000000), Color(0.556342, 0.000000, 0.000000, 1.000000), Color(0.587227, 0.000000, 0.000000, 1.000000), Color(0.607816, 0.000000, 0.000000, 1.000000), Color(0.638701, 0.000000, 0.000000, 1.000000), Color(0.659291, 0.000000, 0.000000, 1.000000), Color(0.690175, 0.000000, 0.000000, 1.000000), Color(0.710765, 0.000000, 0.000000, 1.000000), Color(0.741649, 0.000000, 0.000000, 1.000000), Color(0.762239, 0.000000, 0.000000, 1.000000), Color(0.793124, 0.000000, 0.000000, 1.000000), Color(0.824008, 0.000000, 0.000000, 1.000000), Color(0.844598, 0.000000, 0.000000, 1.000000), Color(0.875482, 0.000000, 0.000000, 1.000000), Color(0.896072, 0.000000, 0.000000, 1.000000), Color(0.926957, 0.000000, 0.000000, 1.000000), Color(0.947546, 0.000000, 0.000000, 1.000000), Color(0.978431, 0.000000, 0.000000, 1.000000), Color(0.999020, 0.000000, 0.000000, 1.000000), Color(1.000000, 0.029903, 0.000000, 1.000000), Color(1.000000, 0.050491, 0.000000, 1.000000), Color(1.000000, 0.081373, 0.000000, 1.000000), Color(1.000000, 0.101962, 0.000000, 1.000000), Color(1.000000, 0.132844, 0.000000, 1.000000), Color(1.000000, 0.153432, 0.000000, 1.000000), Color(1.000000, 0.184314, 0.000000, 1.000000), Color(1.000000, 0.215197, 0.000000, 1.000000), Color(1.000000, 0.235785, 0.000000, 1.000000), Color(1.000000, 0.266667, 0.000000, 1.000000), Color(1.000000, 0.287255, 0.000000, 1.000000), Color(1.000000, 0.318138, 0.000000, 1.000000), Color(1.000000, 0.338726, 0.000000, 1.000000), Color(1.000000, 0.369608, 0.000000, 1.000000), Color(1.000000, 0.390196, 0.000000, 1.000000), Color(1.000000, 0.421079, 0.000000, 1.000000), Color(1.000000, 0.441667, 0.000000, 1.000000), Color(1.000000, 0.472549, 0.000000, 1.000000), Color(1.000000, 0.493137, 0.000000, 1.000000), Color(1.000000, 0.524020, 0.000000, 1.000000), Color(1.000000, 0.554902, 0.000000, 1.000000), Color(1.000000, 0.575490, 0.000000, 1.000000), Color(1.000000, 0.606373, 0.000000, 1.000000), Color(1.000000, 0.626961, 0.000000, 1.000000), Color(1.000000, 0.657843, 0.000000, 1.000000), Color(1.000000, 0.678431, 0.000000, 1.000000), Color(1.000000, 0.709314, 0.000000, 1.000000), Color(1.000000, 0.729902, 0.000000, 1.000000), Color(1.000000, 0.760784, 0.000000, 1.000000), Color(1.000000, 0.781372, 0.000000, 1.000000), Color(1.000000, 0.812255, 0.000000, 1.000000), Color(1.000000, 0.832843, 0.000000, 1.000000), Color(1.000000, 0.863725, 0.000000, 1.000000), Color(1.000000, 0.884313, 0.000000, 1.000000), Color(1.000000, 0.915196, 0.000000, 1.000000), Color(1.000000, 0.946078, 0.000000, 1.000000), Color(1.000000, 0.966666, 0.000000, 1.000000), Color(1.000000, 0.997548, 0.000000, 1.000000), Color(1.000000, 1.000000, 0.027205, 1.000000), Color(1.000000, 1.000000, 0.073528, 1.000000), Color(1.000000, 1.000000, 0.104411, 1.000000), Color(1.000000, 1.000000, 0.150734, 1.000000), Color(1.000000, 1.000000, 0.181617, 1.000000), Color(1.000000, 1.000000, 0.227940, 1.000000), Color(1.000000, 1.000000, 0.258823, 1.000000), Color(1.000000, 1.000000, 0.305146, 1.000000), Color(1.000000, 1.000000, 0.336029, 1.000000), Color(1.000000, 1.000000, 0.382352, 1.000000), Color(1.000000, 1.000000, 0.413235, 1.000000), Color(1.000000, 1.000000, 0.459558, 1.000000), Color(1.000000, 1.000000, 0.505882, 1.000000), Color(1.000000, 1.000000, 0.536764, 1.000000), Color(1.000000, 1.000000, 0.583088, 1.000000), Color(1.000000, 1.000000, 0.613970, 1.000000), Color(1.000000, 1.000000, 0.660294, 1.000000), Color(1.000000, 1.000000, 0.691176, 1.000000), Color(1.000000, 1.000000, 0.737500, 1.000000), Color(1.000000, 1.000000, 0.768382, 1.000000), Color(1.000000, 1.000000, 0.814706, 1.000000), Color(1.000000, 1.000000, 0.845588, 1.000000), Color(1.000000, 1.000000, 0.891912, 1.000000), Color(1.000000, 1.000000, 0.922794, 1.000000), Color(1.000000, 1.000000, 0.969118, 1.000000) };
int gist_ncer_colormap_length = 101;

inline CUDA_DEVICE __host__ int clamp(int value, int min, int max) {
    if(value < min) return min;
    if(value > max) return max;
    return value;
}

inline CUDA_DEVICE __host__ float clampf(float value, float min, float max) {
    if(value < min) return min;
    if(value > max) return max;
    return value;
}

CUDA_DEVICE __host__ Color tf_interpolate(Color* tf, float tf_min, float tf_max, int tf_size, float t) {
    float pos = (t - tf_min) / (tf_max - tf_min) * tf_size - 0.5f;
    int idx = floor(pos);
    idx = clamp(idx, 0, tf_size - 2);
    float diff = pos - idx;
    Color t0 = tf[idx];
    Color t1 = tf[idx + 1];
    return t0 * (1.0 - diff) + t1 * diff;
}

class TransferFunctionImpl : public TransferFunction {
public:

    TransferFunctionImpl() {
        metadata.size = 1600;
        metadata.input_min = 0;
        metadata.input_max = 20;
        metadata.is_log_scale = true;
        content_cpu = new Color[metadata.size];
        content_gpu = cudaAllocate<Color>(metadata.size);
        for(int i = 0; i < metadata.size; i++) {
            content_cpu[i] = Color(0, 0, 0, 0);
        }
        for(float i = 0; i <= metadata.input_max; i+=1) {
            float t = (float)i / metadata.input_max;
            Color c = tf_interpolate(rainbow_colormap, 0, 1, rainbow_colormap_length, t);
            c.a = pow(t, 0.5f);
            blendGaussian(i, 0.1, c);
        }

        cudaUpload<Color>(content_gpu, content_cpu, metadata.size);
    }

    void blendGaussian(float center, float sigma, Color value) {
        for(int i = 0; i < metadata.size; i++) {
            double t = (float)(i + 0.5f) / metadata.size;
            t = t * (metadata.input_max - metadata.input_min) + metadata.input_min;
            double gauss = exp(-(center - t) * (center - t) / sigma / sigma / 2.0f);
            Color v = Color(value.r, value.g, value.b, value.a * gauss);
            content_cpu[i] = v.blendTo(content_cpu[i]);
        }
    }

    virtual void setParameter(const char* name, void* value) {
    }

    virtual Metadata* getMetadata() {
        return &metadata;
    }
    virtual Color* getContent() {
        return content_cpu;
    }
    virtual Color* getContentGPU() {
        return content_gpu;
    }

    ~TransferFunctionImpl() {
        delete [] content_cpu;
        cudaDeallocate(content_gpu);
    }

    Metadata metadata;
    Color* content_cpu;
    Color* content_gpu;
};

TransferFunction* TransferFunction::CreateTest() {
    return new TransferFunctionImpl();
}

template<typename T>
struct MirroredMemory {
    T* cpu;
    T* gpu;
    size_t size, capacity;
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
            capacity = count * 2;
            delete [] cpu;
            cudaDeallocate(gpu);
            cpu = new T[capacity];
            gpu = cudaAllocate<T>(capacity);
        }
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
        delete [] cpu;
        cudaDeallocate(gpu);
    }
};

struct ray_marching_parameters_t {
    Lens::Ray* rays;
    Color* pixels;
    Color* tf;
    float tf_min, tf_max;
    bool tf_is_log;
    int tf_size;
    BlockDescription* blocks;
    float* data;
    int pixel_count;
    int block_count;
    float step_size;
};

inline CUDA_DEVICE float fminf(float a, float b) { return min(a, b); }
inline CUDA_DEVICE float fmaxf(float a, float b) { return max(a, b); }
inline CUDA_DEVICE Vector fminf(Vector a, Vector b) { return Vector(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
inline CUDA_DEVICE Vector fmaxf(Vector a, Vector b) { return Vector(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
inline CUDA_DEVICE Vector vmul(Vector a, Vector b) { return Vector(a.x * b.x, a.y * b.y, a.z * b.z); }

inline CUDA_DEVICE
int intersectBox(Vector origin, Vector direction, Vector boxmin, Vector boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    Vector invR = Vector(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
    Vector tbot = vmul(invR, boxmin - origin);
    Vector ttop = vmul(invR, boxmax - origin);

    // re-order intersections to find smallest and largest on each axis
    Vector tmin = fminf(ttop, tbot);
    Vector tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

inline CUDA_DEVICE float intersect_bbox(Vector p, Vector direction, Vector bmin, Vector bmax) {
    // if inside, no intersection.
    if(p <= bmax && p >= bmin) return 0;
    // Test for x.
    if(p.x < bmin.x && direction.x > 0) {
        float t = (bmin.x - p.x) / direction.x;
        Vector r = p + direction * t;
        if(r.y >= bmin.y && r.y <= bmax.y && r.z >= bmin.z && r.z <= bmax.z) {
            // there's only one possible, if we find it, return.
            return t;
        }
    }
    if(p.x > bmax.x && direction.x < 0) {
        float t = (bmax.x - p.x) / direction.x;
        Vector r = p + direction * t;
        if(r.y >= bmin.y && r.y <= bmax.y && r.z >= bmin.z && r.z <= bmax.z) {
            return t;
        }
    }
    if(p.y < bmin.y && direction.y > 0) {
        float t = (bmin.y - p.y) / direction.y;
        Vector r = p + direction * t;
        if(r.x >= bmin.x && r.x <= bmax.x && r.z >= bmin.z && r.z <= bmax.z) {
            return t;
        }
    }
    if(p.y > bmax.y && direction.y < 0) {
        float t = (bmax.y - p.y) / direction.y;
        Vector r = p + direction * t;
        if(r.x >= bmin.x && r.x <= bmax.x && r.z >= bmin.z && r.z <= bmax.z) {
            return t;
        }
    }
    if(p.z < bmin.z && direction.z > 0) {
        float t = (bmin.z - p.z) / direction.z;
        Vector r = p + direction * t;
        if(r.x >= bmin.x && r.x <= bmax.x && r.y >= bmin.y && r.y <= bmax.y) {
            return t;
        }
    }
    if(p.z > bmax.z && direction.z < 0) {
        float t = (bmax.z - p.z) / direction.z;
        Vector r = p + direction * t;
        if(r.x >= bmin.x && r.x <= bmax.x && r.y >= bmin.y && r.y <= bmax.y) {
            return t;
        }
    }
    // no intersection.
    return -1;
}

// inline CUDA_DEVICE float intersect_bbox_outpoint(Vector p, Vector direction, Vector bmin, Vector bmax) {

// }


inline CUDA_DEVICE float access_volume(float* data, int xsize, int ysize, int zsize, int ix, int iy, int iz) {
    //return data[ix * ysize * zsize + iy * zsize + iz];
    return data[iz * xsize * ysize + iy * xsize + ix];
}


CUDA_DEVICE float block_interploate(Vector pos, Vector min, Vector max, float* data, int xsize, int ysize, int zsize) {
    // [ 0 | 1 | 2 | 3 ]
    Vector p(
        (pos.x - min.x) / (max.x - min.x) * xsize - 0.5f,
        (pos.y - min.y) / (max.y - min.y) * ysize - 0.5f,
        (pos.z - min.z) / (max.z - min.z) * zsize - 0.5f
    );
    int ix = p.x, iy = p.y, iz = p.z;
    ix = clamp(ix, 0, xsize - 2);
    iy = clamp(iy, 0, ysize - 2);
    iz = clamp(iz, 0, zsize - 2);
    float tx = p.x - ix, ty = p.y - iy, tz = p.z - iz;
    float t000 = access_volume(data, xsize, ysize, zsize, ix, iy, iz);
    float t001 = access_volume(data, xsize, ysize, zsize, ix, iy, iz + 1);
    float t010 = access_volume(data, xsize, ysize, zsize, ix, iy + 1, iz);
    float t011 = access_volume(data, xsize, ysize, zsize, ix, iy + 1, iz + 1);
    float t100 = access_volume(data, xsize, ysize, zsize, ix + 1, iy, iz);
    float t101 = access_volume(data, xsize, ysize, zsize, ix + 1, iy, iz + 1);
    float t110 = access_volume(data, xsize, ysize, zsize, ix + 1, iy + 1, iz);
    float t111 = access_volume(data, xsize, ysize, zsize, ix + 1, iy + 1, iz + 1);
    float t00 = t000 * (1.0f - tz) + t001 * tz;
    float t01 = t010 * (1.0f - tz) + t011 * tz;
    float t10 = t100 * (1.0f - tz) + t101 * tz;
    float t11 = t110 * (1.0f - tz) + t111 * tz;
    float t0 = t00 * (1.0f - ty) + t01 * ty;
    float t1 = t10 * (1.0f - ty) + t11 * ty;
    float t = t0 * (1.0 - tx) + t1 * tx;
    return t;
}

CUDA_KERNEL void ray_marching_kernel(ray_marching_parameters_t p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= p.pixel_count) return;
    Lens::Ray ray = p.rays[idx];
    Vector pos = ray.origin;
    Vector d = ray.direction;
    int block_cursor = 0;
    Color color(0, 0, 0, 0);
    // Simple solution: fixed step size.
    float kmax = 1e40;
    while(block_cursor < p.block_count) {
        BlockDescription block = p.blocks[block_cursor];
        float kin, kout;
        if(intersectBox(pos, d, block.min, block.max, &kin, &kout) && kout >= 0) {
            if(kin < 0) kin = 0;
            if(kout > kmax) kout = kmax;
            if(kin < kout) {
                // Render this block.
                float distance = kout - kin;
                float voxel_size = (block.max.x - block.min.x) / block.xsize;
                int steps = ceil(distance / voxel_size * 3);
                float step_size = distance / steps;

                for(int i = steps - 1; i >= 0; i--) {
                    float k = kin + step_size * ((float)i + 0.5f);
                    Vector pt = pos + d * k;

                    // pt /= 5e10;
                    // float value = (sin(pt.x) + sin(pt.y) + sin(pt.z)) / 3;
                    // float v = value;
                    // Color c = tf_interpolate(p.tf, -1, 1, p.tf_size, v);

                    float value = block_interploate(pt, block.min, block.max, p.data + block.offset, block.xsize, block.ysize, block.zsize);
                    float v;
                    if(p.tf_is_log) v = log(1 + value);
                    else v = value;
                    Color c = tf_interpolate(p.tf, p.tf_min, p.tf_max, p.tf_size, v);

                    // float v = sin(pt.x / 4e8) + sin(pt.y / 4e8) + sin(pt.z / 4e8);
                    // Color c = tf_interpolate(p.tf, -3, 3, p.tf_size, v);
                    // pt /= 1e9;
                    // float v = sin(pt.x / 4) + sin(pt.y / 4) + sin(pt.z / 4);
                    // v = v * v / 30;
                    //printf("c = %f %f %f %f\n", c.r, c.g, c.b, c.a);
                    color = c.blendToDifferential(color, step_size / 1e9);
                    //color = color + c * step_size / 1e9;
                }
            }
            //kmax = kin;
        }
        block_cursor += 1;
    }
    //printf("color = %f %f %f %f\n", color.r, color.g, color.b, color.a);
    p.pixels[idx] = color;
}

class VolumeRendererImpl : public VolumeRenderer {
public:

    VolumeRendererImpl() : blocks(500), data(500 * 32 * 32 * 32), rays(1000 * 1000) {
    }

    struct BlockCompare {
        BlockCompare(Vector center_) {
            center = center_;
        }
        bool operator () (const BlockDescription& a, const BlockDescription& b) {
            float d1 = ((a.min + a.max) / 2.0f - center).len2();
            float d2 = ((b.min + b.max) / 2.0f - center).len2();
            return d1 > d2;
        }

        Vector center;
    };

    virtual void setVolume(VolumeBlocks* volume_) {
        // Copy volume data.
        volume = volume_;
        data.allocate(volume->getDataSize());
        data.upload(volume->getData());

    }
    virtual void setTransferFunction(TransferFunction* tf_) {
        tf = tf_;
    }
    virtual void setLens(Lens* lens_) {
        lens = lens_;
    }
    virtual void setImage(Image* image_) {
        image = image_;
    }
    virtual void render() {
        // Prepare blocks.
        int block_count = volume->getBlockCount();
        blocks.allocate(block_count);
        for(int i = 0; i < block_count; i++) {
            blocks[i] = *volume->getBlockDescription(i);
        }
        // Sort blocks.
        BlockCompare block_compare(lens->getCenter());
        sort(blocks.cpu, blocks.cpu + block_count, block_compare);
        blocks.upload();

        // Prepare image.
        int pixel_count = image->getWidth() * image->getHeight();
        rays.allocate(pixel_count);
        lens->getRaysGPU(image->getWidth(), image->getHeight(), rays.gpu);

        int cuda_blocks = pixel_count / CUDA_MAX_THREADS;
        if(pixel_count % CUDA_MAX_THREADS != 0) cuda_blocks += 1;
        ray_marching_parameters_t pms;
        pms.rays = rays.gpu;
        pms.pixels = image->getPixelsGPU();
        pms.blocks = blocks.gpu;
        pms.data = data.gpu;
        pms.pixel_count = pixel_count;
        pms.block_count = block_count;
        pms.step_size = 0.01;
        pms.tf = tf->getContentGPU();
        pms.tf_min = tf->getMetadata()->input_min;
        pms.tf_max = tf->getMetadata()->input_max;
        pms.tf_is_log = tf->getMetadata()->is_log_scale;
        pms.tf_size = tf->getMetadata()->size;
    #ifndef CPU_EMULATE
        ray_marching_kernel<<<cuda_blocks, CUDA_MAX_THREADS>>>(pms);
    #else
        blockDim.x = CUDA_MAX_THREADS;
        for(int i = 0; i < cuda_blocks; i++) {
            for(int j = 0; j < CUDA_MAX_THREADS; j++) {
                blockIdx.x = i;
                threadIdx.x = j;
                ray_marching_kernel(pms);
            }
        }
    #endif
    }

    MirroredMemory<BlockDescription> blocks;
    MirroredMemory<float> data;
    MirroredMemory<Lens::Ray> rays;
    VolumeBlocks* volume;
    TransferFunction* tf;
    Lens* lens;
    Image* image;
};

VolumeRenderer* VolumeRenderer::CreateGPU() {
    return new VolumeRendererImpl();
}

}
