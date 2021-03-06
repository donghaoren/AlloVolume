#include <cuda_runtime.h>

#include "allovolume/renderer.h"
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#include <yaml-cpp/yaml.h>

using namespace std;

namespace allovolume {

namespace {
    Color rainbow_colormap[] = { Color(0.471412, 0.108766, 0.527016), Color(0.445756, 0.110176, 0.549008), Color(0.420099, 0.111586, 0.571001), Color(0.394443, 0.112997, 0.592993), Color(0.368787, 0.114407, 0.614986), Color(0.34313, 0.115817, 0.636978), Color(0.317474, 0.117227, 0.658971), Color(0.30382, 0.130517, 0.677031), Color(0.294167, 0.147766, 0.69378), Color(0.284514, 0.165015, 0.71053), Color(0.274861, 0.182264, 0.727279), Color(0.265208, 0.199513, 0.744028), Color(0.255555, 0.216762, 0.760777), Color(0.250196, 0.236254, 0.772907), Color(0.249132, 0.257991, 0.780416), Color(0.248069, 0.279728, 0.787925), Color(0.247005, 0.301465, 0.795434), Color(0.245941, 0.323202, 0.802943), Color(0.244878, 0.344939, 0.810452), Color(0.244962, 0.366259, 0.815542), Color(0.248488, 0.386326, 0.813373), Color(0.252015, 0.406394, 0.811204), Color(0.255542, 0.426461, 0.809035), Color(0.259069, 0.446529, 0.806867), Color(0.262595, 0.466596, 0.804698), Color(0.266122, 0.486664, 0.802529), Color(0.27249, 0.50249, 0.792471), Color(0.278857, 0.518316, 0.782413), Color(0.285225, 0.534141, 0.772355), Color(0.291592, 0.549967, 0.762297), Color(0.29796, 0.565793, 0.752239), Color(0.304327, 0.581619, 0.742181), Color(0.312466, 0.593997, 0.728389), Color(0.321196, 0.605227, 0.713353), Color(0.329926, 0.616456, 0.698317), Color(0.338656, 0.627685, 0.683282), Color(0.347385, 0.638915, 0.668246), Color(0.356115, 0.650144, 0.65321), Color(0.366029, 0.659446, 0.637262), Color(0.377127, 0.666821, 0.620403), Color(0.388225, 0.674195, 0.603544), Color(0.399323, 0.681569, 0.586684), Color(0.410421, 0.688944, 0.569825), Color(0.421519, 0.696318, 0.552966), Color(0.433185, 0.702972, 0.536335), Color(0.446557, 0.707463, 0.520393), Color(0.459929, 0.711955, 0.504451), Color(0.473301, 0.716446, 0.488509), Color(0.486673, 0.720937, 0.472566), Color(0.500045, 0.725429, 0.456624), Color(0.513417, 0.72992, 0.440682), Color(0.528494, 0.732128, 0.427547), Color(0.543572, 0.734335, 0.414412), Color(0.558649, 0.736543, 0.401277), Color(0.573727, 0.738751, 0.388142), Color(0.588804, 0.740958, 0.375007), Color(0.603882, 0.743166, 0.361872), Color(0.619337, 0.743583, 0.351457), Color(0.634919, 0.743402, 0.34195), Color(0.650501, 0.743222, 0.332443), Color(0.666083, 0.743042, 0.322935), Color(0.681665, 0.742861, 0.313428), Color(0.697247, 0.742681, 0.303921), Color(0.712199, 0.740876, 0.2961), Color(0.726521, 0.737447, 0.289966), Color(0.740842, 0.734018, 0.283832), Color(0.755164, 0.730589, 0.277698), Color(0.769486, 0.727159, 0.271564), Color(0.783808, 0.72373, 0.26543), Color(0.797308, 0.719143, 0.259858), Color(0.808342, 0.711081, 0.255976), Color(0.819376, 0.703019, 0.252094), Color(0.83041, 0.694957, 0.248211), Color(0.841444, 0.686895, 0.244329), Color(0.852478, 0.678833, 0.240446), Color(0.863512, 0.670771, 0.236564), Color(0.869512, 0.6567, 0.23336), Color(0.875513, 0.642629, 0.230157), Color(0.881513, 0.628557, 0.226953), Color(0.887513, 0.614486, 0.22375), Color(0.893514, 0.600415, 0.220546), Color(0.899514, 0.586344, 0.217343), Color(0.901235, 0.567363, 0.213599), Color(0.901529, 0.546745, 0.209674), Color(0.901823, 0.526127, 0.20575), Color(0.902117, 0.505509, 0.201825), Color(0.902412, 0.484891, 0.197901), Color(0.902706, 0.464273, 0.193976), Color(0.900873, 0.441104, 0.189491), Color(0.896914, 0.415383, 0.184446), Color(0.892955, 0.389662, 0.179401), Color(0.888995, 0.363941, 0.174356), Color(0.885036, 0.33822, 0.16931), Color(0.881077, 0.312499, 0.164265), Color(0.877277, 0.286724, 0.159347), Color(0.873957, 0.260788, 0.15481), Color(0.870638, 0.234851, 0.150274), Color(0.867318, 0.208915, 0.145737), Color(0.863998, 0.182979, 0.141201), Color(0.860679, 0.157042, 0.136664), Color(0.857359, 0.131106, 0.132128) };
    int rainbow_colormap_length = 101;

    Color temperature_colormap[] = { Color(0.178927, 0.305394, 0.933501), Color(0.194505, 0.321768, 0.934388), Color(0.210084, 0.338142, 0.935275), Color(0.225662, 0.354515, 0.936162), Color(0.24124, 0.370889, 0.93705), Color(0.256818, 0.387263, 0.937937), Color(0.272397, 0.403637, 0.938824), Color(0.287975, 0.42001, 0.939711), Color(0.303553, 0.436384, 0.940598), Color(0.320312, 0.45186, 0.941631), Color(0.33766, 0.466886, 0.942736), Color(0.355009, 0.481913, 0.943842), Color(0.372358, 0.496939, 0.944947), Color(0.389706, 0.511966, 0.946053), Color(0.407055, 0.526992, 0.947158), Color(0.424404, 0.542019, 0.948264), Color(0.441752, 0.557045, 0.949369), Color(0.46088, 0.573202, 0.950701), Color(0.483565, 0.591619, 0.952487), Color(0.506249, 0.610035, 0.954273), Color(0.528934, 0.628452, 0.956059), Color(0.551619, 0.646868, 0.957845), Color(0.574304, 0.665285, 0.95963), Color(0.596989, 0.683702, 0.961416), Color(0.619674, 0.702118, 0.963202), Color(0.642359, 0.720535, 0.964988), Color(0.663674, 0.737186, 0.967112), Color(0.684989, 0.753838, 0.969237), Color(0.706304, 0.770489, 0.971361), Color(0.727619, 0.787141, 0.973486), Color(0.748934, 0.803792, 0.97561), Color(0.770249, 0.820444, 0.977735), Color(0.791564, 0.837095, 0.979859), Color(0.812879, 0.853747, 0.981984), Color(0.829241, 0.866678, 0.983575), Color(0.843127, 0.877751, 0.984899), Color(0.857013, 0.888823, 0.986224), Color(0.870899, 0.899895, 0.987548), Color(0.884784, 0.910967, 0.988873), Color(0.89867, 0.922039, 0.990197), Color(0.912556, 0.933111, 0.991522), Color(0.926442, 0.944184, 0.992846), Color(0.937639, 0.953012, 0.990446), Color(0.943458, 0.957352, 0.980595), Color(0.949277, 0.961691, 0.970745), Color(0.955096, 0.966031, 0.960895), Color(0.960915, 0.970371, 0.951044), Color(0.966735, 0.974711, 0.941194), Color(0.972554, 0.979051, 0.931344), Color(0.978373, 0.983391, 0.921493), Color(0.984192, 0.987731, 0.911643), Color(0.985523, 0.988281, 0.889588), Color(0.986854, 0.988832, 0.867533), Color(0.988184, 0.989382, 0.845479), Color(0.989515, 0.989932, 0.823424), Color(0.990846, 0.990483, 0.801369), Color(0.992177, 0.991033, 0.779314), Color(0.993508, 0.991583, 0.757259), Color(0.994838, 0.992134, 0.735205), Color(0.99506, 0.991841, 0.703655), Color(0.994726, 0.991128, 0.667358), Color(0.994393, 0.990415, 0.63106), Color(0.994059, 0.989702, 0.594763), Color(0.993726, 0.988988, 0.558466), Color(0.993392, 0.988275, 0.522169), Color(0.993059, 0.987562, 0.485871), Color(0.992725, 0.986849, 0.449574), Color(0.991041, 0.981443, 0.419698), Color(0.986657, 0.966652, 0.402664), Color(0.982272, 0.951861, 0.38563), Color(0.977887, 0.93707, 0.368596), Color(0.973502, 0.922279, 0.351561), Color(0.969117, 0.907488, 0.334527), Color(0.964733, 0.892697, 0.317493), Color(0.960348, 0.877906, 0.300459), Color(0.955963, 0.863115, 0.283425), Color(0.949755, 0.838501, 0.27843), Color(0.943546, 0.813887, 0.273434), Color(0.937338, 0.789273, 0.268439), Color(0.93113, 0.764659, 0.263444), Color(0.924921, 0.740045, 0.258448), Color(0.918713, 0.715431, 0.253453), Color(0.912505, 0.690818, 0.248457), Color(0.906296, 0.666204, 0.243462), Color(0.900561, 0.641354, 0.238738), Color(0.895063, 0.616386, 0.23415), Color(0.889564, 0.591418, 0.229562), Color(0.884065, 0.56645, 0.224974), Color(0.878567, 0.541481, 0.220385), Color(0.873068, 0.516513, 0.215797), Color(0.867569, 0.491545, 0.211209), Color(0.862071, 0.466577, 0.206621), Color(0.856762, 0.4373, 0.201988), Color(0.851831, 0.399403, 0.197267), Color(0.846901, 0.361507, 0.192546), Color(0.841971, 0.32361, 0.187824), Color(0.83704, 0.285713, 0.183103), Color(0.83211, 0.247817, 0.178382), Color(0.82718, 0.20992, 0.173661), Color(0.822249, 0.172024, 0.168939), Color(0.817319, 0.134127, 0.164218) };
    int temperature_colormap_length = 101;

    Color gist_ncer_colormap[] = { Color(0.041600, 0.000000, 0.000000, 1.000000), Color(0.062190, 0.000000, 0.000000, 1.000000), Color(0.093074, 0.000000, 0.000000, 1.000000), Color(0.113664, 0.000000, 0.000000, 1.000000), Color(0.144548, 0.000000, 0.000000, 1.000000), Color(0.165138, 0.000000, 0.000000, 1.000000), Color(0.196023, 0.000000, 0.000000, 1.000000), Color(0.216612, 0.000000, 0.000000, 1.000000), Color(0.247497, 0.000000, 0.000000, 1.000000), Color(0.268087, 0.000000, 0.000000, 1.000000), Color(0.298971, 0.000000, 0.000000, 1.000000), Color(0.319561, 0.000000, 0.000000, 1.000000), Color(0.350445, 0.000000, 0.000000, 1.000000), Color(0.371035, 0.000000, 0.000000, 1.000000), Color(0.401920, 0.000000, 0.000000, 1.000000), Color(0.432804, 0.000000, 0.000000, 1.000000), Color(0.453394, 0.000000, 0.000000, 1.000000), Color(0.484278, 0.000000, 0.000000, 1.000000), Color(0.504868, 0.000000, 0.000000, 1.000000), Color(0.535753, 0.000000, 0.000000, 1.000000), Color(0.556342, 0.000000, 0.000000, 1.000000), Color(0.587227, 0.000000, 0.000000, 1.000000), Color(0.607816, 0.000000, 0.000000, 1.000000), Color(0.638701, 0.000000, 0.000000, 1.000000), Color(0.659291, 0.000000, 0.000000, 1.000000), Color(0.690175, 0.000000, 0.000000, 1.000000), Color(0.710765, 0.000000, 0.000000, 1.000000), Color(0.741649, 0.000000, 0.000000, 1.000000), Color(0.762239, 0.000000, 0.000000, 1.000000), Color(0.793124, 0.000000, 0.000000, 1.000000), Color(0.824008, 0.000000, 0.000000, 1.000000), Color(0.844598, 0.000000, 0.000000, 1.000000), Color(0.875482, 0.000000, 0.000000, 1.000000), Color(0.896072, 0.000000, 0.000000, 1.000000), Color(0.926957, 0.000000, 0.000000, 1.000000), Color(0.947546, 0.000000, 0.000000, 1.000000), Color(0.978431, 0.000000, 0.000000, 1.000000), Color(0.999020, 0.000000, 0.000000, 1.000000), Color(1.000000, 0.029903, 0.000000, 1.000000), Color(1.000000, 0.050491, 0.000000, 1.000000), Color(1.000000, 0.081373, 0.000000, 1.000000), Color(1.000000, 0.101962, 0.000000, 1.000000), Color(1.000000, 0.132844, 0.000000, 1.000000), Color(1.000000, 0.153432, 0.000000, 1.000000), Color(1.000000, 0.184314, 0.000000, 1.000000), Color(1.000000, 0.215197, 0.000000, 1.000000), Color(1.000000, 0.235785, 0.000000, 1.000000), Color(1.000000, 0.266667, 0.000000, 1.000000), Color(1.000000, 0.287255, 0.000000, 1.000000), Color(1.000000, 0.318138, 0.000000, 1.000000), Color(1.000000, 0.338726, 0.000000, 1.000000), Color(1.000000, 0.369608, 0.000000, 1.000000), Color(1.000000, 0.390196, 0.000000, 1.000000), Color(1.000000, 0.421079, 0.000000, 1.000000), Color(1.000000, 0.441667, 0.000000, 1.000000), Color(1.000000, 0.472549, 0.000000, 1.000000), Color(1.000000, 0.493137, 0.000000, 1.000000), Color(1.000000, 0.524020, 0.000000, 1.000000), Color(1.000000, 0.554902, 0.000000, 1.000000), Color(1.000000, 0.575490, 0.000000, 1.000000), Color(1.000000, 0.606373, 0.000000, 1.000000), Color(1.000000, 0.626961, 0.000000, 1.000000), Color(1.000000, 0.657843, 0.000000, 1.000000), Color(1.000000, 0.678431, 0.000000, 1.000000), Color(1.000000, 0.709314, 0.000000, 1.000000), Color(1.000000, 0.729902, 0.000000, 1.000000), Color(1.000000, 0.760784, 0.000000, 1.000000), Color(1.000000, 0.781372, 0.000000, 1.000000), Color(1.000000, 0.812255, 0.000000, 1.000000), Color(1.000000, 0.832843, 0.000000, 1.000000), Color(1.000000, 0.863725, 0.000000, 1.000000), Color(1.000000, 0.884313, 0.000000, 1.000000), Color(1.000000, 0.915196, 0.000000, 1.000000), Color(1.000000, 0.946078, 0.000000, 1.000000), Color(1.000000, 0.966666, 0.000000, 1.000000), Color(1.000000, 0.997548, 0.000000, 1.000000), Color(1.000000, 1.000000, 0.027205, 1.000000), Color(1.000000, 1.000000, 0.073528, 1.000000), Color(1.000000, 1.000000, 0.104411, 1.000000), Color(1.000000, 1.000000, 0.150734, 1.000000), Color(1.000000, 1.000000, 0.181617, 1.000000), Color(1.000000, 1.000000, 0.227940, 1.000000), Color(1.000000, 1.000000, 0.258823, 1.000000), Color(1.000000, 1.000000, 0.305146, 1.000000), Color(1.000000, 1.000000, 0.336029, 1.000000), Color(1.000000, 1.000000, 0.382352, 1.000000), Color(1.000000, 1.000000, 0.413235, 1.000000), Color(1.000000, 1.000000, 0.459558, 1.000000), Color(1.000000, 1.000000, 0.505882, 1.000000), Color(1.000000, 1.000000, 0.536764, 1.000000), Color(1.000000, 1.000000, 0.583088, 1.000000), Color(1.000000, 1.000000, 0.613970, 1.000000), Color(1.000000, 1.000000, 0.660294, 1.000000), Color(1.000000, 1.000000, 0.691176, 1.000000), Color(1.000000, 1.000000, 0.737500, 1.000000), Color(1.000000, 1.000000, 0.768382, 1.000000), Color(1.000000, 1.000000, 0.814706, 1.000000), Color(1.000000, 1.000000, 0.845588, 1.000000), Color(1.000000, 1.000000, 0.891912, 1.000000), Color(1.000000, 1.000000, 0.922794, 1.000000), Color(1.000000, 1.000000, 0.969118, 1.000000) };
    int gist_ncer_colormap_length = 101;

    inline int clampi(int value, int min, int max) {
        if(value < min) return min;
        if(value > max) return max;
        return value;
    }

    inline float clamp01f_host(float value) {
        if(value < 0) return 0;
        if(value > 1) return 1;
        return value;
    }

    inline Color tf_interpolate_host(Color* tf, float tf_min, float tf_max, int tf_size, float t) {
        float pos = clamp01f_host((t - tf_min) / (tf_max - tf_min)) * (tf_size - 1.0f);
        int idx = floor(pos);
        idx = clampi(idx, 0, tf_size - 2);
        float diff = pos - idx;
        Color t0 = tf[idx];
        Color t1 = tf[idx + 1];
        return t0 * (1.0 - diff) + t1 * diff;
    }
}

class TransferFunctionImpl : public TransferFunction {
public:

    TransferFunctionImpl() {
        content_cpu = NULL;
        capacity = 0;

        domain_min = 0;
        domain_max = 1;
        scale = kLinearScale;

        allocate(1600);
        size = 1600;
        for(int i = 0; i < size; i++) {
            content_cpu[i] = Color(0, 0, 0, 0);
        }
    }

    void addGaussianTicks(int ticks) {
        for(float i = 0; i < ticks; i++) {
            float t = (float)i / ((float)ticks - 1);
            Color c = tf_interpolate_host(rainbow_colormap, 0, 1, rainbow_colormap_length, t);
            c.a = t * t * 0.9999;
            blendGaussian(t, 1.0 / ticks / 10.0f, c);
        }
    }

    void addLinearGradient() {
        for(int i = 0; i < size; i++) {
            double t = (float)(i) / (size - 1);
            Color c = tf_interpolate_host(rainbow_colormap, 0, 1, rainbow_colormap_length, t);
            c.a = t;
            content_cpu[i] = c.blendToCorrected(content_cpu[i]);
        }
    }

    void blendGaussian(float center, float sigma, Color value) {
        for(int i = 0; i < size; i++) {
            double t = (float)(i) / (size - 1);
            double gauss = exp(-(center - t) * (center - t) / sigma / sigma / 2.0f);
            Color v = Color(value.r, value.g, value.b, value.a * gauss);
            content_cpu[i] = v.blendToCorrected(content_cpu[i]);
        }
    }

    virtual Color* getContent() {
        return content_cpu;
    }

    virtual void setContent(const Color* color, size_t size_) {
        allocate(size_);
        size = size_;
        memcpy(content_cpu, color, sizeof(Color) * size);
    }

    virtual void getDomain(float& min, float& max) {
        min = domain_min;
        max = domain_max;
    }

    virtual Scale getScale() {
        return scale;
    }

    virtual size_t getSize() {
        return size;
    }

    virtual void setDomain(float min, float max) {
        domain_min = min;
        domain_max = max;
    }

    virtual void setScale(Scale scale_) {
        scale = scale_;
    }

    ~TransferFunctionImpl() {
        free();
    }

private:
    void free() {
        if(content_cpu) delete [] content_cpu;
        content_cpu = NULL;
        capacity = 0;
    }
    void allocate(size_t capacity_) {
        if(capacity >= capacity_) return;
        free();
        capacity = capacity_;
        content_cpu = new Color[capacity];
    }

    float domain_min, domain_max;
    Scale scale;

    Color* content_cpu;
    size_t size, capacity;
};

TransferFunction* TransferFunction::CreateTransparent(float min, float max, Scale scale, size_t size) {
    TransferFunctionImpl* r = new TransferFunctionImpl();
    r->setDomain(min, max);
    r->setScale(scale);
    Color* transparent = new Color[size];
    for(int i = 0; i < size; i++) transparent[i] = Color(0, 0, 0, 0);
    r->setContent(transparent, size);
    delete [] transparent;
    return r;
}

TransferFunction* TransferFunction::CreateGaussianTicks(float min, float max, Scale scale, int ticks) {
    TransferFunctionImpl* r = new TransferFunctionImpl();
    r->setDomain(min, max);
    r->setScale(scale);
    r->addGaussianTicks(ticks);
    return r;
}

TransferFunction* TransferFunction::CreateLinearGradient(float min, float max, Scale scale) {
    TransferFunctionImpl* r = new TransferFunctionImpl();
    r->setDomain(min, max);
    r->setScale(scale);
    r->addLinearGradient();
    return r;
}

Color sample_gradient(vector<Color>& colors, float t) {
    float pos = t * colors.size();
    int i = floor(pos);
    if(i < 0) i = 0;
    if(i >= colors.size() - 1) i = colors.size() - 2;
    float k = pos - i;
    return colors[i] * (1.0f - k) + colors[i + 1] * k;
}

float sigmoid(float t) {
    return 1.0 / (1.0 + exp(-t));
}

void TransferFunction::ParseLayers(TransferFunction* target, size_t size, const char* layers) {
    Color* data = new Color[size];
    YAML::Node node = YAML::Load(layers);

    for(int i = 0; i < size; i++) {
        data[i] = Color(0, 0, 0, 0);
    }

    for(YAML::Node::iterator it = node.begin(); it != node.end(); ++it) {
        YAML::Node layer = *it;
        string type = layer["t"].as<string>();
        if(type == "gaussians") {
            int ticks = layer["ticks"].as<int>();
            float t0 = layer["t0"].as<float>();
            float t1 = layer["t1"].as<float>();
            float alpha0 = layer["alpha0"].as<float>();
            float alpha1 = layer["alpha1"].as<float>();
            float alpha_pow = layer["alpha_pow"].as<float>();
            float sigma0 = layer["sigma0"].as<float>();
            float sigma1 = layer["sigma1"].as<float>();
            vector<Color> gradient;
            for(YAML::Node::iterator gradient_it = layer["gradient"].begin(); gradient_it != layer["gradient"].end(); ++gradient_it) {
                gradient.push_back(Color((*gradient_it)[0].as<float>(), (*gradient_it)[1].as<float>(), (*gradient_it)[2].as<float>(), 1.0f));
            }
            for(int tick = 0; tick < ticks; ++tick) {
                float tick_t = (float)tick / (float)(ticks - 1);
                float t_center = tick_t * (t1 - t0) + t0;
                float alpha = pow(tick_t, alpha_pow) * (alpha1 - alpha0) + alpha0;
                float sigma = tick_t * (sigma1 - sigma0) + sigma0;
                sigma = sigma * (t1 - t0) / ticks;
                Color c = sample_gradient(gradient, tick_t);

                for(int i = 0; i < size; i++) {
                    float t = ((float)i + 0.5f) / (float)size;
                    float gaussian = exp(- (t - t_center) / sigma * (t - t_center) / sigma / 2) * alpha;
                    c.a = gaussian;
                    data[i] = c.blendToCorrected(data[i]);
                }
            }
        }
        if(type == "gradient") {
            float t0 = layer["t0"].as<float>();
            float t1 = layer["t1"].as<float>();
            float alpha0 = layer["alpha0"].as<float>();
            float alpha1 = layer["alpha1"].as<float>();
            float alpha_pow = layer["alpha_pow"].as<float>();
            vector<Color> gradient;
            for(YAML::Node::iterator gradient_it = layer["gradient"].begin(); gradient_it != layer["gradient"].end(); ++gradient_it) {
                gradient.push_back(Color((*gradient_it)[0].as<float>(), (*gradient_it)[1].as<float>(), (*gradient_it)[2].as<float>(), 1.0f));
            }
            for(int i = 0; i < size; i++) {
                float t = ((float)i + 0.5f) / (float)size;
                if(t0 < t1 && (t < t0 || t > t1)) continue;
                if(t0 > t1 && (t > t0 || t < t1)) continue;
                float gradient_t = (t - t0) / (t1 - t0);
                float alpha = pow(gradient_t , alpha_pow) * (alpha1 - alpha0) + alpha0;
                Color c = sample_gradient(gradient, gradient_t);
                c.a = alpha;
                data[i] = c.blendToCorrected(data[i]);
            }
        }
        if(type == "block") {
            float tm = layer["tm"].as<float>();
            float span = layer["span"].as<float>();
            float feather = layer["feather"].as<float>();
            YAML::Node color = layer["color"];
            Color c0(color[0].as<float>(), color[1].as<float>(), color[2].as<float>(), color[3].as<float>());
            for(int i = 0; i < size; i++) {
                float t = ((float)i + 0.5f) / (float)size;
                Color c = c0;
                c.a *= (sigmoid((t - (tm - span)) / feather) + sigmoid((tm + span - t) / feather) - 1.0);
                data[i] = c.blendToCorrected(data[i]);
            }
        }
    }

    target->setContent(data, size);
    delete [] data;
}

}
