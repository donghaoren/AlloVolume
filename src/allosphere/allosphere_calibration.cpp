#include "allovolume/allosphere_calibration.h"
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>
#include <stdio.h>
#include <FreeImage.h>
#include <unistd.h>

using namespace std;

namespace allovolume {

class AllosphereCalibrationImpl : public AllosphereCalibration {
public:
    AllosphereCalibrationImpl(const char* basepath_) {
        basepath = basepath_;

        dummy_slave.projections = &dummy_projection;
        dummy_slave.num_projections = 1;
        dummy_projection.viewport_x = 0;
        dummy_projection.viewport_y = 0;
        dummy_projection.viewport_w = 1;
        dummy_projection.viewport_h = 1;
        dummy_projection.warpData = NULL;
        dummy_projection.blendData = NULL;
    }

    void loadProjection(const char* hostname) {
        if(renderers.find(hostname) != renderers.end()) return;
        YAML::Node conf = YAML::LoadFile(basepath + "/" + hostname + ".json");
        RenderSlave& info = renderers[hostname];
        info.num_projections = conf["projections"].size();
        info.projections = new Projection[info.num_projections];
        for(int p = 0; p < info.num_projections; p++) {
            YAML::Node proj = conf["projections"][p];
            Projection& proj_out = info.projections[p];
            // Viewport information.
            proj_out.viewport_x = proj["viewport"]["l"].as<float>();
            proj_out.viewport_y = proj["viewport"]["b"].as<float>();
            proj_out.viewport_w = proj["viewport"]["w"].as<float>();
            proj_out.viewport_h = proj["viewport"]["h"].as<float>();

            // Read blend file.
            string blendFile = proj["blend"]["file"].as<string>();
            FIBITMAP *blendImage = FreeImage_Load(FIF_PNG, (basepath + "/" + blendFile).c_str(), PNG_DEFAULT);
            blendImage = FreeImage_ConvertTo24Bits(blendImage);
            proj_out.blendWidth = FreeImage_GetWidth(blendImage);
            proj_out.blendHeight = FreeImage_GetHeight(blendImage);
            proj_out.blendData = new float[proj_out.blendWidth * proj_out.blendHeight];
            for(int y = 0; y < proj_out.blendHeight; y++) {
                BYTE* scanline = FreeImage_GetScanLine(blendImage, y);
                float* out = proj_out.blendData + y * proj_out.blendWidth;
                for(int x = 0; x < proj_out.blendWidth; x++) {
                    out[x] = scanline[x * 3 + 0] / 255.0;
                }
            }

            // Read warp file.
            string warpFile = proj["warp"]["file"].as<string>();
            int32_t warpsize[2];
            FILE* fwarp = fopen((string(basepath) + "/" + warpFile).c_str(), "rb");
            fread(warpsize, sizeof(int32_t), 2, fwarp);
            proj_out.warpWidth = warpsize[1];
            proj_out.warpHeight = warpsize[0] / 3;
            proj_out.warpData = new Vector4[proj_out.warpWidth * proj_out.warpHeight];
            float* buf = new float[proj_out.warpWidth * proj_out.warpHeight];
            fread(buf, sizeof(float), proj_out.warpWidth * proj_out.warpHeight, fwarp); // x
            for(int j = 0; j < proj_out.warpWidth * proj_out.warpHeight; j++) {
                proj_out.warpData[j].x = buf[j];
            }
            fread(buf, sizeof(float), proj_out.warpWidth * proj_out.warpHeight, fwarp); // y
            for(int j = 0; j < proj_out.warpWidth * proj_out.warpHeight; j++) {
                proj_out.warpData[j].y = buf[j];
            }
            fread(buf, sizeof(float), proj_out.warpWidth * proj_out.warpHeight, fwarp); // z
            for(int j = 0; j < proj_out.warpWidth * proj_out.warpHeight; j++) {
                proj_out.warpData[j].z = buf[j];
                proj_out.warpData[j].w = 1.0f;
            }
            delete [] buf;
            fclose(fwarp);
        }
    }

    virtual ~AllosphereCalibrationImpl() {
        for(map<string, RenderSlave>::iterator it = renderers.begin(); it != renderers.end(); ++it) {
            for(int i = 0; i < it->second.num_projections; i++) {
                delete [] it->second.projections[i].warpData;
                delete [] it->second.projections[i].blendData;
            }
            delete [] it->second.projections;
        }
    }

    virtual RenderSlave* getRenderer(const char* hostname) {
        if(hostname) return &renderers[hostname];
        char myhostname[256];
        gethostname(myhostname, 256);
        try {
            loadProjection(myhostname);
            return &renderers[myhostname];
        } catch(...) {
            return &dummy_slave;
        }
    }

    map<string, RenderSlave> renderers;
    RenderSlave dummy_slave;
    Projection dummy_projection;
    string basepath;
};

AllosphereCalibration* AllosphereCalibration::Load(const char* basepath) {
    return new AllosphereCalibrationImpl(basepath);
}

}
