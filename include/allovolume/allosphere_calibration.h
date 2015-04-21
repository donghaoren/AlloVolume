#ifndef ALLOSPHERE_CALIBRATION_H_INCLUDED
#define ALLOSPHERE_CALIBRATION_H_INCLUDED

// Read allosphere calibration data.

#include "utils.h"
#include "renderer.h"

namespace allovolume {

    class AllosphereLens : public Lens {
    public:
        // Perform blending.
        virtual void performBlend(Image* img) = 0;
    };

    class AllosphereCalibration {
    public:
        struct Projection {
            Vector4* warpData;
            int warpWidth, warpHeight;

            float* blendData;
            int blendWidth, blendHeight;

            float viewport_x, viewport_y, viewport_w, viewport_h;
        };

        struct RenderSlave {
            int num_projections;
            Projection* projections;
        };

        virtual RenderSlave* getRenderer(const char* hostname = NULL) = 0;

        virtual ~AllosphereCalibration() { }

        static const char* getHostname();

        static AllosphereCalibration* Load(const char* basepath);
        static AllosphereLens* CreateLens(Projection* projection);
    };

}

#endif
