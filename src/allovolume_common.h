#ifndef ALLOVOLUME_COMMON_H_INCLUDED
#define ALLOVOLUME_COMMON_H_INCLUDED

#include "dataset.h"
#include "renderer.h"
#include "allosphere/allosphere_calibration.h"
#include <boost/shared_ptr.hpp>

// The renderer has 4 state varaibles:
//  volume: The volume to be rendered.
//  pose: The pose of the viewer. (x, y, z, qx, qy, qz, qw)
//  lens: The information for the lens. (eye_separation, focal_distance)
//  transfer_function: The transfer function to use.
//  RGB curve: The rgb curve for final output.
// Volume is set only, others are get/set.

const int kRGBCurveSize = 256;

struct RGBCurve {
    float data[kRGBCurveSize];
};

struct AlloVolumeState {
    boost::shared_ptr<allovolume::VolumeBlocks> volume;
    boost::shared_ptr<allovolume::TransferFunction> transfer_function;
    RGBCurve curve;
    allovolume::Pose pose;
};

#endif
