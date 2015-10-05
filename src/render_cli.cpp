// Command line interface for the renderer.

#include "allovolume/dataset.h"
#include "allovolume/renderer.h"

#include <stdio.h>
#include <string>
#include <fstream>
#include <map>
#include <yaml-cpp/yaml.h>

#include "allovolume_protocol.pb.h"

#include "timeprofiler.h"

using namespace std;
using namespace allovolume;

Vector node2vector(const YAML::Node& node) {
    YAML::Node::const_iterator it = node.begin();
    Vector r;
    r.x = it->as<float>(); it++;
    r.y = it->as<float>(); it++;
    r.z = it->as<float>(); it++;
    return r;
}

int main(int argc, char* argv[]) {

    TimeMeasure time_measure("Main");
    //TimeProfiler::Default()->setDelegate(TimeProfiler::STDERR_DELEGATE);

    YAML::Node parameters;
    if(argc >= 2) {
        parameters = YAML::LoadFile(argv[1]);
    } else {
        parameters = YAML::LoadFile("render_cli.yaml");
    }

    VolumeBlocks* volume = NULL;
    TransferFunction* tf = NULL;
    Lens* lens = NULL;
    Image* img = NULL;

    if(parameters["volume"].IsDefined()) {
        volume = VolumeBlocks::LoadFromFile(parameters["volume"].as<std::string>().c_str());
    }

    if(parameters["volume_hdf5"].IsDefined()) {
        volume = Dataset_FLASH_Create(
            parameters["volume_hdf5"].as<std::string>().c_str(),
            parameters["volume_hdf5_field"].as<std::string>().c_str()
        );
    }

    if(!volume) {
        fprintf(stderr, "Error: Volume not loaded.\n");
        return -1;
    }

    img = Image::Create(!parameters["width"].IsDefined() ? 800 : parameters["width"].as<int>(),
                        !parameters["height"].IsDefined() ? 400 : parameters["height"].as<int>());

    protocol::ParameterPreset preset;

    try {
        std::fstream input(parameters["preset"].as<std::string>().c_str(), std::ios::in | std::ios::binary);
        preset.ParseFromIstream(&input);
    } catch(...) {
        fprintf(stderr, "Error: Preset not found.\n");
        return -1;
    }


    lens = Lens::CreatePerspective(90.0);

    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();
    renderer->setBackgroundColor(Color(0, 0, 0, 0));

    Pose pose;
    pose.position = Vector(preset.pose().x(), preset.pose().y(), preset.pose().z());
    pose.rotation = Quaternion(preset.pose().qw(), preset.pose().qx(), preset.pose().qy(), preset.pose().qz());
    renderer->setPose(pose);

    lens->setEyeSeparation(preset.lens_parameters().eye_separation());
    lens->setFocalDistance(preset.lens_parameters().focal_distance());

    renderer->setBlendingCoefficient(preset.renderer_parameters().blending_coefficient());
    renderer->setStepSizeMultiplier(preset.renderer_parameters().step_size());
    switch(preset.renderer_parameters().method()) {
        case protocol::RendererParameters_RenderingMethod_BasicBlending: {
            renderer->setRaycastingMethod(VolumeRenderer::kBasicBlendingMethod);
        } break;
        case protocol::RendererParameters_RenderingMethod_AdaptiveRKF: {
            renderer->setRaycastingMethod(VolumeRenderer::kAdaptiveRKFMethod);
        } break;
        case protocol::RendererParameters_RenderingMethod_PreIntegration: {
            renderer->setRaycastingMethod(VolumeRenderer::kPreIntegrationMethod);
        } break;
    }
    renderer->setRaycastingMethod(VolumeRenderer::kPreIntegrationMethod);

    tf = TransferFunction::CreateGaussianTicks(1e-3, 1e8, TransferFunction::kLogScale, 16);
    tf->setDomain(preset.transfer_function().domain_min(), preset.transfer_function().domain_max());
    switch(preset.transfer_function().scale()) {
        case protocol::TransferFunction_Scale_Linear: {
            tf->setScale(TransferFunction::kLinearScale);
        } break;
        case protocol::TransferFunction_Scale_Log: {
            tf->setScale(TransferFunction::kLogScale);
        } break;
    }
    TransferFunction::ParseLayers(tf, 1024, preset.transfer_function().layers().c_str());

    renderer->setInternalFormat(VolumeRenderer::kUInt8);

    renderer->setVolume(volume);
    renderer->setLens(lens);
    renderer->setTransferFunction(tf);
    Image* img_back = Image::Create(!parameters["width"].IsDefined() ? 800 : parameters["width"].as<int>(),
                        !parameters["height"].IsDefined() ? 400 : parameters["height"].as<int>());
    renderer->setImage(img);
    renderer->setBackImage(img_back);

    int trails = 5;
    {
        renderer->setEnableZIndex(false);
        double ts = 0;
        for(int i = 0; i < trails; i++) {
            time_measure.begin("Render");
            renderer->render();
            img->setNeedsDownload();
            img_back->setNeedsDownload();
            img->getPixels(); img_back->getPixels();
            ts += time_measure.done();
        }
        printf("Average Render Time (UInt8, PreInt, no z-index): %.3lf ms\n", ts * 1000 / trails);
    }
    {
        renderer->setEnableZIndex(true);
        double ts = 0;
        for(int i = 0; i < trails; i++) {
            time_measure.begin("Render");
            renderer->render();
            img->setNeedsDownload();
            img_back->setNeedsDownload();
            img->getPixels(); img_back->getPixels();
            ts += time_measure.done();
        }
        printf("Average Render Time (UInt8, PreInt, with z-index): %.3lf ms\n", ts * 1000 / trails);
    }
    img->setNeedsDownload();
    img_back->setNeedsDownload();
    printf("Saving image.\n");
    img->save(parameters["output"].as<std::string>().c_str(), "png16");
    img_back->save((string("back_") + parameters["output"].as<std::string>()).c_str(), "png16");
}
