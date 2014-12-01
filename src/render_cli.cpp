// Command line interface for the renderer.

#include "dataset.h"
#include "renderer.h"

#include <stdio.h>
#include <string>
#include <map>
#include <yaml-cpp/yaml.h>

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

    if(!parameters["volume"].IsNull()) {
        volume = VolumeBlocks::LoadFromFile(parameters["volume"].as<std::string>().c_str());
    }

    if(!parameters["hdf5"].IsNull()) {
        volume = Dataset_FLASH_Create(
            parameters["hdf5"].as<std::string>().c_str(),
            parameters["field"].as<std::string>().c_str()
        );
    }

    if(!volume) {
        fprintf(stderr, "Error: Volume not loaded.\n");
        return -1;
    }

    img = Image::Create(parameters["width"].IsNull() ? 800 : parameters["width"].as<int>(),
                        parameters["height"].IsNull() ? 400 : parameters["height"].as<int>());

    if(!parameters["equirectangular"].IsNull()) {
        Vector origin = node2vector(parameters["equirectangular"]["origin"]);
        Vector up = node2vector(parameters["equirectangular"]["up"]);
        Vector direction = node2vector(parameters["equirectangular"]["direction"]);
        lens = Lens::CreateEquirectangular(origin, up, direction);
    }

    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();

    tf = TransferFunction::CreateGaussianTicks(1e-3, 1e8, 20, true);
    tf->getMetadata()->blend_coefficient = 1e10;


    int image_size = 1600 * 800;
    img = Image::Create(1600, 1600);
    Image* img_render = Image::Create(1600, 800);

    renderer->setVolume(volume);
    renderer->setLens(lens1);
    renderer->setTransferFunction(tf);
    renderer->setImage(img_render);
    renderer->render();
    img_render->setNeedsDownload();
    Color* pixels = img_render->getPixels();
    for(int i = 0; i < image_size; i++) {
        pixels[i].r = transform_color(pixels[i].r);
        pixels[i].g = transform_color(pixels[i].g);
        pixels[i].b = transform_color(pixels[i].b);
        pixels[i].a = 1;
    }
    memcpy(img->getPixels(), pixels, sizeof(Color) * image_size);

    renderer->setLens(lens2);
    renderer->setTransferFunction(tf);
    renderer->setImage(img_render);
    renderer->render();
    img_render->setNeedsDownload();
    pixels = img_render->getPixels();
    for(int i = 0; i < image_size; i++) {
        pixels[i].r = transform_color(pixels[i].r);
        pixels[i].g = transform_color(pixels[i].g);
        pixels[i].b = transform_color(pixels[i].b);
        pixels[i].a = 1;
    }
    memcpy(img->getPixels() + image_size, pixels, sizeof(Color) * image_size);

    img->save(argv[2], "png16");
}
