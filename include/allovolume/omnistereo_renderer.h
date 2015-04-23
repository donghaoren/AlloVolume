#ifndef ALLOVOLUME_OMNISTEREO_RENDERER_H_INCLUDED
#define ALLOVOLUME_OMNISTEREO_RENDERER_H_INCLUDED

#include "allovolume/dataset.h"
#include "allovolume/utils.h"

namespace allovolume {

// Interoperation with OmniStereo.
// The process is:
//   OmniStereo render to cubemap, with color and depth for each face.
//   AlloVolume takes the depth buffer, render the volume into two parts.
//   Finally, blend volume back, cubemap color buffer, volume front together to produce the final image.
// For speed concerns, AlloVolume works in another thread.

// Typical interoperation sequence:
/*
  void init() {
    OmnistereoRenderer* allovolume = OmnistereoRenderer::CreateWithYAMLConfig("allovolume.yaml");
    allovolume->setDelegate(myDelegate);
  }

  void omnistereoFrame() {
    // Here we're in the OpenGL main thread.
    // Capture cubemap...
    // Grab the depth part of the cubemap...
    allovolume->setCubemap(id);

    // Wait for callback onPresent() from allovolume.
    // Calling present() directly is okay, but the images won't sync correctly.
    // Since allovolume can be slow, it's recommended to present() directly.
  }

  void MyDelegate::onPresent() {
    // Here we are in allovolume's main thread.
    // Notify the OpenGL main thread to refresh the scene.
  }

  void present() {
    // Here we're in the OpenGL main thread.
    allovolume->uploadTextures();
    for(int vp = 0; vp < vp_count; vp++) {
        for(int eye = 0; eye < eye_count; eye++) {
            Textures textures = allovolume->getTextures(vp, eye);
            // Render current viewport by blending
            // textures.back, omnistereo_cubemap and textures.front together and apply blending.
        }
    }
  }
 */

class OmnistereoRenderer {
public:

    struct Textures {
        unsigned int front;
        unsigned int back;
    };

    class Delegate {
    public:
        // Functions from the delegate are called from another thread.

        // Results are available to present.
        virtual void onPresent() { }

        virtual ~Delegate() { }
    };

    // Is all initialization completed?
    virtual bool isReady() = 0;

    // Set the cubemap for depth buffer.
    virtual void loadDepthCubemap(unsigned int texDepth_left, unsigned int texDepth_right, float near, float far, float eyesep) = 0;

    // Upload viewport textures.
    virtual void uploadTextures() = 0;

    // Get viewport textures.
    virtual Textures getTextures(int viewport, int eye) = 0;

    // Set the delegate.
    virtual void setDelegate(Delegate* delegate) = 0;

    virtual ~OmnistereoRenderer() { }

    // Create the renderer with allovolume.yaml.
    // Call this in the same thread of your OpenGL context.
    static OmnistereoRenderer* CreateWithYAMLConfig(const char* path);
};

}

#endif
