#ifndef ALLOVOLUME_OMNISTEREO_RENDERER_H_INCLUDED
#define ALLOVOLUME_OMNISTEREO_RENDERER_H_INCLUDED

#include "dataset.h"
#include "utils.h"

namespace allovolume {

// Interoperation with OmniStereo.
// The process is:
//   OmniStereo render to cubemap, with color and depth for each face.
//   AlloVolume takes the depth buffer, render the volume into two parts.
//   Finally, blend volume back, cubemap color buffer, volume front together to produce the final image.
// For speed concerns, AlloVolume works in another thread.
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

    // Set the cubemap for depth buffer.
    virtual void setCubemap(unsigned int cubemap_id) = 0;

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
