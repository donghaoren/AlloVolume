#include "renderer.h"
#include "allosync.h"
#include "configparser.h"
#include "allosphere/allosphere_calibration.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <readline/readline.h>

#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <zmq.h>

#include "renderer.pb.h"

#include <cuda_runtime.h>

using namespace std;
using namespace allovolume;

// The renderer has 4 state varaibles:
//  volume: The volume to be rendered.
//  pose: The pose of the viewer. (x, y, z, qx, qy, qz, qw)
//  lens: The information for the lens. (eye_separation, focal_distance)
//  transfer_function: The transfer function to use.
//  RGB curve: The rgb curve for final output.
// Volume is set only, others are get/set.

class GPURenderThread {
public:
    GPURenderThread() { }
    void initialize(int gpuid) {
        cudaSetDevice(gpuid);
    }
};

class Renderer : public SyncSystem::Delegate {
public:
    Renderer(SyncSystem* sync_) {
        sync = sync_;
        sync->setDelegate(this);
        sync->start();

        calibration = AllosphereCalibration::Load((string(getenv("HOME")) + "/calibration-current").c_str());
        render_slave = calibration->getRenderer();

        volume = NULL;
        for(int i = 0; i < render_slave->num_projections; i++) {
            lenses.push_back(AllosphereCalibration::CreateLens(&render_slave->projections[i]));
            images.push_back(Image::Create(96, 64));
        }
        renderer = VolumeRenderer::CreateGPU();
        tf = TransferFunction::CreateGaussianTicks(1e-3, 1e8, 20, true);
        tf->getMetadata()->blend_coefficient = 1e10;

        initWindow();
        for(int i = 0; i < render_slave->num_projections; i++) {
            GLuint tex;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            textures.push_back(tex);
        }

        while(1) {
            sync->waitEvent();
        }
    }

    GLFWwindow* window;
    AllosphereCalibration::RenderSlave* render_slave;

    void initWindow() {
        if(!glfwInit()) {
            fprintf(stderr, "Failed to initialize GLFW\n");
        }
        //glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
        // Open a window and create its OpenGL context
        int monitor_count = 0;
        GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);

        // Get the size of the virtual screen.
        int screen_width = 0, screen_height = 0;
        for(int i = 0; i < monitor_count; i++) {
            int x, y;
            const GLFWvidmode *mode = glfwGetVideoMode(monitors[i]);
            glfwGetMonitorPos(monitors[i], &x, &y);
            x += mode->width;
            y += mode->height;
            screen_width = max(screen_width, x);
            screen_height = max(screen_height, y);
        }
        printf("Screen: %d %d\n", screen_width, screen_height);

        bool fullscreen_mode = false;

        if(fullscreen_mode) {
            glfwWindowHint(GLFW_DECORATED, GL_FALSE);
            window = glfwCreateWindow(screen_width, screen_height, "Allosphere Volume Renderer", NULL, NULL);
            glfwSetWindowPos(window, 0, 0);
        } else {
            window = glfwCreateWindow(960, 640, "Allosphere Volume Renderer", NULL, NULL);
        }

        if(window == NULL) {
            fprintf(stderr, "Failed to open GLFW window.\n");
            glfwTerminate();
        }

        glfwMakeContextCurrent(window);

        // Get info of GPU and supported OpenGL version
        printf("Renderer: %s\n", glGetString(GL_RENDERER));
        printf("OpenGL version supported %s\n", glGetString(GL_VERSION));
    }

    void display() {
        GLint windowWidth, windowHeight;
        glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);
        printf("framebuffer: %d %d\n", windowWidth, windowHeight);

        // Draw stuff
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_TEXTURE);
        glEnable(GL_TEXTURE_2D);

        for(int i = 0; i < render_slave->num_projections; i++) {
            glViewport(render_slave->projections[i].viewport_x * windowWidth, render_slave->projections[i].viewport_y * windowHeight, render_slave->projections[i].viewport_w * windowWidth, render_slave->projections[i].viewport_h * windowHeight);
            glBindTexture(GL_TEXTURE_2D, textures[i]);
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(-1, -1);
            glTexCoord2f(0, 1); glVertex2f(-1, +1);
            glTexCoord2f(1, 1); glVertex2f(+1, +1);
            glTexCoord2f(1, 0); glVertex2f(+1, -1);
            glEnd();
        }

        // Update Screen
        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    virtual void onMessage(SyncSystem* sync, void* data, size_t length) {
        protocol::RendererBroadcast msg;
        msg.ParseFromArray(data, length);

        switch(msg.type()) {
            case protocol::RendererBroadcast_RequestType_LoadVolume: {
                const void* content = &msg.volume_data()[0];
                size_t content_length = msg.volume_data().size();
                volume = VolumeBlocks::LoadFromBuffer(content, content_length);
                renderer->setVolume(volume);
                printf("Volume: %llu blocks.\n", volume->getBlockCount());
            } break;
            case protocol::RendererBroadcast_RequestType_Render: {
                for(int i = 0; i < render_slave->num_projections; i++) {
                    printf("Render for projection %d\n", i);
                    renderer->setLens(lenses[i]);
                    renderer->setImage(images[i]);
                    renderer->setTransferFunction(tf);
                    renderer->render();
                    lenses[i]->performBlend(images[i]);
                    images[i]->setNeedsDownload();
                    glBindTexture(GL_TEXTURE_2D, textures[i]);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, images[i]->getWidth(), images[i]->getHeight(), 0, GL_RGBA, GL_FLOAT, images[i]->getPixels());
                    printf("Render for projection %d done.\n", i);
                }
            } break;
            case protocol::RendererBroadcast_RequestType_SetPose: {
                Vector origin(msg.pose().x(), msg.pose().y(), msg.pose().z());
                for(int i = 0; i < render_slave->num_projections; i++) {
                    lenses[i]->setParameter("origin", &origin);
                }
            } break;
            case protocol::RendererBroadcast_RequestType_Present: {
                display();
            } break;
        }
    }

    virtual void onBarrier(SyncSystem* sync, size_t sequence_id) {
        printf("Barrier: %llu\n", sequence_id);
        sync->clearBarrier(sequence_id);
    }

    SyncSystem* sync;
    AllosphereCalibration* calibration;

    VolumeBlocks* volume;
    VolumeRenderer* renderer;

    TransferFunction* tf;

    vector<AllosphereLens*> lenses;
    vector<Image*> images;
    vector<GLuint> textures;
};

int main(int argc, char* argv[]) {
    SyncSystem* sync = SyncSystem::Create("allovolume.client.yaml");
    Renderer renderer(sync);
}
