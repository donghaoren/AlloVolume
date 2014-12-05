#include "renderer.h"
#include "allosync.h"
#include "allosphere_calibration.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <readline/readline.h>

#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

using namespace std;
using namespace allovolume;

#define MESSAGE_LoadVolume            0
#define MESSAGE_LoadVolumeFromFile    1
#define MESSAGE_SetPose               2
#define MESSAGE_SetTransferFunction   3
#define MESSAGE_Render                4
#define MESSAGE_Present               5

struct MessageHeader {
    unsigned int type;
};

class Controller : public SyncSystem::Delegate {
public:
    Controller(SyncSystem* sync_) {
        sync = sync_;
        sync->setDelegate(this);
        sync->start();
    }
    void loadVolume(VolumeBlocks* volume) {
        size_t length = VolumeBlocks::WriteToBufferSize(volume);
        unsigned char* message = (unsigned char*)malloc(sizeof(MessageHeader) + length);
        MessageHeader* header = (MessageHeader*)message;
        VolumeBlocks::WriteToBuffer(volume, message + sizeof(MessageHeader), length);

        header->type = MESSAGE_LoadVolume;

        sync->sendMessage(message, sizeof(MessageHeader) + length);
        free(message);
    }

    void render() {
        MessageHeader header;
        header.type = MESSAGE_Render;
        sync->sendMessage(&header, sizeof(MessageHeader));
    }

    void present() {
        MessageHeader header;
        header.type = MESSAGE_Present;
        sync->sendMessage(&header, sizeof(MessageHeader));
    }

    void barrier() {
        waiting_for_barrier = true;
        sync->sendBarrier();
        while(waiting_for_barrier) { usleep(1000); }
    }

    virtual void onBarrierClear(SyncSystem* sync, size_t sequence_id) {
        waiting_for_barrier = false;
    }

    volatile bool waiting_for_barrier = false;

    SyncSystem* sync;
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
            images.push_back(Image::Create(100, 100));
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
        //window = glfwCreateWindow(mode->width, mode->height, "Allosphere Volume Renderer", glfwGetPrimaryMonitor(), NULL);
        glfwWindowHint(GLFW_DECORATED, GL_FALSE);
        window = glfwCreateWindow(screen_width, screen_width, "Allosphere Volume Renderer", NULL, NULL);
        glfwSetWindowPos(window, 0, 0);

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
        MessageHeader* header = (MessageHeader*)data;
        unsigned char* content = (unsigned char*)(header + 1);

        if(header->type == MESSAGE_LoadVolume) {
            if(volume) delete volume;
            volume = VolumeBlocks::LoadFromBuffer(content, length - sizeof(MessageHeader));
            renderer->setVolume(volume);
            printf("Volume loaded. blocks = %llu, size = %llu\n", volume->getBlockCount(), volume->getDataSize());
        } else
        if(header->type == MESSAGE_Render) {
            for(int i = 0; i < render_slave->num_projections; i++) {
                renderer->setLens(lenses[i]);
                renderer->setImage(images[i]);
                renderer->setTransferFunction(tf);
                renderer->render();
                lenses[i]->performBlend(images[i]);
                images[i]->setNeedsDownload();
                glBindTexture(GL_TEXTURE_2D, textures[i]);
                glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA, images[i]->getWidth(), images[i]->getHeight(), 0, GL_RGBA, GL_FLOAT, images[i]->getPixels());
            }
        }
        if(header->type == MESSAGE_Present) {
            display();
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
    if(argc == 1) {
        printf("Server...\n");
        SyncSystem* sync = SyncSystem::Create("allovolume.server.yaml");
        Controller controller(sync);
        while(1) {
            char* input = readline("> ");
            if(!input) break;
            string in(input);
            free(input);
            if(in == "volume") {
                VolumeBlocks* vol = VolumeBlocks::LoadFromFile("super3d_hdf5_plt_cnt_0122.volume");
                controller.loadVolume(vol);
                delete vol;
            }
            if(in == "render") {
                controller.render();
            }
            if(in == "present") {
                controller.barrier();
                controller.present();
            }
        }
    } else {
        SyncSystem* sync = SyncSystem::Create("allovolume.client.yaml");
        Renderer renderer(sync);
    }
}
