#include "renderer.h"
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

#include "allovolume_protocol.pb.h"
#include "allovolume_common.h"

#include <cuda_runtime.h>

using namespace std;
using namespace allovolume;

void* zmq_context;

ConfigParser config;

// The renderer has 4 state varaibles:
//  volume: The volume to be rendered.
//  pose: The pose of the viewer. (x, y, z, qx, qy, qz, qw)
//  lens: The information for the lens. (eye_separation, focal_distance)
//  transfer_function: The transfer function to use.
//  RGB curve: The rgb curve for final output.
// Volume is set only, others are get/set.

typedef boost::shared_ptr<VolumeRenderer> VolumeRendererPointer;
typedef boost::shared_ptr<TransferFunction> TransferFunctionPointer;
typedef boost::shared_ptr<Image> ImagePointer;
typedef boost::shared_ptr<AllosphereLens> AllosphereLensPointer;

struct GPURenderThreadEvent {
    enum Type {
        kRenderComplete
    };
    Type type;
    int thread_id;
};

class GPURenderThread {
public:
    VolumeRendererPointer renderer;
    TransferFunctionPointer tf;

    vector<AllosphereLensPointer> lenses;
    vector<ImagePointer> images;

    // Not accessible from GPU threads.
    vector<GLuint> textures;

    AllosphereCalibration::RenderSlave* slave;

    int thread_id;


    GPURenderThread() { }

    static void* thread_proc(void* this_) {
        ((GPURenderThread*)this_)->thread_entrypoint();
    }

    void thread_entrypoint() {
        cudaSetDevice(gpu_id);
        renderer.reset(VolumeRenderer::CreateGPU());
        tf.reset(TransferFunction::CreateGaussianTicks(1e-3, 1e8, TransferFunction::kLogScale, 16));
        renderer->setBlendingCoefficient(1e10);
        for(int i = 0; i < slave->num_projections; i++) {
            lenses.push_back(AllosphereLensPointer(AllosphereCalibration::CreateLens(&slave->projections[i])));
            images.push_back(ImagePointer(Image::Create(960, 640)));
        }
        for(int i = 0; i < slave->num_projections; i++) {
            lenses[i]->setEyeSeparation(1e9 * eye);
            lenses[i]->setFocalDistance(1e10);
        }

        void* socket_sub = zmq_socket(zmq_context, ZMQ_SUB);
        zmq_connect(socket_sub, "inproc://render_slaves");
        zmq_setsockopt(socket_sub, ZMQ_SUBSCRIBE, "", 0);

        void* socket_push = zmq_socket(zmq_context, ZMQ_PUSH);
        zmq_connect(socket_push, "inproc://render_slaves_push");

        while(1) {
            zmq_msg_t zmsg;
            zmq_msg_init(&zmsg);
            int r = zmq_msg_recv(&zmsg, socket_sub, 0);
            if(r < 0) {
                fprintf(stderr, "render_thread: %s\n", zmq_strerror(zmq_errno()));
                break;
            }
            protocol::RendererBroadcast msg;
            msg.ParseFromArray(zmq_msg_data(&zmsg), zmq_msg_size(&zmsg));

            printf("Renderer: %s\n", protocol::RendererBroadcast_RequestType_Name(msg.type()).c_str());

            switch(msg.type()) {
                case protocol::RendererBroadcast_RequestType_LoadVolume: {
                    const void* content = &msg.volume_data()[0];
                    size_t content_length = msg.volume_data().size();
                    volume.reset(VolumeBlocks::LoadFromBuffer(content, content_length));
                    renderer->setVolume(volume.get());
                    printf("Volume: %llu blocks.\n", volume->getBlockCount());
                } break;
                case protocol::RendererBroadcast_RequestType_LoadVolumeFromFile: {
                    // TODO
                } break;
                case protocol::RendererBroadcast_RequestType_SetPose: {
                    Pose pose;
                    pose.position = Vector(msg.pose().x(), msg.pose().y(), msg.pose().z());
                    pose.rotation = Quaternion(msg.pose().qw(), msg.pose().qx(), msg.pose().qy(), msg.pose().qz());
                    renderer->setPose(pose);
                } break;
                case protocol::RendererBroadcast_RequestType_SetLensParameters: {
                    for(int i = 0; i < slave->num_projections; i++) {
                        lenses[i]->setEyeSeparation(msg.lens_parameters().eye_separation());
                        lenses[i]->setFocalDistance(msg.lens_parameters().focal_distance());
                    }
                } break;
                case protocol::RendererBroadcast_RequestType_SetRendererParameters: {
                    renderer->setBlendingCoefficient(msg.renderer_parameters().blending_coefficient());
                    switch(msg.renderer_parameters().method()) {
                        case protocol::RendererParameters_RenderingMethod_RK4: {
                            renderer->setRaycastingMethod(VolumeRenderer::kRK4Method);
                        } break;
                        case protocol::RendererParameters_RenderingMethod_AdaptiveRKV: {
                            renderer->setRaycastingMethod(VolumeRenderer::kAdaptiveRKVMethod);
                        } break;
                    }
                } break;
                case protocol::RendererBroadcast_RequestType_SetTransferFunction: {
                    tf->setDomain(msg.transfer_function().domain_min(), msg.transfer_function().domain_max());
                    switch(msg.transfer_function().scale()) {
                        case protocol::TransferFunction_Scale_Linear: {
                            tf->setScale(TransferFunction::kLinearScale);
                        } break;
                        case protocol::TransferFunction_Scale_Log: {
                            tf->setScale(TransferFunction::kLogScale);
                        } break;
                    }
                    tf->setContent((Color*)&msg.transfer_function().content()[0], msg.transfer_function().content().size() / sizeof(Color));
                } break;
                case protocol::RendererBroadcast_RequestType_SetRGBCurve: {
                } break;
                case protocol::RendererBroadcast_RequestType_Render: {
                    t_render_scene();
                    GPURenderThreadEvent event;
                    event.type = GPURenderThreadEvent::kRenderComplete;
                    event.thread_id = thread_id;
                    zmq_msg_t msg;
                    zmq_msg_init_size(&msg, sizeof(GPURenderThreadEvent));
                    memcpy(zmq_msg_data(&msg), &event, sizeof(GPURenderThreadEvent));
                    zmq_msg_send(&msg, socket_push, 0);
                } break;
                case protocol::RendererBroadcast_RequestType_Present: {
                } break;
            }
        }
    }

    void t_render_scene() {
        for(int i = 0; i < slave->num_projections; i++) {
            renderer->setTransferFunction(tf.get());
            renderer->setLens(lenses[i].get());
            renderer->setImage(images[i].get());
            renderer->render();
            // TODO: Perform RGB curve here.
            lenses[i]->performBlend(images[i].get());
            images[i]->setNeedsDownload();
            images[i]->getPixels();
        }
        is_dirty = true;
    }

    // This should be run in main thread.
    void initialize(int gpu_id_, int thread_id_, AllosphereCalibration::RenderSlave* slave_, float eye_) {
        thread_id = thread_id_;
        gpu_id = gpu_id_;
        slave = slave_;
        eye = eye_;
        for(int i = 0; i < slave->num_projections; i++) {
            GLuint tex;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            textures.push_back(tex);
        }
        is_dirty = true;
        // Spawn the rendering thread.
        pthread_create(&thread, NULL, thread_proc, this);
    }

    void uploadImages() {
        if(is_dirty && !textures.empty()) {
            for(int i = 0; i < slave->num_projections; i++) {
                glBindTexture(GL_TEXTURE_2D, textures[i]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, images[i]->getWidth(), images[i]->getHeight(), 0, GL_RGBA, GL_FLOAT, images[i]->getPixels());
            }
            is_dirty = false;
        }
    }

    ~GPURenderThread() {
        should_exit = true;
        pthread_join(thread, NULL);
    }

    bool should_exit;
    bool is_dirty;
    int gpu_id;
    bool needs_upload_tf;
    bool needs_render;

    float eye;

    boost::shared_ptr<VolumeBlocks> volume;

    pthread_t thread;

    void* socket_push;
    void* socket_sub;
};

class Renderer {
public:

    enum StereoMode {
        kActiveStereoMode = 0,
        kAnaglyphStereoMode = 1,
        kMonoLeftStereoMode = 2,
        kMonoRightStereoMode = 3
    };

    Renderer() {
        calibration = AllosphereCalibration::Load((string(getenv("HOME")) + "/calibration-current").c_str());
        render_slave = calibration->getRenderer();

        int device_count;
        cudaGetDeviceCount(&device_count);

        initWindow();

        renderer_left.initialize(0 % device_count, 0, render_slave, 1);
        renderer_right.initialize(1 % device_count, 1, render_slave, -1);
    }

    static void* subscription_thread_pthread(void* this_) {
        ((Renderer*)this_)->subscription_thread();
    }
    void subscription_thread() {
        void* socket_sub = zmq_socket(zmq_context, ZMQ_SUB);
        int value;
        value = config.get<int>("SyncSystem.zmq.rcvhwm", 10000);
        zmq_setsockopt(socket_sub, ZMQ_RCVHWM, &value, sizeof(int));
        value = config.get<int>("SyncSystem.zmq.rcvbuf", 0);
        zmq_setsockopt(socket_sub, ZMQ_RCVBUF, &value, sizeof(int));
        value = config.get<int>("SyncSystem.zmq.rate", 10000000);
        zmq_setsockopt(socket_sub, ZMQ_RATE, &value, sizeof(int));
        zmq_connect(socket_sub, config.get<string>("SyncSystem.broadcast").c_str());
        zmq_setsockopt(socket_sub, ZMQ_SUBSCRIBE, "", 0);

        void* socket_feedback = zmq_socket(zmq_context, ZMQ_PUSH);
        zmq_connect(socket_feedback, config.get<string>("SyncSystem.feedback").c_str());

        void* socket_pub = zmq_socket(zmq_context, ZMQ_PUB);
        zmq_bind(socket_pub, "inproc://render_slaves");

        while(1) {
            zmq_msg_t msg;
            zmq_msg_init(&msg);
            int r = zmq_msg_recv(&msg, socket_sub, 0);
            if(r < 0) {
                fprintf(stderr, "zmq_msg_recv: %s\n", zmq_strerror(zmq_errno()));
                break;
            }
            zmq_msg_send(&msg, socket_pub, 0);
        }
    }

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

        bool fullscreen_mode = config.get<string>("allovolume.fullscreen", "false") == "true";

        if(fullscreen_mode) {
            glfwWindowHint(GLFW_DECORATED, GL_FALSE);
            glfwWindowHint(GLFW_STEREO, GL_TRUE);
            window = glfwCreateWindow(screen_width, screen_height, "Allosphere Volume Renderer", NULL, NULL);
            stereo_mode = kActiveStereoMode;
            if(!window) {
                // Unable to create stereo window, fallback to anaglyph.
                glfwWindowHint(GLFW_STEREO, GL_FALSE);
                window = glfwCreateWindow(screen_width, screen_height, "Allosphere Volume Renderer", NULL, NULL);
                stereo_mode = kAnaglyphStereoMode;
            }
            glfwSetWindowPos(window, 0, 0);
        } else {
            glfwWindowHint(GLFW_STEREO, GL_TRUE);
            window = glfwCreateWindow(960, 640, "Allosphere Volume Renderer", NULL, NULL);
            stereo_mode = kActiveStereoMode;
            if(!window) {
                // Unable to create stereo window, fallback to anaglyph.
                glfwWindowHint(GLFW_STEREO, GL_FALSE);
                window = glfwCreateWindow(960, 640, "Allosphere Volume Renderer", NULL, NULL);
                stereo_mode = kAnaglyphStereoMode;
            }
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

    void startup() {
        pthread_create(&thread, NULL, subscription_thread_pthread, this);
        void* socket_pull = zmq_socket(zmq_context, ZMQ_PULL);
        zmq_bind(socket_pull, "inproc://render_slaves_push");
        set<int> barrier_thread;
        while(1) {
            zmq_msg_t msg;
            zmq_msg_init(&msg);
            zmq_msg_recv(&msg, socket_pull, 0);
            GPURenderThreadEvent event = *(GPURenderThreadEvent*)zmq_msg_data(&msg);
            zmq_msg_close(&msg);
            barrier_thread.insert(event.thread_id);
            if(barrier_thread.size() == 2) {
                display();
                glfwPollEvents();
                barrier_thread.clear();
            }

        }
    }

    void render_image(GPURenderThread& renderer) {
        GLint windowWidth, windowHeight;
        glfwGetFramebufferSize(window, &windowWidth, &windowHeight);

        renderer.uploadImages();
        for(int i = 0; i < render_slave->num_projections; i++) {
            glViewport(render_slave->projections[i].viewport_x * windowWidth, render_slave->projections[i].viewport_y * windowHeight, render_slave->projections[i].viewport_w * windowWidth, render_slave->projections[i].viewport_h * windowHeight);
            glBindTexture(GL_TEXTURE_2D, renderer.textures[i]);
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(-1, -1);
            glTexCoord2f(0, 1); glVertex2f(-1, +1);
            glTexCoord2f(1, 1); glVertex2f(+1, +1);
            glTexCoord2f(1, 0); glVertex2f(+1, -1);
            glEnd();
        }
    }

    void display() {
        // Draw stuff
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE);
        glEnable(GL_TEXTURE_2D);

        switch(stereo_mode) {
            case kActiveStereoMode: {
            } break;
            case kAnaglyphStereoMode: {
                glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE);
                render_image(renderer_left);
                glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE);
                render_image(renderer_right);

            } break;
            case kMonoLeftStereoMode: {
                render_image(renderer_left);
            } break;
            case kMonoRightStereoMode: {
                render_image(renderer_right);
            } break;
        }


        // Update Screen
        glfwSwapBuffers(window);
    }

    // virtual void onMessage(SyncSystem* sync, void* data, size_t length) {
    //     protocol::RendererBroadcast msg;
    //     msg.ParseFromArray(data, length);

    //     switch(msg.type()) {
    //         case protocol::RendererBroadcast_RequestType_LoadVolume: {
    //             const void* content = &msg.volume_data()[0];
    //             size_t content_length = msg.volume_data().size();
    //             boost::shared_ptr<VolumeBlocks> volume(VolumeBlocks::LoadFromBuffer(content, content_length));
    //             renderer_left.setVolume(volume);
    //             renderer_right.setVolume(volume);
    //             printf("Volume: %llu blocks.\n", volume->getBlockCount());
    //         } break;
    //         case protocol::RendererBroadcast_RequestType_Render: {
    //             renderer_left.triggerRender();
    //             renderer_right.triggerRender();
    //         } break;
    //         case protocol::RendererBroadcast_RequestType_SetPose: {
    //             Pose pose;
    //             pose.position = Vector(msg.pose().x(), msg.pose().y(), msg.pose().z());
    //             pose.rotation = Quaternion(msg.pose().qw(), msg.pose().qx(), msg.pose().qy(), msg.pose().qz());
    //             renderer_left.setPose(pose);
    //             renderer_right.setPose(pose);
    //         } break;
    //         case protocol::RendererBroadcast_RequestType_Present: {
    //             display();
    //         } break;
    //     }
    // }

    // virtual void onBarrier(SyncSystem* sync, size_t sequence_id) {
    //     printf("Barrier: %llu\n", sequence_id);
    //     sync->clearBarrier(sequence_id);
    // }

    AllosphereCalibration* calibration;
    GPURenderThread renderer_left, renderer_right;
    GLFWwindow* window;
    AllosphereCalibration::RenderSlave* render_slave;
    pthread_t thread;
    StereoMode stereo_mode;
};

int main(int argc, char* argv[]) {
    zmq_context = zmq_ctx_new();
    config.parseFile("allovolume.client.yaml");
    Renderer renderer;
    renderer.startup();
}
