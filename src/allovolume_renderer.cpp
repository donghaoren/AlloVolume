#include "renderer.h"
#include "configparser.h"
#include "allosphere/allosphere_calibration.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <readline/readline.h>

#include <string>
#include <vector>

#include <zmq.h>

#include "allovolume_protocol.pb.h"
#include "allovolume_common.h"

#include <cuda_runtime.h>

#ifdef __APPLE__
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
    #include <GL/freeglut_ext.h>
#endif

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
        kPresent
    };
    Type type;
    int thread_id;
};

class GPURenderThread {
public:
    // Note: Each GPU thread has its own CUDA contexts, the pointers are not interchangable!!
    VolumeRendererPointer renderer;
    TransferFunctionPointer tf;

    vector<AllosphereLensPointer> lenses;
    vector<ImagePointer> images;

    // Not accessible from GPU threads.
    vector<GLuint> textures;

    AllosphereCalibration::RenderSlave* slave;

    int thread_id;

    Pose current_pose;

    float rgb_levels_min;
    float rgb_levels_max;
    float rgb_levels_pow;


    GPURenderThread() {
        initialize_complete = false;
        rgb_levels_min = 0;
        rgb_levels_max = 1;
        rgb_levels_pow = 1;
    }

    static void* thread_proc(void* this_) {
        ((GPURenderThread*)this_)->thread_entrypoint();
    }

    void thread_entrypoint() {
        cudaSetDevice(gpu_id);
        renderer.reset(VolumeRenderer::CreateGPU());
        tf.reset(TransferFunction::CreateGaussianTicks(1e-3, 1e8, TransferFunction::kLogScale, 16));
        renderer->setBlendingCoefficient(1e9);
        renderer->setBackgroundColor(Color(0, 0, 0, 1));
        for(int i = 0; i < slave->num_projections; i++) {
            lenses.push_back(AllosphereLensPointer(AllosphereCalibration::CreateLens(&slave->projections[i])));
            images.push_back(ImagePointer(Image::Create(config.get<int>("allovolume.resolution.width", 960), config.get<int>("allovolume.resolution.height", 640))));
        }
        for(int i = 0; i < slave->num_projections; i++) {
            lenses[i]->setEyeSeparation(1e9 * eye);
            lenses[i]->setFocalDistance(1e10);
        }

        initialize_complete = true;

        void* socket_sub = zmq_socket(zmq_context, ZMQ_SUB);
        zmq_connect(socket_sub, "inproc://render_slaves");
        zmq_setsockopt(socket_sub, ZMQ_SUBSCRIBE, "", 0);

        void* socket_push = zmq_socket(zmq_context, ZMQ_PUSH);
        zmq_connect(socket_push, "inproc://render_slaves_push");

        void* socket_feedback = zmq_socket(zmq_context, ZMQ_PUSH);
        zmq_connect(socket_feedback, config.get<string>("sync.feedback").c_str());

        protocol::RendererFeedback feedback;
        feedback.set_client_name(client_name);
        feedback.set_type(protocol::RendererFeedback_Type_Register);
        zmq_protobuf_send(feedback, socket_feedback);

        while(1) {
            protocol::RendererBroadcast msg;
            if(zmq_protobuf_recv(msg, socket_sub) < 0) {
                fprintf(stderr, "render_thread: %s\n", zmq_strerror(zmq_errno()));
                break;
            }

            printf("Renderer: %s\n", protocol::RendererBroadcast_Type_Name(msg.type()).c_str());

            switch(msg.type()) {
                case protocol::RendererBroadcast_Type_LoadVolume: {
                    const void* content = &msg.volume_data()[0];
                    size_t content_length = msg.volume_data().size();
                    volume.reset(VolumeBlocks::LoadFromBuffer(content, content_length));
                    renderer->setVolume(volume.get());
                    printf("Volume: %llu blocks.\n", volume->getBlockCount());
                } break;
                case protocol::RendererBroadcast_Type_LoadVolumeFromFile: {
                    // TODO
                } break;
                case protocol::RendererBroadcast_Type_SetPose: {
                    Pose pose;
                    pose.position = Vector(msg.pose().x(), msg.pose().y(), msg.pose().z());
                    pose.rotation = Quaternion(msg.pose().qw(), msg.pose().qx(), msg.pose().qy(), msg.pose().qz());
                    renderer->setPose(pose);
                    current_pose = pose;
                } break;
                case protocol::RendererBroadcast_Type_SetLensParameters: {
                    for(int i = 0; i < slave->num_projections; i++) {
                        lenses[i]->setEyeSeparation(msg.lens_parameters().eye_separation() * eye);
                        lenses[i]->setFocalDistance(msg.lens_parameters().focal_distance());
                    }
                } break;
                case protocol::RendererBroadcast_Type_SetRendererParameters: {
                    renderer->setBlendingCoefficient(msg.renderer_parameters().blending_coefficient());
                    switch(msg.renderer_parameters().method()) {
                        case protocol::RendererParameters_RenderingMethod_BasicBlending: {
                            renderer->setRaycastingMethod(VolumeRenderer::kBasicBlendingMethod);
                        } break;
                        case protocol::RendererParameters_RenderingMethod_RK4: {
                            renderer->setRaycastingMethod(VolumeRenderer::kRK4Method);
                        } break;
                        case protocol::RendererParameters_RenderingMethod_AdaptiveRKF: {
                            renderer->setRaycastingMethod(VolumeRenderer::kAdaptiveRKFMethod);
                        } break;
                    }
                } break;
                case protocol::RendererBroadcast_Type_SetTransferFunction: {
                    tf->setDomain(msg.transfer_function().domain_min(), msg.transfer_function().domain_max());
                    switch(msg.transfer_function().scale()) {
                        case protocol::TransferFunction_Scale_Linear: {
                            tf->setScale(TransferFunction::kLinearScale);
                        } break;
                        case protocol::TransferFunction_Scale_Log: {
                            tf->setScale(TransferFunction::kLogScale);
                        } break;
                    }
                    TransferFunction::ParseLayers(tf.get(), msg.transfer_function().size(), msg.transfer_function().layers().c_str());
                } break;
                case protocol::RendererBroadcast_Type_SetRGBLevels: {
                    const protocol::RGBLevels& levels = msg.rgb_levels();
                    rgb_levels_min = levels.min();
                    rgb_levels_max = levels.max();
                    rgb_levels_pow = levels.pow();
                } break;
                case protocol::RendererBroadcast_Type_Render: {
                    t_render_scene();
                } break;
                case protocol::RendererBroadcast_Type_Present: {
                    GPURenderThreadEvent event;
                    event.type = GPURenderThreadEvent::kPresent;
                    event.thread_id = thread_id;
                    zmq_msg_t msg;
                    zmq_msg_init_size(&msg, sizeof(GPURenderThreadEvent));
                    memcpy(zmq_msg_data(&msg), &event, sizeof(GPURenderThreadEvent));
                    zmq_msg_send(&msg, socket_push, 0);
                } break;
                case protocol::RendererBroadcast_Type_Barrier: {
                    protocol::RendererFeedback feedback;
                    feedback.set_client_name(client_name);
                    feedback.set_type(protocol::RendererFeedback_Type_BarrierReady);
                    feedback.set_barrier_info(msg.barrier_info());
                    zmq_protobuf_send(feedback, socket_feedback);
                } break;
                case protocol::RendererBroadcast_Type_HDRendering: {
                    const protocol::HDRenderingTask& task = msg.hd_rendering_task();
                    if(task.task_slave() == client_name) {
                        printf("HDRendering: %s\n", task.task_id().c_str());
                        Lens* render_lens;
                        switch(task.lens_type()) {
                            case protocol::HDRenderingTask_LensType_Perspective: {
                                render_lens = Lens::CreatePerspective(task.perspective_fovx());
                            } break;
                            case protocol::HDRenderingTask_LensType_Equirectangular: {
                                render_lens = Lens::CreateEquirectangular();
                            } break;
                        }
                        renderer->setLens(render_lens);
                        VolumeRenderer::RaycastingMethod raycasting_method = renderer->getRaycastingMethod();
                        renderer->setRaycastingMethod(VolumeRenderer::kAdaptiveRKFMethod);
                        renderer->setTransferFunction(tf.get());
                        Image* img = Image::Create(task.task_vp_w(), task.task_vp_h());
                        renderer->setImage(img);
                        renderer->render(task.task_vp_x(), task.task_vp_y(), task.total_width(), task.total_height());
                        Image::LevelsGPU(img, rgb_levels_min, rgb_levels_max, rgb_levels_pow);
                        img->setNeedsDownload();

                        protocol::RendererFeedback feedback;
                        feedback.set_client_name(client_name);
                        feedback.set_type(protocol::RendererFeedback_Type_HDRenderingResponse);

                        protocol::HDRenderingResponse& resp = *feedback.mutable_hd_rendering_response();
                        resp.set_identifier(task.identifier());
                        resp.set_task_id(task.task_id());
                        resp.set_task_vp_x(task.task_vp_x());
                        resp.set_task_vp_y(task.task_vp_y());
                        resp.set_task_vp_w(task.task_vp_w());
                        resp.set_task_vp_h(task.task_vp_h());
                        resp.set_pixel_data(img->getPixels(), sizeof(Color) * task.task_vp_w() * task.task_vp_h());

                        delete img;
                        delete render_lens;

                        zmq_protobuf_send(feedback, socket_feedback);

                        renderer->setRaycastingMethod(raycasting_method);
                    }
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
            Image::LevelsGPU(images[i].get(), rgb_levels_min, rgb_levels_max, rgb_levels_pow);
            // TODO: Perform RGB curve here.
            lenses[i]->performBlend(images[i].get());
            images[i]->setNeedsDownload();
            images[i]->getPixels(); // call getPixels to actually download the pixels from the GPU.
        }
        is_dirty = true;
    }

    // This should be run in main thread.
    void initialize(int gpu_id_, int thread_id_, AllosphereCalibration::RenderSlave* slave_, float eye_) {

        thread_id = thread_id_;
        gpu_id = gpu_id_;
        slave = slave_;
        eye = eye_;
        {
            // Make client_name.
            int pid = getpid();
            char hostname[64] = { '\0' };
            gethostname(hostname, 63);
            sprintf(client_name, "%s:%d:%d", hostname, pid, thread_id);
            printf("Renderer: %s\n", client_name);
        }
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
        if(!initialize_complete) return;
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

    bool initialize_complete;

    float eye;

    boost::shared_ptr<VolumeBlocks> volume;

    pthread_t thread;

    void* socket_push;
    void* socket_sub;
    char client_name[256];
};

class Renderer;

Renderer* single_instance = NULL;

class Renderer {
public:

    enum StereoMode {
        kActiveStereoMode,
        kAnaglyphStereoMode,
        kMonoLeftStereoMode,
        kMonoRightStereoMode,
        kMonoStereoMode
    };

    Renderer() {
        calibration = AllosphereCalibration::Load((string(getenv("HOME")) + "/calibration-current").c_str());
        render_slave = calibration->getRenderer();

        initWindow();

        int device_count;
        cudaGetDeviceCount(&device_count);

        if(stereo_mode == kActiveStereoMode || stereo_mode == kAnaglyphStereoMode) {
            renderer_left.initialize(0 % device_count, 0, render_slave, 1);
            renderer_right.initialize(1 % device_count, 1, render_slave, -1);
            total_threads = 2;
        } else if(stereo_mode == kMonoLeftStereoMode) {
            renderer_left.initialize(0 % device_count, 0, render_slave, 1);
            total_threads = 1;
        } else if(stereo_mode == kMonoRightStereoMode) {
            renderer_right.initialize(0 % device_count, 0, render_slave, -1);
            total_threads = 1;
        } else {
            renderer_left.initialize(0 % device_count, 0, render_slave, 0);
            total_threads = 1;
        }
    }

    static void* subscription_thread_pthread(void* this_) {
        ((Renderer*)this_)->subscription_thread();
    }

    void subscription_thread() {
        void* socket_sub = zmq_socket(zmq_context, ZMQ_SUB);

        zmq_setsockopt_ez(socket_sub, ZMQ_RCVHWM, config.get<int>("sync.zmq.rcvhwm", 10000));
        zmq_setsockopt_ez(socket_sub, ZMQ_RCVBUF, config.get<int>("sync.zmq.rcvbuf", 0));
        zmq_setsockopt_ez(socket_sub, ZMQ_RATE, config.get<int>("sync.zmq.rate", 10000000));

        zmq_connect(socket_sub, config.get<string>("sync.broadcast").c_str());
        zmq_setsockopt(socket_sub, ZMQ_SUBSCRIBE, "", 0);

        void* socket_pub = zmq_socket(zmq_context, ZMQ_PUB);
        zmq_bind(socket_pub, "inproc://render_slaves");

        // Forward messages to our two GPU renderers.
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

    static void display_callback() {
        single_instance->display();
    }

    static void idle_callback() {
    }

    static void keyboard_callback(unsigned char key, int x, int y) {
        if(key == 'q') {
            exit(0);
        }
    }

    void initWindow() {

        bool is_stereo = config.get<string>("allovolume.stereo", "false") == "true";
        bool is_fullscreen = config.get<string>("allovolume.fullscreen", "false") == "true";

        int argc = 0;
        glutInit(&argc, NULL);

        if(is_stereo) {
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_STEREO);
            stereo_mode = kActiveStereoMode;
        } else {
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
            stereo_mode = kMonoStereoMode;
            if(config.get<string>("allovolume.stereo", "false") == "anaglyph")
                stereo_mode = kAnaglyphStereoMode;
            if(config.get<string>("allovolume.stereo", "false") == "left")
                stereo_mode = kMonoLeftStereoMode;
            if(config.get<string>("allovolume.stereo", "false") == "right")
                stereo_mode = kMonoRightStereoMode;
        }

        single_instance = this;

        if(is_fullscreen) {
            int width = glutGet(GLUT_SCREEN_WIDTH);
            int height = glutGet(GLUT_SCREEN_HEIGHT);
            char game_mode_string[64];
            sprintf(game_mode_string, "%dx%d:24", width, height);
            glutGameModeString(game_mode_string);
            glutEnterGameMode();
            glutDisplayFunc(display_callback);
            glutKeyboardFunc(keyboard_callback);
            glutIdleFunc(idle_callback);
        } else {
            glutInitWindowSize(config.get<int>("allovolume.window.width", 600),
                               config.get<int>("allovolume.window.height", 400));
            glutCreateWindow("Allosphere Volume Renderer");
            glutDisplayFunc(display_callback);
            glutKeyboardFunc(keyboard_callback);
            glutIdleFunc(idle_callback);
        }

        // Get info of GPU and supported OpenGL version
        printf("Renderer: %s\n", glGetString(GL_RENDERER));
        printf("OpenGL version supported %s\n", glGetString(GL_VERSION));
    }

    void startup() {
        pthread_create(&thread, NULL, subscription_thread_pthread, this);

        void* socket_pull = zmq_socket(zmq_context, ZMQ_PULL);
        zmq_bind(socket_pull, "inproc://render_slaves_push");
        zmq_setsockopt_ez(socket_pull, ZMQ_RCVTIMEO, 10); // recv timeout = 10ms.

        set<int> barrier_thread;

        while(1) {
            zmq_msg_t msg;
            zmq_msg_init(&msg);
            int r = zmq_msg_recv(&msg, socket_pull, 0);
            if(r >= 0) {
                GPURenderThreadEvent event = *(GPURenderThreadEvent*)zmq_msg_data(&msg);

                barrier_thread.insert(event.thread_id);
                if(barrier_thread.size() == total_threads) {
                    if(event.type == GPURenderThreadEvent::kPresent) {
                        renderer_left.uploadImages();
                        renderer_right.uploadImages();
                        glutPostRedisplay();
                    }
                    barrier_thread.clear();
                }
            }
            zmq_msg_close(&msg);

            #ifdef __APPLE__
                glutCheckLoop();
            #else
                glutMainLoopEvent();
            #endif
        }
    }

    void render_image(GPURenderThread& renderer) {
        if(!renderer.initialize_complete) return;

        int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
        int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);

        if(renderer.textures.size() >= render_slave->num_projections) {
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
    }

    void display() {
        // Draw stuff
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_DEPTH_BUFFER_BIT);

        glEnable(GL_TEXTURE);
        glEnable(GL_TEXTURE_2D);

        switch(stereo_mode) {
            case kActiveStereoMode: {

                glDrawBuffer(GL_BACK_LEFT);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                render_image(renderer_left);

                glDrawBuffer(GL_BACK_RIGHT);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                render_image(renderer_right);

                glDrawBuffer(GL_BACK);

            } break;
            case kAnaglyphStereoMode: {

                glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE);
                glClear(GL_DEPTH_BUFFER_BIT);
                render_image(renderer_left);
                glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE);
                glClear(GL_DEPTH_BUFFER_BIT);
                render_image(renderer_right);

            } break;

            case kMonoLeftStereoMode:
            case kMonoStereoMode: {

                render_image(renderer_left);

            } break;
            case kMonoRightStereoMode: {

                render_image(renderer_right);

            } break;
        }

        glFlush();
        // Update Screen
        glutSwapBuffers();
    }

    AllosphereCalibration* calibration;
    GPURenderThread renderer_left, renderer_right;
    AllosphereCalibration::RenderSlave* render_slave;
    pthread_t thread;
    int total_threads;
    StereoMode stereo_mode;
};

int main(int argc, char* argv[]) {
    zmq_context = zmq_ctx_new();
    config.parseFile("allovolume.yaml");
    char hostname[256];
    gethostname(hostname, 256);
    config.parseFile("allovolume.yaml", hostname);
    config.parseFile("allovolume.yaml", "renderer");
    Renderer renderer;
    renderer.startup();
}
