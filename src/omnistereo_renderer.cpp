#include "allovolume/renderer.h"
#include "allovolume/allosphere_calibration.h"
#include "configparser.h"
#include "timeprofiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <float.h>

#include <string>
#include <vector>

#include <zmq.h>

#include "allovolume_protocol.pb.h"
#include "allovolume_common.h"

#include "allovolume/omnistereo_renderer.h"

#include <cuda_runtime.h>

#include "opengl_include.h"

using namespace std;
using namespace allovolume;

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

enum StereoMode {
    kActiveStereoMode,
    kAnaglyphStereoMode,
    kMonoLeftStereoMode,
    kMonoRightStereoMode,
    kMonoStereoMode
};

struct GPURenderThreadEvent {
    enum Type {
        kPresent,
        kRenderDone
    };
    Type type;
    int thread_id;
};

class GPURenderThread {
public:
    // Note: Each GPU thread has its own CUDA contexts, the pointers are not interchangable.
    int thread_id;

    // Global state.
    boost::shared_ptr<VolumeBlocks> volume;
    VolumeRendererPointer renderer;
    TransferFunctionPointer tf;
    Pose current_pose;
    float rgb_levels_min;
    float rgb_levels_max;
    float rgb_levels_pow;

    class ClipImageData : public boost::noncopyable {
    public:
        ClipImageData(int width, int height) {
            size = width * height;
            data = new VolumeRenderer::ClipRange[width * height];
            for(int i = 0; i < width * height; i++) {
                data[i].t_near = 0;
                data[i].t_front = FLT_MAX;
                data[i].t_far = FLT_MAX;
            }
        }
        ~ClipImageData() {
            delete [] data;
        }

        VolumeRenderer::ClipRange* data;
        int size;
    };

    // For each viewport.
    struct ViewportData {
        AllosphereLensPointer lens;
        ImagePointer image_front;
        ImagePointer image_back;
        boost::shared_ptr<ClipImageData> clip_range;

        int width, height;

        GLuint texture_front, texture_back;

        void allocate(int width_, int height_) {
            width = width_;
            height = height_;
            image_front.reset(Image::Create(width, height));
            image_back.reset(Image::Create(width, height));
            clip_range.reset(new ClipImageData(width, height));
            Color* p = image_front->getPixels();
            for(int i = 0; i < width * height; i++) { p[i] = Color(0, 0, 0, 0); }
            p = image_back->getPixels();
            for(int i = 0; i < width * height; i++) { p[i] = Color(0, 0, 0, 0); }
        }
    };
    vector<ViewportData> viewports;

    AllosphereCalibration::RenderSlave* slave;

    // States.
    bool should_exit;
    bool is_dirty;
    int gpu_id;
    bool needs_upload_tf;
    bool needs_render;

    bool initialize_complete;

    float eye;

    pthread_t thread;
    pthread_mutex_t mutex;

    void lock() {
        pthread_mutex_lock(&mutex);
    }
    void unlock() {
        pthread_mutex_unlock(&mutex);
    }

    void* zmq_context;
    void* socket_push;
    void* socket_sub;
    char client_name[256];

    ConfigParser* config;

    GPURenderThread() {
        initialize_complete = false;
        rgb_levels_min = 0;
        rgb_levels_max = 1;
        rgb_levels_pow = 1;

        pthread_mutex_init(&mutex, 0);
    }

    static void* thread_proc(void* this_) {
        ((GPURenderThread*)this_)->thread_entrypoint();
        return NULL;
    }

    void thread_entrypoint() {
        // Select the GPU device.
        cudaSetDevice(gpu_id);

        // Create new renderer.
        renderer.reset(VolumeRenderer::CreateGPU());

        // Assign default transfer function.
        tf.reset(TransferFunction::CreateGaussianTicks(0, 1, TransferFunction::kLinearScale, 16));

        // Default blending coefficient
        renderer->setBlendingCoefficient(1);

        renderer->setTransferFunction(tf.get());

        // Background color set to transparent (we are using front/back mode, which must use transparent to work).
        renderer->setBackgroundColor(Color(0, 0, 0, 0));

        // Assign lens for each projection, and the corresponding images.
        int viewport_width = config->get<int>("allovolume.resolution.width", 960);
        int viewport_height = config->get<int>("allovolume.resolution.height", 640);
        for(int i = 0; i < slave->num_projections; i++) {
            // Lens for this viewport.
            ViewportData& vp = viewports[i];
            vp.lens.reset(AllosphereCalibration::CreateLens(&slave->projections[i]));
            vp.allocate(viewport_width, viewport_height);
        }
        // Set the len parameters.
        for(int i = 0; i < slave->num_projections; i++) {
            viewports[i].lens->setEyeSeparation(eye * 0.065 / 5.00);
            viewports[i].lens->setFocalDistance(1.0);
        }

        is_dirty = true;

        // Mark that we have finished initialization.
        initialize_complete = true;
        printf("RendererThread: %s initialized. vp:%dx%d\n", client_name, viewport_width, viewport_height);

        // socket_sub: Subscribe the messages from the main renderer thread.
        void* socket_sub = zmq_socket(zmq_context, ZMQ_SUB);
        zmq_connect(socket_sub, "inproc://render_slaves");
        zmq_setsockopt(socket_sub, ZMQ_SUBSCRIBE, "", 0);

        // socket_push: Send messages to the main renderer thread..
        void* socket_push = zmq_socket(zmq_context, ZMQ_PUSH);
        zmq_connect(socket_push, "inproc://render_slaves_push");

        // socket_feedback: Messages to the controller.
        void* socket_feedback = zmq_socket(zmq_context, ZMQ_PUSH);
        zmq_connect(socket_feedback, config->get<string>("sync.feedback").c_str());

        // Register this renderer thread to the controller.
        protocol::RendererFeedback feedback;
        feedback.set_client_name(client_name);
        feedback.set_type(protocol::RendererFeedback_Type_Register);
        zmq_protobuf_send(feedback, socket_feedback);

        while(1) {
            // Receive message from the main rendering thread.
            protocol::RendererBroadcast msg;
            if(zmq_protobuf_recv(msg, socket_sub) < 0) {
                fprintf(stderr, "render_thread: %s\n", zmq_strerror(zmq_errno()));
                break;
            }

            //printf("Renderer: %s\n", protocol::RendererBroadcast_Type_Name(msg.type()).c_str());

            switch(msg.type()) {
                case protocol::RendererBroadcast_Type_LoadVolume: {
                    // Load volume from message data.
                    const void* content = &msg.volume_data()[0];
                    size_t content_length = msg.volume_data().size();
                    volume.reset(VolumeBlocks::LoadFromBuffer(content, content_length));
                    renderer->setVolume(volume.get());
                } break;
                case protocol::RendererBroadcast_Type_LoadVolumeFromFile: {
                    // Load volume from file.
                    volume.reset(VolumeBlocks::LoadFromFile(msg.volume_filename().c_str()));
                    renderer->setVolume(volume.get());
                } break;
                case protocol::RendererBroadcast_Type_SetPose: {
                    // Set pose.
                    Pose pose;
                    pose.position = Vector(msg.pose().x(), msg.pose().y(), msg.pose().z());
                    pose.rotation = Quaternion(msg.pose().qw(), msg.pose().qx(), msg.pose().qy(), msg.pose().qz());
                    renderer->setPose(pose);
                    current_pose = pose;
                } break;
                case protocol::RendererBroadcast_Type_SetLensParameters: {
                    // Set lens parameters.
                    for(int i = 0; i < slave->num_projections; i++) {
                        viewports[i].lens->setEyeSeparation(msg.lens_parameters().eye_separation() * eye);
                        viewports[i].lens->setFocalDistance(msg.lens_parameters().focal_distance());
                    }
                } break;
                case protocol::RendererBroadcast_Type_SetRendererParameters: {
                    // Renderer parameters.
                    renderer->setBlendingCoefficient(msg.renderer_parameters().blending_coefficient());
                    renderer->setStepSizeMultiplier(msg.renderer_parameters().step_size());
                    renderer->setEnableZIndex(msg.renderer_parameters().enable_z_index());
                    // Raycasting method.
                    switch(msg.renderer_parameters().method()) {
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
                    // Internal format.
                    switch(msg.renderer_parameters().internal_format()) {
                        case protocol::RendererParameters_InternalFormat_Float32: {
                            renderer->setInternalFormat(VolumeRenderer::kFloat32);
                        } break;
                        case protocol::RendererParameters_InternalFormat_UInt16: {
                            renderer->setInternalFormat(VolumeRenderer::kUInt16);
                        } break;
                        case protocol::RendererParameters_InternalFormat_UInt8: {
                            renderer->setInternalFormat(VolumeRenderer::kUInt8);
                        } break;
                    }
                } break;
                case protocol::RendererBroadcast_Type_SetTransferFunction: {
                    // Set transfer function.
                    tf->setDomain(msg.transfer_function().domain_min(), msg.transfer_function().domain_max());
                    switch(msg.transfer_function().scale()) {
                        case protocol::TransferFunction_Scale_Linear: {
                            tf->setScale(TransferFunction::kLinearScale);
                        } break;
                        case protocol::TransferFunction_Scale_Log: {
                            tf->setScale(TransferFunction::kLogScale);
                        } break;
                    }
                    // Generate transfer function.
                    TransferFunction::ParseLayers(tf.get(), msg.transfer_function().size(), msg.transfer_function().layers().c_str());
                    renderer->setTransferFunction(tf.get());
                } break;
                case protocol::RendererBroadcast_Type_SetRGBLevels: {
                    // Set RGB levels.
                    const protocol::RGBLevels& levels = msg.rgb_levels();
                    rgb_levels_min = levels.min();
                    rgb_levels_max = levels.max();
                    rgb_levels_pow = levels.pow();
                } break;
                case protocol::RendererBroadcast_Type_Render: {
                    // Render the scene.
                    thread_render_scene();
                    // Present the rendered results.
                    GPURenderThreadEvent event;
                    event.type = GPURenderThreadEvent::kRenderDone;
                    event.thread_id = thread_id;
                    // Send the present message to the main render thread.
                    zmq_msg_t msg;
                    zmq_msg_init_size(&msg, sizeof(GPURenderThreadEvent));
                    memcpy(zmq_msg_data(&msg), &event, sizeof(GPURenderThreadEvent));
                    zmq_msg_send(&msg, socket_push, 0);
                } break;
                case protocol::RendererBroadcast_Type_Present: {
                    // Present the rendered results.
                    GPURenderThreadEvent event;
                    event.type = GPURenderThreadEvent::kPresent;
                    event.thread_id = thread_id;
                    // Send the present message to the main render thread.
                    zmq_msg_t msg;
                    zmq_msg_init_size(&msg, sizeof(GPURenderThreadEvent));
                    memcpy(zmq_msg_data(&msg), &event, sizeof(GPURenderThreadEvent));
                    zmq_msg_send(&msg, socket_push, 0);
                } break;
                case protocol::RendererBroadcast_Type_Barrier: {
                    // Barrier.
                    protocol::RendererFeedback feedback;
                    feedback.set_client_name(client_name);
                    feedback.set_type(protocol::RendererFeedback_Type_BarrierReady);
                    feedback.set_barrier_info(msg.barrier_info());
                    zmq_protobuf_send(feedback, socket_feedback);
                } break;
                case protocol::RendererBroadcast_Type_HDRendering: {
                    // HDRendering task handling.
                    const protocol::HDRenderingTask& task = msg.hd_rendering_task();
                    if(task.task_slave() == client_name) {
                        fprintf(stderr, "HDRendering: %s\n", task.task_id().c_str());
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

    void thread_render_scene() {
        if(!volume) return;
        for(int i = 0; i < slave->num_projections; i++) {
            const ViewportData& vp = viewports[i];
            // Set the lens.
            renderer->setRaycastingMethod(VolumeRenderer::kPreIntegrationMethod);
            renderer->setLens(vp.lens.get());
            lock();
            renderer->setClipRanges(vp.clip_range->data, vp.clip_range->size);
            unlock();
            // Set image: front and back.
            renderer->setImage(vp.image_front.get());
            renderer->setBackImage(vp.image_back.get());
            // Render.
            renderer->render();
        }
        // Call getPixels to actually download the pixels from the GPU.
        lock();
        for(int i = 0; i < slave->num_projections; i++) {
            const ViewportData& vp = viewports[i];
            vp.image_front->setNeedsDownload();
            vp.image_front->getPixels();
            vp.image_back->setNeedsDownload();
            vp.image_back->getPixels();
        }
        unlock();
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
            fprintf(stderr, "Renderer Initialize: %s\n", client_name);
        }
        // Create the textures.
        viewports.resize(slave->num_projections);
        for(int i = 0; i < slave->num_projections; i++) {
            GLuint tex;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            viewports[i].texture_front = tex;

            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            viewports[i].texture_back = tex;
        }
        is_dirty = true;
        // Spawn the rendering thread.
        pthread_create(&thread, NULL, thread_proc, this);
    }

    void uploadImages() {
        if(!initialize_complete) return;

        if(is_dirty && !viewports.empty()) {
            lock();
            for(int i = 0; i < slave->num_projections; i++) {
                const ViewportData& vp = viewports[i];
                glBindTexture(GL_TEXTURE_2D, vp.texture_front);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, vp.image_front->getWidth(), vp.image_front->getHeight(), 0,
                    GL_RGBA, GL_FLOAT, vp.image_front->getPixels());
                glBindTexture(GL_TEXTURE_2D, vp.texture_back);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, vp.image_back->getWidth(), vp.image_back->getHeight(), 0,
                    GL_RGBA, GL_FLOAT, vp.image_back->getPixels());
            }
            unlock();
            is_dirty = false;
        }
    }

    ~GPURenderThread() {
        should_exit = true;
        pthread_join(thread, NULL);
        pthread_mutex_destroy(&mutex);
    }
};

GLuint compileShaderProgram(const std::string& vertex_code, const std::string& fragment_code) {
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    const GLchar* v_source = (const GLchar*)vertex_code.c_str();
    glShaderSource(vertex_shader, 1, &v_source, 0);
    glCompileShader(vertex_shader);
    GLint isCompiled = 0;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &isCompiled);
    if(isCompiled == GL_FALSE) {
        GLint maxLength = 0;
        glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &maxLength);
        // The maxLength includes the NULL character
        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(vertex_shader, maxLength, &maxLength, &infoLog[0]);
        // We don't need the shader anymore.
        glDeleteShader(vertex_shader);
        // Use the infoLog as you see fit.
        fprintf(stderr, "Error compile vertex shader:\n%s\n", &infoLog[0]);
        return 0;
    }

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar* f_source = (const GLchar*)fragment_code.c_str();
    glShaderSource(fragment_shader, 1, &f_source, 0);
    glCompileShader(fragment_shader);
    isCompiled = 0;
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &isCompiled);
    if(isCompiled == GL_FALSE) {
        GLint maxLength = 0;
        glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &maxLength);
        // The maxLength includes the NULL character
        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(fragment_shader, maxLength, &maxLength, &infoLog[0]);
        // We don't need the shader anymore.
        glDeleteShader(fragment_shader);
        glDeleteShader(vertex_shader);
        // Use the infoLog as you see fit.
        fprintf(stderr, "Error compile fragment shader:\n%s\n", &infoLog[0]);
        return 0;
    }

    GLuint program = glCreateProgram();

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    // Note the different functions here: glGetProgram* instead of glGetShader*.
    GLint isLinked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, (int *)&isLinked);
    if(isLinked == GL_FALSE) {
        GLint maxLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);
        // The maxLength includes the NULL character
        std::vector<GLchar> infoLog(maxLength);
        glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);
        // We don't need the program anymore.
        glDeleteProgram(program);
        // Don't leak shaders either.
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        fprintf(stderr, "Error linking program:\n%s\n", &infoLog[0]);
        return 0;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return program;
}

#define STRINGIFY(s) #s

const char* kGLSL_loadDepthCubemap_vertex = STRINGIFY(
    varying vec2 T;
    void main(void) {
        // Pass through the texture coordinate (normalized pixel):
        T = vec2(gl_MultiTexCoord0);
        gl_Position = vec4(T * 2.0 - 1.0, 0.0, 1.0);
        gl_Position.y = -gl_Position.y;
    }
);

const char* kGLSL_loadDepthCubemap_fragment = STRINGIFY(
    uniform sampler2D pixel_map;
    uniform samplerCube depth_cube_map;
    uniform float omni_near;
    uniform float omni_far;
    uniform float scale_factor;

    varying vec2 T;

    void main (void) {
        // ray location (calibration space):
        vec3 v = normalize(texture2D(pixel_map, T).rgb);
        // index into cubemap:
        float zb = textureCube(depth_cube_map, v.xyz).r;
        float depth = omni_far * omni_near / (omni_far + zb * (omni_near - omni_far));
        vec3 absv = abs(v);
        if(absv.x >= absv.y && absv.x >= absv.z) {  // x is the biggest.
            v /= absv.x;
        } else if(absv.y >= absv.x && absv.y >= absv.z) {
            v /= absv.y;
        } else {
            v /= absv.z;
        }
        depth *= length(v);

        gl_FragColor.rgba = vec4(omni_near * scale_factor, depth * scale_factor, omni_far * scale_factor, 1.0);
    }
);

class OmnistereoRendererImpl : public OmnistereoRenderer {
public:

    OmnistereoRendererImpl() {
        zmq_context = zmq_ctx_new();
        renderer_left.zmq_context = zmq_context;
        renderer_right.zmq_context = zmq_context;
        renderer_left.config = &config;
        renderer_right.config = &config;
        load_depth_cubemap_program = 0;
    }

    void loadConfigurationFile(const char* path) {
        config.parseFile(path);
        char hostname[256];
        gethostname(hostname, 256);
        config.parseFile(path, hostname);
        config.parseFile(path, "renderer");

        string calibration_dir = string(getenv("HOME")) + "/calibration-current";
        calibration = AllosphereCalibration::Load(config.get<string>("allosphere.calibration", calibration_dir).c_str());

        render_slave = calibration->getRenderer();

        int device_count;
        cudaGetDeviceCount(&device_count);

        bool is_stereo = config.get<string>("allovolume.stereo", "false") == "true";
        if(is_stereo) {
            stereo_mode = kActiveStereoMode;
        } else {
            stereo_mode = kMonoStereoMode;
        }

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

        load_depth_cubemap_program = compileShaderProgram(kGLSL_loadDepthCubemap_vertex, kGLSL_loadDepthCubemap_fragment);
        glUseProgram(load_depth_cubemap_program);
        glUniform1i(glGetUniformLocation(load_depth_cubemap_program, "pixel_map"), 1);
        glUniform1i(glGetUniformLocation(load_depth_cubemap_program, "depth_cube_map"), 0);
        glUseProgram(0);

        viewport_width = config.get<int>("allovolume.resolution.width", 960);
        viewport_height = config.get<int>("allovolume.resolution.height", 640);

        glGenFramebuffers(1, &load_depth_cubemap_framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, load_depth_cubemap_framebuffer);

        GLuint colorrenderbuffer;
        glGenRenderbuffers(1, &colorrenderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, colorrenderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB32F, viewport_width, viewport_height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colorrenderbuffer);

        GLenum status;
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if(status != GL_FRAMEBUFFER_COMPLETE) {
            printf("Framebuffer incomplete: %d\n", status);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    static void* subscription_thread_pthread(void* this_) {
        ((OmnistereoRendererImpl*)this_)->subscription_thread();
        return NULL;
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
            { // Inspect:
                int size = zmq_msg_size(&msg);
                if(size < 256) {
                    protocol::RendererBroadcast message;
                    message.ParseFromString(string((char*)zmq_msg_data(&msg), zmq_msg_size(&msg)));
                    if(message.type() == protocol::RendererBroadcast_Type_Render) {
                        if(delegate) delegate->onBeforeRender();
                    }
                }
            }

            zmq_msg_send(&msg, socket_pub, 0);
        }
    }

    static void* main_thread_pthread(void* this_) {
        ((OmnistereoRendererImpl*)this_)->main_thread();
        return NULL;
    }

    void main_thread() {
        void* socket_pull = zmq_socket(zmq_context, ZMQ_PULL);
        zmq_bind(socket_pull, "inproc://render_slaves_push");
        // zmq_setsockopt_ez(socket_pull, ZMQ_RCVTIMEO, 10); // recv timeout = 10ms.

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
                        if(delegate) delegate->onPresent();
                    }
                    if(event.type == GPURenderThreadEvent::kRenderDone) {
                        if(delegate) delegate->onRender();
                    }
                    barrier_thread.clear();
                }
            }
            zmq_msg_close(&msg);
        }
    }

    void startup() {
        pthread_create(&thread, NULL, subscription_thread_pthread, this);
        pthread_create(&thread_main, NULL, main_thread_pthread, this);
    }

    void loadDepthCubemapRenderer(GPURenderThread& renderer, unsigned int texDepth, float near, float far, float omni_eyesep) {
        // Use the depth cubemap to clip range shader.
        glUseProgram(load_depth_cubemap_program);
        glUniform1f(glGetUniformLocation(load_depth_cubemap_program, "omni_near"), near);
        glUniform1f(glGetUniformLocation(load_depth_cubemap_program, "omni_far"), far);
        float scale_factor = 1.0;
        if(omni_eyesep != 0) {
            scale_factor = fabs(renderer.viewports[0].lens->getEyeSeparation()) / fabs(omni_eyesep);
        }
        glUniform1f(glGetUniformLocation(load_depth_cubemap_program, "scale_factor"), scale_factor);

        // Bind the framebuffer.
        glBindFramebuffer(GL_FRAMEBUFFER, load_depth_cubemap_framebuffer);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glViewport(0, 0, viewport_width, viewport_height);

        glActiveTexture(GL_TEXTURE0);
        glEnable(GL_TEXTURE_CUBE_MAP);
        glBindTexture(GL_TEXTURE_CUBE_MAP, texDepth);

        for(int vp_idx = 0; vp_idx < renderer.viewports.size(); vp_idx++) {
            if(!renderer.initialize_complete) continue;
            const GPURenderThread::ViewportData& vp = renderer.viewports[vp_idx];
            GLuint wrap_texture = vp.lens->getWrapTexture();

            glActiveTexture(GL_TEXTURE1);
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, wrap_texture);

            // double tt = (TimeProfiler::Default()->getTime() - 1429829725) * 1000;
            // Draw the stuff.
            // glClearColor(tt, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT);

            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex3f(-1, -1, 0);
            glTexCoord2f(0, 1); glVertex3f(-1, +1, 0);
            glTexCoord2f(1, 1); glVertex3f(+1, +1, 0);
            glTexCoord2f(1, 0); glVertex3f(+1, -1, 0);
            glEnd();

            // Read pixels.
            // double t0 = TimeProfiler::Default()->getTime();

            renderer.lock();
            glReadPixels(0, 0, viewport_width, viewport_height, GL_RGB, GL_FLOAT, vp.clip_range->data);
            renderer.unlock();

            //int idx_middle = viewport_width / 2 + viewport_height / 2 * viewport_width;
            //printf("t:%d vp:%d wrap=%d = %lu, tt=%f, middle=%f\n", renderer.gpu_id, vp_idx, wrap_texture, vp.clip_range->data, tt, vp.clip_range->data[idx_middle].t_front);

            // Write the image to file.
            /*
            fprintf(stderr, "File writing.\n");
            char filename[64];
            sprintf(filename, "depthRanges-t%d-vp%d.bin", renderer.gpu_id, vp_idx);
            FILE* fp = fopen(filename, "wb");
            fwrite(vp.clip_range->data, sizeof(VolumeRenderer::ClipRange), vp.clip_range->size, fp);
            fclose(fp);
            fprintf(stderr, "File wrote.\n");
            */

            // double t1 = TimeProfiler::Default()->getTime();
            // int err = glGetError();
            // printf("getTexImage2D: %.3lf ms err=%d\n", (t1 - t0) * 1000, err);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(0);
    }

    // Set the cubemap for depth buffer.
    virtual void loadDepthCubemap(unsigned int texDepth_left, unsigned int texDepth_right, float near, float far, float omni_eyesep) {
        glPushAttrib(GL_ALL_ATTRIB_BITS);

        glDepthMask(GL_FALSE);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

        loadDepthCubemapRenderer(renderer_left, texDepth_left, near, far, omni_eyesep);
        if(total_threads >= 2) {
            loadDepthCubemapRenderer(renderer_right, texDepth_right, near, far, omni_eyesep);
        }
        glPopAttrib();
    }

    // Upload viewport textures.
    virtual void uploadTextures() {
        renderer_left.uploadImages();
        if(total_threads == 2) {
            renderer_right.uploadImages();
        }
    }

    // Get viewport textures.
    virtual Textures getTextures(int viewport, int eye) {
        GPURenderThread* r_eyes[2];
        if(stereo_mode == kActiveStereoMode || stereo_mode == kAnaglyphStereoMode) {
            r_eyes[0] = &renderer_left;
            r_eyes[1] = &renderer_right;
        } else if(stereo_mode == kMonoLeftStereoMode) {
            r_eyes[0] = &renderer_left;
            r_eyes[1] = &renderer_left;
        } else if(stereo_mode == kMonoRightStereoMode) {
            r_eyes[0] = &renderer_right;
            r_eyes[1] = &renderer_right;
        } else {
            r_eyes[0] = &renderer_left;
            r_eyes[1] = &renderer_left;
        }
        Textures result;
        result.front = r_eyes[eye]->viewports[viewport].texture_front;
        result.back = r_eyes[eye]->viewports[viewport].texture_back;
        return result;
    }

    virtual bool isReady() {
        if(total_threads == 1) {
            return renderer_left.initialize_complete;
        } else {
            return renderer_left.initialize_complete && renderer_right.initialize_complete;
        }
    }

    // Set the delegate.
    virtual void setDelegate(Delegate* delegate_) {
        delegate = delegate_;
    }

    ~OmnistereoRendererImpl() {
        zmq_ctx_destroy(zmq_context);
    }


    AllosphereCalibration* calibration;
    GPURenderThread renderer_left, renderer_right;
    AllosphereCalibration::RenderSlave* render_slave;
    pthread_t thread;
    pthread_t thread_main;
    int total_threads;
    StereoMode stereo_mode;

    int viewport_width, viewport_height;

    ConfigParser config;
    void* zmq_context;

    GLuint load_depth_cubemap_program, load_depth_cubemap_render_texture, load_depth_cubemap_framebuffer;

    Delegate* delegate;
};

OmnistereoRenderer* OmnistereoRenderer::CreateWithYAMLConfig(const char* path) {
    TimeProfiler::Default()->setDelegate(TimeProfiler::STDERR_DELEGATE);
    OmnistereoRendererImpl* impl = new OmnistereoRendererImpl();
    impl->loadConfigurationFile(path);
    impl->startup();
    return impl;
}

void OmnistereoRenderer::Delegate::onBeforeRender() { }
void OmnistereoRenderer::Delegate::onRender() { }
void OmnistereoRenderer::Delegate::onPresent() { }
OmnistereoRenderer::Delegate::~Delegate() { }

OmnistereoRenderer::~OmnistereoRenderer() { }
