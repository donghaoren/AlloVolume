#include "renderer.h"
#include "configparser.h"
#include "allosphere/allosphere_calibration.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <readline/readline.h>

#include <string>
#include <vector>
#include <fstream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <zmq.h>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>

#include "allovolume_protocol.pb.h"

#include "allovolume_common.h"

using namespace std;
using namespace allovolume;

ConfigParser config;
void* zmq_context;

double getPreciseTime() {
    timeval t;
    gettimeofday(&t, 0);
    double s = t.tv_sec;
    s += t.tv_usec / 1000000.0;
    return s;
}

class Controller {
public:
    pthread_t thread;
    void* socket_pubsub;
    void* socket_feedback;
    void* socket_events;
    bool needs_render;
    bool is_rendering;

    enum BarrierInfo {
        kRenderPresentBarrier
    };

    // The current state.
    AlloVolumeState state;
    string dataset_name;

    set<string> barrier_clients;
    set<string> all_clients;

    class HDRenderingTask {
    public:
        set<string> waiting;
        Color* pixels;
        int width, height;
        string output_filename;

        HDRenderingTask() {
            pixels = NULL;
        }
        void init(int width_, int height_) {
            width = width_;
            height = height_;
            pixels = new Color[width * height];
        }
        ~HDRenderingTask() {
            if(pixels) delete [] pixels;
        }
    private:
        HDRenderingTask(const HDRenderingTask&) { }
        HDRenderingTask& operator = (const HDRenderingTask&) { }
    };

    map<string, boost::shared_ptr<HDRenderingTask> > hd_rendering_tasks;
    // set<string> hd_rendering_tasks_waiting;
    // Color* hd_rendering_pixels;
    // int hd_rendering_width, hd_rendering_height;
    // string hd_rendering_output_filename;

    Controller() {
        needs_render = false;
        is_rendering = false;

        register_controller_requests();

        pthread_create(&thread, NULL, thread_process_pthread, this);
    }

    static void* thread_process_pthread(void* this_) {
        ((Controller*)this_)->thread_process();
    }

    void set_needs_render() {
        if(all_clients.size() == 0) return;
        if(is_rendering) {
            needs_render = true;
        } else {
            is_rendering = true;
            {
                protocol::RendererBroadcast msg;
                msg.set_type(protocol::RendererBroadcast_Type_Render);
                zmq_protobuf_send(msg, socket_pubsub);
            }
            {
                protocol::RendererBroadcast msg;
                msg.set_type(protocol::RendererBroadcast_Type_Barrier);
                msg.set_barrier_info(kRenderPresentBarrier);
                zmq_protobuf_send(msg, socket_pubsub);
            }
            printf("Render request sent.\n");
            needs_render = false;
        }
    }

    void process_renderer_feedback(protocol::RendererFeedback& feedback) {
        // Process the feedback message.
        switch(feedback.type()) {
            case protocol::RendererFeedback_Type_Register: {

                all_clients.insert(feedback.client_name());
                printf("Renderer connected: %s\n", feedback.client_name().c_str());

            } break;
            case protocol::RendererFeedback_Type_BarrierReady: {

                barrier_clients.insert(feedback.client_name());
                if(barrier_clients.size() >= all_clients.size()) {
                    barrier_clients.clear();
                    // Barrier cleared.
                    switch(feedback.barrier_info()) {
                        case kRenderPresentBarrier: {
                            // Present!
                            protocol::RendererBroadcast msg;
                            msg.set_type(protocol::RendererBroadcast_Type_Present);
                            zmq_protobuf_send(msg, socket_pubsub);
                            is_rendering = false;
                            if(needs_render) set_needs_render();
                        } break;
                    }
                }

            } break;
            case protocol::RendererFeedback_Type_HDRenderingResponse: {

                const protocol::HDRenderingResponse& resp = feedback.hd_rendering_response();

                if(hd_rendering_tasks.find(resp.identifier()) != hd_rendering_tasks.end()) {
                    HDRenderingTask& task = *hd_rendering_tasks[resp.identifier()].get();

                    Color* task_pixels = (Color*)&resp.pixel_data()[0];
                    int vp_x = resp.task_vp_x();
                    int vp_y = resp.task_vp_y();
                    int vp_w = resp.task_vp_w();
                    int vp_h = resp.task_vp_h();
                    for(int y = 0; y < vp_h; y++) {
                        for(int x = 0; x < vp_w; x++) {
                            int px = x + vp_x;
                            int py = y + vp_y;
                            task.pixels[py * task.width + px] = task_pixels[y * vp_w + x];
                        }
                    }

                    task.waiting.erase(resp.task_id());

                    if(task.waiting.empty()) {
                        Image::WriteImageFile(task.output_filename.c_str(), "png16", task.width, task.height, task.pixels);
                        hd_rendering_tasks.erase(resp.identifier());
                        protocol::ParameterChangeEvent event;
                        event.set_sender("controller");
                        event.set_type(protocol::ParameterChangeEvent_Type_HDRenderingComplete);
                        event.set_hd_rendering_filename(task.output_filename);
                        zmq_protobuf_send(event, socket_events);
                    }
                }

            } break;
        }
    }

    void RequestHandler_LoadVolume(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        dataset_name = request.volume_dataset();

        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_Type_LoadVolume);
        VolumeBlocks* volume = VolumeBlocks::LoadFromFile(request.volume_filename().c_str());
        size_t length = VolumeBlocks::WriteToBufferSize(volume);
        unsigned char* volume_data = (unsigned char*)malloc(length);
        VolumeBlocks::WriteToBuffer(volume, volume_data, length);
        delete volume;
        msg.set_volume_data(volume_data, length);
        free(volume_data);
        zmq_protobuf_send(msg, socket_pubsub);

        set_needs_render();
    }

    void RequestHandler_LoadVolumeFromFile(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        dataset_name = request.volume_dataset();

        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_Type_LoadVolumeFromFile);
        msg.set_volume_filename(request.volume_filename());
        zmq_protobuf_send(msg, socket_pubsub);

        set_needs_render();
    }

    void RequestHandler_SetPose(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        state.pose = request.pose();

        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_Type_SetPose);
        *msg.mutable_pose() = state.pose;
        zmq_protobuf_send(msg, socket_pubsub);

        protocol::ParameterChangeEvent event;
        event.set_sender(request.sender());
        event.set_type(protocol::ParameterChangeEvent_Type_SetPose);
        *event.mutable_pose() = request.pose();
        zmq_protobuf_send(event, socket_events);

        set_needs_render();
    }
    void RequestHandler_GetPose(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        *response.mutable_pose() = state.pose;
    }

    void RequestHandler_SetTransferFunction(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        state.transfer_function = request.transfer_function();

        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_Type_SetTransferFunction);
        *msg.mutable_transfer_function() = request.transfer_function();
        zmq_protobuf_send(msg, socket_pubsub);

        protocol::ParameterChangeEvent event;
        event.set_sender(request.sender());
        event.set_type(protocol::ParameterChangeEvent_Type_SetTransferFunction);
        *event.mutable_transfer_function() = request.transfer_function();
        zmq_protobuf_send(event, socket_events);

        set_needs_render();
    }
    void RequestHandler_GetTransferFunction(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        *response.mutable_transfer_function() = state.transfer_function;
    }

    void RequestHandler_SetRGBLevels(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        state.rgb_levels = request.rgb_levels();

        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_Type_SetRGBLevels);
        *msg.mutable_rgb_levels() = request.rgb_levels();
        zmq_protobuf_send(msg, socket_pubsub);

        protocol::ParameterChangeEvent event;
        event.set_sender(request.sender());
        event.set_type(protocol::ParameterChangeEvent_Type_SetRGBLevels);
        *event.mutable_rgb_levels() = request.rgb_levels();
        zmq_protobuf_send(event, socket_events);

        set_needs_render();
    }
    void RequestHandler_GetRGBLevels(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        *response.mutable_rgb_levels() = state.rgb_levels;
    }

    void RequestHandler_SetRendererParameters(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        state.renderer_parameters = request.renderer_parameters();

        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_Type_SetRendererParameters);
        *msg.mutable_renderer_parameters() = request.renderer_parameters();
        zmq_protobuf_send(msg, socket_pubsub);

        protocol::ParameterChangeEvent event;
        event.set_sender(request.sender());
        event.set_type(protocol::ParameterChangeEvent_Type_SetRendererParameters);
        *event.mutable_renderer_parameters() = request.renderer_parameters();
        zmq_protobuf_send(event, socket_events);

        set_needs_render();
    }
    void RequestHandler_GetRendererParameters(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        *response.mutable_renderer_parameters() = state.renderer_parameters;
    }

    void RequestHandler_SetLensParameters(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        state.lens_parameters = request.lens_parameters();

        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_Type_SetLensParameters);
        *msg.mutable_lens_parameters() = request.lens_parameters();
        zmq_protobuf_send(msg, socket_pubsub);

        protocol::ParameterChangeEvent event;
        event.set_sender(request.sender());
        event.set_type(protocol::ParameterChangeEvent_Type_SetLensParameters);
        *event.mutable_lens_parameters() = request.lens_parameters();
        zmq_protobuf_send(event, socket_events);

        set_needs_render();
    }
    void RequestHandler_GetLensParameters(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        *response.mutable_lens_parameters() = state.lens_parameters;
    }

    void RequestHandler_SavePreset(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        protocol::ParameterPreset preset;

        preset.set_dataset(dataset_name);
        preset.set_timestamp(getPreciseTime() * 1000);
        preset.set_name(request.preset_name());
        preset.set_description(request.preset_description());

        *preset.mutable_transfer_function() = state.transfer_function;
        *preset.mutable_pose() = state.pose;
        *preset.mutable_rgb_levels() = state.rgb_levels;
        *preset.mutable_renderer_parameters() = state.renderer_parameters;
        *preset.mutable_lens_parameters() = state.lens_parameters;

        std::fstream output((string("presets/") + request.preset_name()).c_str(), std::ios::out | std::ios::binary);
        preset.SerializeToOstream(&output);
    }

    void RequestHandler_LoadPreset(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        protocol::ParameterPreset preset;
        try {
            std::fstream input((string("presets/") + request.preset_name()).c_str(), std::ios::in | std::ios::binary);
            preset.ParseFromIstream(&input);
        } catch(...) {
            response.set_status("E_NOT_FOUND");
            return;
        }
        {
            state.pose = preset.pose();
            protocol::RendererBroadcast msg;
            msg.set_type(protocol::RendererBroadcast_Type_SetPose);
            *msg.mutable_pose() = state.pose;
            zmq_protobuf_send(msg, socket_pubsub);

            protocol::ParameterChangeEvent event;
            event.set_sender(request.sender());
            event.set_type(protocol::ParameterChangeEvent_Type_SetPose);
            *event.mutable_pose() = state.pose;
            zmq_protobuf_send(event, socket_events);
        }
        {
            state.lens_parameters = preset.lens_parameters();
            protocol::RendererBroadcast msg;
            msg.set_type(protocol::RendererBroadcast_Type_SetLensParameters);
            *msg.mutable_lens_parameters() = state.lens_parameters;
            zmq_protobuf_send(msg, socket_pubsub);

            protocol::ParameterChangeEvent event;
            event.set_sender(request.sender());
            event.set_type(protocol::ParameterChangeEvent_Type_SetLensParameters);
            *event.mutable_lens_parameters() = state.lens_parameters;
            zmq_protobuf_send(event, socket_events);
        }
        {
            state.renderer_parameters = preset.renderer_parameters();
            protocol::RendererBroadcast msg;
            msg.set_type(protocol::RendererBroadcast_Type_SetRendererParameters);
            *msg.mutable_renderer_parameters() = state.renderer_parameters;
            zmq_protobuf_send(msg, socket_pubsub);

            protocol::ParameterChangeEvent event;
            event.set_sender(request.sender());
            event.set_type(protocol::ParameterChangeEvent_Type_SetRendererParameters);
            *event.mutable_renderer_parameters() = state.renderer_parameters;
            zmq_protobuf_send(event, socket_events);
        }
        {
            state.transfer_function = preset.transfer_function();
            protocol::RendererBroadcast msg;
            msg.set_type(protocol::RendererBroadcast_Type_SetTransferFunction);
            *msg.mutable_transfer_function() = state.transfer_function;
            zmq_protobuf_send(msg, socket_pubsub);

            protocol::ParameterChangeEvent event;
            event.set_sender(request.sender());
            event.set_type(protocol::ParameterChangeEvent_Type_SetTransferFunction);
            *event.mutable_transfer_function() = state.transfer_function;
            zmq_protobuf_send(event, socket_events);
        }
        {
            state.rgb_levels = preset.rgb_levels();
            protocol::RendererBroadcast msg;
            msg.set_type(protocol::RendererBroadcast_Type_SetRGBLevels);
            *msg.mutable_rgb_levels() = state.rgb_levels;
            zmq_protobuf_send(msg, socket_pubsub);

            protocol::ParameterChangeEvent event;
            event.set_sender(request.sender());
            event.set_type(protocol::ParameterChangeEvent_Type_SetRGBLevels);
            *event.mutable_rgb_levels() = state.rgb_levels;
            zmq_protobuf_send(event, socket_events);
        }

        set_needs_render();
    }

    void RequestHandler_ListPresets(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        boost::filesystem::directory_iterator iter("presets"), iter_end;
        for(; iter != iter_end; ++iter) {
            if(boost::filesystem::is_regular_file(iter->path())) {
                string filename = iter->path().filename().string();
                *response.add_preset_list() = filename;
            }
        }
    }

    void RequestHandler_GetImage(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        FILE* fin = fopen(request.image_filename().c_str(), "rb");
        if(!fin) return;
        fseeko(fin, 0, SEEK_END);
        size_t file_size = ftello(fin);
        fseeko(fin, 0, SEEK_SET);
        unsigned char* data = new unsigned char[file_size];
        fread(data, 1, file_size, fin);
        fclose(fin);
        response.set_binary_data(data, file_size);
        delete [] data;
    }

    void RequestHandler_HDRendering(protocol::ControllerRequest& request, protocol::ControllerResponse& response) {
        const protocol::HDRenderingTask& task = request.hd_rendering_task();
        int block_width = 500;
        int block_height = 500;
        int total_width = task.total_width();
        int total_height = task.total_height();

        static int hd_rendering_index = 0;

        hd_rendering_index++;
        char identifier[64];
        sprintf(identifier, "task.%d", hd_rendering_index);

        hd_rendering_tasks[identifier].reset(new HDRenderingTask());
        HDRenderingTask& task_struct = *hd_rendering_tasks[identifier].get();

        vector<string> slaves(all_clients.begin(), all_clients.end());
        if(slaves.size() == 0) {
            printf("Unable to perform HDRendering: No renderer connected!\n");
            return;
        }
        int choice_index = 0;

        for(int x = 0; x < total_width; x += block_width) {
            for(int y = 0; y < total_height; y += block_height) {
                int width = min(block_width, total_width - x);
                int height = min(block_height, total_height - y);
                char task_id[64];
                sprintf(task_id, "t.%d.%d.%d.%d", x, y, width, height);
                protocol::HDRenderingTask distributed_task = task;
                distributed_task.set_task_vp_x(x);
                distributed_task.set_task_vp_y(y);
                distributed_task.set_task_vp_w(width);
                distributed_task.set_task_vp_h(height);
                distributed_task.set_identifier(identifier);
                distributed_task.set_task_id(task_id);
                distributed_task.set_task_slave(slaves[choice_index]);
                choice_index = (choice_index + 1) % slaves.size();

                task_struct.waiting.insert(task_id);

                protocol::RendererBroadcast msg;
                msg.set_type(protocol::RendererBroadcast_Type_HDRendering);
                *msg.mutable_hd_rendering_task() = distributed_task;
                zmq_protobuf_send(msg, socket_pubsub);
            }
        }
        task_struct.init(total_width, total_height);
        task_struct.output_filename = task.output_filename();
    }

    typedef void (Controller::*controller_handler_t)(protocol::ControllerRequest&, protocol::ControllerResponse&);

    std::map<protocol::ControllerRequest_Type, controller_handler_t> controller_request_handlers;
    void register_controller_requests() {
        controller_request_handlers[protocol::ControllerRequest_Type_LoadVolume] = &Controller::RequestHandler_LoadVolume;
        controller_request_handlers[protocol::ControllerRequest_Type_LoadVolumeFromFile] = &Controller::RequestHandler_LoadVolumeFromFile;
        controller_request_handlers[protocol::ControllerRequest_Type_SetPose] = &Controller::RequestHandler_SetPose;
        controller_request_handlers[protocol::ControllerRequest_Type_GetPose] = &Controller::RequestHandler_GetPose;
        controller_request_handlers[protocol::ControllerRequest_Type_SetTransferFunction] = &Controller::RequestHandler_SetTransferFunction;
        controller_request_handlers[protocol::ControllerRequest_Type_GetTransferFunction] = &Controller::RequestHandler_GetTransferFunction;
        controller_request_handlers[protocol::ControllerRequest_Type_SetRGBLevels] = &Controller::RequestHandler_SetRGBLevels;
        controller_request_handlers[protocol::ControllerRequest_Type_GetRGBLevels] = &Controller::RequestHandler_GetRGBLevels;
        controller_request_handlers[protocol::ControllerRequest_Type_SetRendererParameters] = &Controller::RequestHandler_SetRendererParameters;
        controller_request_handlers[protocol::ControllerRequest_Type_GetRendererParameters] = &Controller::RequestHandler_GetRendererParameters;
        controller_request_handlers[protocol::ControllerRequest_Type_SetLensParameters] = &Controller::RequestHandler_SetLensParameters;
        controller_request_handlers[protocol::ControllerRequest_Type_GetLensParameters] = &Controller::RequestHandler_GetLensParameters;

        controller_request_handlers[protocol::ControllerRequest_Type_SavePreset] = &Controller::RequestHandler_SavePreset;
        controller_request_handlers[protocol::ControllerRequest_Type_LoadPreset] = &Controller::RequestHandler_LoadPreset;
        controller_request_handlers[protocol::ControllerRequest_Type_ListPresets] = &Controller::RequestHandler_ListPresets;

        controller_request_handlers[protocol::ControllerRequest_Type_HDRendering] = &Controller::RequestHandler_HDRendering;

        controller_request_handlers[protocol::ControllerRequest_Type_GetImage] = &Controller::RequestHandler_GetImage;
    }


    void thread_process() {
        socket_pubsub = zmq_socket(zmq_context, ZMQ_PUB);

        zmq_setsockopt_ez(socket_pubsub, ZMQ_SNDHWM, config.get<int>("sync.zmq.sndhwm", 10000));
        zmq_setsockopt_ez(socket_pubsub, ZMQ_SNDBUF, config.get<int>("sync.zmq.sndbuf", 0));
        zmq_setsockopt_ez(socket_pubsub, ZMQ_RATE, config.get<int>("sync.zmq.rate", 10000000));

        if(zmq_bind(socket_pubsub, config.get<string>("sync.broadcast").c_str()) < 0) {
            fprintf(stderr, "zmq_bind: %s\n", zmq_strerror(zmq_errno()));
            return;
        }

        socket_feedback = zmq_socket(zmq_context, ZMQ_PULL);
        if(zmq_bind(socket_feedback, config.get<string>("sync.feedback").c_str()) < 0) {
            fprintf(stderr, "zmq_bind: %s\n", zmq_strerror(zmq_errno()));
            return;
        }

        void* socket_commands = zmq_socket(zmq_context, ZMQ_REP);
        if(zmq_bind(socket_commands, config.get<string>("allovolume.controller").c_str()) < 0) {
            fprintf(stderr, "zmq_bind: %s\n", zmq_strerror(zmq_errno()));
            return;
        }

        socket_events = zmq_socket(zmq_context, ZMQ_PUB);
        if(zmq_bind(socket_events, config.get<string>("allovolume.events").c_str()) < 0) {
            fprintf(stderr, "zmq_bind: %s\n", zmq_strerror(zmq_errno()));
            return;
        }

        while(1) {
            zmq_pollitem_t items[2];
            items[0].socket = socket_feedback;
            items[0].events = ZMQ_POLLIN;
            items[1].socket = socket_commands;
            items[1].events = ZMQ_POLLIN;
            if(zmq_poll(items, 2, -1) < 0) {
                fprintf(stderr, "zmq_poll: %s\n", zmq_strerror(zmq_errno()));
                break;
            }
            if(items[0].revents & ZMQ_POLLIN) {
                // We got a message from socket_feedback.
                protocol::RendererFeedback feedback;
                zmq_protobuf_recv(feedback, socket_feedback);
                process_renderer_feedback(feedback);
            }
            if(items[1].revents & ZMQ_POLLIN) {
                // We got a message from socket_commands.
                protocol::ControllerRequest request;
                zmq_protobuf_recv(request, socket_commands);
                printf("%s\n", protocol::ControllerRequest_Type_Name(request.type()).c_str());
                // Send to the handlers.
                protocol::ControllerResponse response;
                std::map<protocol::ControllerRequest_Type, controller_handler_t>::iterator it =
                    controller_request_handlers.find(request.type());
                if(it != controller_request_handlers.end()) {
                    try {
                        response.set_status("success");
                        (this->*(it->second))(request, response);
                    } catch(...) {
                        response.set_status("E_INTERNAL_ERROR");
                    }
                } else {
                    response.set_status("E_NOT_IMPLEMENTED");
                }
                zmq_protobuf_send(response, socket_commands);
            }
        }
    }
};

int main(int argc, char* argv[]) {
    zmq_context = zmq_ctx_new();
    config.parseFile("allovolume.yaml");
    char hostname[256];
    gethostname(hostname, 256);
    config.parseFile("allovolume.yaml", hostname);
    config.parseFile("allovolume.yaml", "controller");
    Controller controller;

    void* socket_cmdline = zmq_socket(zmq_context, ZMQ_REQ);
    if(zmq_connect(socket_cmdline, config.get<string>("allovolume.controller").c_str()) < 0) {
        fprintf(stderr, "zmq_connect: %s\n", zmq_strerror(zmq_errno()));
        return -1;
    }

    Pose pose;

    while(1) {
        char* input_ = readline("allovolume.controller $ ");
        if(!input_) break;
        string input(input_);
        free(input_);
        vector<string> args;
        boost::split(args, input, boost::is_any_of("\t "), boost::token_compress_on);
        if(args.size() == 0) continue;
        boost::to_lower(args[0]);

        if(args[0] == "volume" && args.size() == 2) {

            protocol::ControllerRequest req;
            req.set_type(protocol::ControllerRequest_Type_LoadVolume);
            req.set_volume_filename(args[1]);
            zmq_protobuf_send(req, socket_cmdline);

        } else
        if(args[0] == "pos" && args.size() == 4) {

            protocol::ControllerRequest req;
            req.set_type(protocol::ControllerRequest_Type_SetPose);
            pose.position.x = ::atof(args[1].c_str());
            pose.position.y = ::atof(args[2].c_str());
            pose.position.z = ::atof(args[3].c_str());
            req.mutable_pose()->set_x(pose.position.x);
            req.mutable_pose()->set_y(pose.position.y);
            req.mutable_pose()->set_z(pose.position.z);
            req.mutable_pose()->set_qw(pose.rotation.w);
            req.mutable_pose()->set_qx(pose.rotation.v.x);
            req.mutable_pose()->set_qy(pose.rotation.v.y);
            req.mutable_pose()->set_qz(pose.rotation.v.z);
            zmq_protobuf_send(req, socket_cmdline);

        } else
        if(args[0] == "rot" && args.size() == 5) {

            protocol::ControllerRequest req;
            req.set_type(protocol::ControllerRequest_Type_SetPose);
            pose.rotation = Quaternion::Rotation(
                Vector(::atof(args[1].c_str()), ::atof(args[2].c_str()), ::atof(args[3].c_str())),
                ::atof(args[4].c_str()) / 180.0 * PI
            );
            req.mutable_pose()->set_x(pose.position.x);
            req.mutable_pose()->set_y(pose.position.y);
            req.mutable_pose()->set_z(pose.position.z);
            req.mutable_pose()->set_qw(pose.rotation.w);
            req.mutable_pose()->set_qx(pose.rotation.v.x);
            req.mutable_pose()->set_qy(pose.rotation.v.y);
            req.mutable_pose()->set_qz(pose.rotation.v.z);
            zmq_protobuf_send(req, socket_cmdline);

        } else
        if(args[0] == "hdrendering") {
            protocol::ControllerRequest req;
            req.set_type(protocol::ControllerRequest_Type_HDRendering);

            protocol::HDRenderingTask& task = *req.mutable_hd_rendering_task();

            task.set_lens_type(protocol::HDRenderingTask_LensType_Equirectangular);
            task.set_total_width(2000);
            task.set_total_height(1000);

            task.mutable_lens_parameters()->set_eye_separation(0);
            task.mutable_lens_parameters()->set_focal_distance(1);

            task.set_output_filename("aaa.png");

            zmq_protobuf_send(req, socket_cmdline);

        } else
        if(args[0] == "hdrendering-perspective") {
            protocol::ControllerRequest req;
            req.set_type(protocol::ControllerRequest_Type_HDRendering);

            protocol::HDRenderingTask& task = *req.mutable_hd_rendering_task();

            task.set_lens_type(protocol::HDRenderingTask_LensType_Perspective);
            task.set_perspective_fovx(PI / 2.0);
            task.set_total_width(3000);
            task.set_total_height(2000);

            task.set_output_filename("aaa.png");

            task.mutable_lens_parameters()->set_eye_separation(0);
            task.mutable_lens_parameters()->set_focal_distance(1);

            zmq_protobuf_send(req, socket_cmdline);

        } else
        {
            printf("Unknown command: %s\n", args[0].c_str());
            continue;
        }
        protocol::ControllerResponse response;
        zmq_protobuf_recv(response, socket_cmdline);
        printf("Result: %s\n", response.status().c_str());
    }
}
