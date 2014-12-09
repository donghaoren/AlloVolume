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

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>

#include "allovolume_protocol.pb.h"

#include "allovolume_common.h"

#include <pthread.h>

using namespace std;
using namespace allovolume;

ConfigParser config;
void* zmq_context;

class Controller {
public:
    pthread_t thread;
    void* socket_pubsub;
    bool needs_render;
    bool is_rendering;

    enum BarrierInfo {
        kRenderPresentBarrier
    };

    // The current state.
    AlloVolumeState state;

    Controller() {
        needs_render = false;
        is_rendering = false;
        pthread_create(&thread, NULL, thread_process_pthread, this);
    }

    static void* thread_process_pthread(void* this_) {
        ((Controller*)this_)->thread_process();
    }

    void set_needs_render() {
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
            needs_render = false;
        }
    }

    void thread_process() {
        socket_pubsub = zmq_socket(zmq_context, ZMQ_PUB);

        zmq_setsockopt_ez(socket_pubsub, ZMQ_SNDHWM, config.get<int>("SyncSystem.zmq.sndhwm", 10000));
        zmq_setsockopt_ez(socket_pubsub, ZMQ_SNDBUF, config.get<int>("SyncSystem.zmq.sndbuf", 0));
        zmq_setsockopt_ez(socket_pubsub, ZMQ_RATE, config.get<int>("SyncSystem.zmq.rate", 10000000));

        if(zmq_bind(socket_pubsub, config.get<string>("SyncSystem.broadcast").c_str()) < 0) {
            fprintf(stderr, "zmq_bind: %s\n", zmq_strerror(zmq_errno()));
            return;
        }

        void* socket_feedback = zmq_socket(zmq_context, ZMQ_PULL);
        if(zmq_bind(socket_feedback, config.get<string>("SyncSystem.feedback").c_str()) < 0) {
            fprintf(stderr, "zmq_bind: %s\n", zmq_strerror(zmq_errno()));
            return;
        }

        void* socket_commands = zmq_socket(zmq_context, ZMQ_REP);
        if(zmq_bind(socket_commands, config.get<string>("allovolume.controller").c_str()) < 0) {
            fprintf(stderr, "zmq_bind: %s\n", zmq_strerror(zmq_errno()));
            return;
        }

        void* socket_events = zmq_socket(zmq_context, ZMQ_PUB);
        if(zmq_bind(socket_events, config.get<string>("allovolume.events").c_str()) < 0) {
            fprintf(stderr, "zmq_bind: %s\n", zmq_strerror(zmq_errno()));
            return;
        }

        set<string> barrier_clients;
        set<string> all_clients;

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
                // Process the feedback message.
                switch(feedback.type()) {
                    case protocol::RendererFeedback_Type_Register: {

                        all_clients.insert(feedback.client_name());
                        printf("New client: %s\n", feedback.client_name().c_str());

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
                            printf("Barrier cleared, info = %d!\n", feedback.barrier_info());
                        }

                    } break;
                }
            }
            if(items[1].revents & ZMQ_POLLIN) {
                // We got a message from socket_commands.
                protocol::ControllerRequest request;
                zmq_protobuf_recv(request, socket_commands);

                printf("%s\n", protocol::ControllerRequest_Type_Name(request.type()).c_str());

                switch(request.type()) {
                    case protocol::ControllerRequest_Type_LoadVolume: {

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

                    } break;
                    case protocol::ControllerRequest_Type_LoadVolumeFromFile: {

                        // TODO.

                    } break;
                    case protocol::ControllerRequest_Type_SetPose: {

                        state.pose = request.pose();

                        protocol::RendererBroadcast msg;
                        msg.set_type(protocol::RendererBroadcast_Type_SetPose);
                        *msg.mutable_pose() = state.pose;
                        zmq_protobuf_send(msg, socket_pubsub);

                        protocol::ParameterChangeEvent event;
                        event.set_type(protocol::ParameterChangeEvent_Type_SetPose);
                        *event.mutable_pose() = request.pose();
                        zmq_protobuf_send(event, socket_events);

                        set_needs_render();

                    } break;
                    case protocol::ControllerRequest_Type_SetTransferFunction: {

                        state.transfer_function = request.transfer_function();

                        protocol::RendererBroadcast msg;
                        msg.set_type(protocol::RendererBroadcast_Type_SetTransferFunction);
                        *msg.mutable_transfer_function() = request.transfer_function();
                        zmq_protobuf_send(msg, socket_pubsub);

                        protocol::ParameterChangeEvent event;
                        event.set_type(protocol::ParameterChangeEvent_Type_SetTransferFunction);
                        *event.mutable_transfer_function() = request.transfer_function();
                        zmq_protobuf_send(event, socket_events);

                        set_needs_render();

                    } break;
                    case protocol::ControllerRequest_Type_SetRGBCurve: {

                        state.rgb_curve = request.rgb_curve();

                        protocol::RendererBroadcast msg;
                        msg.set_type(protocol::RendererBroadcast_Type_SetRGBCurve);
                        *msg.mutable_transfer_function() = request.transfer_function();
                        zmq_protobuf_send(msg, socket_pubsub);

                        protocol::ParameterChangeEvent event;
                        event.set_type(protocol::ParameterChangeEvent_Type_SetRGBCurve);
                        *event.mutable_rgb_curve() = request.rgb_curve();
                        zmq_protobuf_send(event, socket_events);

                        set_needs_render();

                    } break;
                    case protocol::ControllerRequest_Type_SetRendererParameters: {

                        state.renderer_parameters = request.renderer_parameters();

                        protocol::RendererBroadcast msg;
                        msg.set_type(protocol::RendererBroadcast_Type_SetRendererParameters);
                        *msg.mutable_renderer_parameters() = request.renderer_parameters();
                        zmq_protobuf_send(msg, socket_pubsub);

                        protocol::ParameterChangeEvent event;
                        event.set_type(protocol::ParameterChangeEvent_Type_SetRendererParameters);
                        *event.mutable_renderer_parameters() = request.renderer_parameters();
                        zmq_protobuf_send(event, socket_events);

                        set_needs_render();

                    } break;
                    case protocol::ControllerRequest_Type_SetLensParameters: {

                        state.lens_parameters = request.lens_parameters();

                        protocol::RendererBroadcast msg;
                        msg.set_type(protocol::RendererBroadcast_Type_SetLensParameters);
                        *msg.mutable_lens_parameters() = request.lens_parameters();
                        zmq_protobuf_send(msg, socket_pubsub);

                        protocol::ParameterChangeEvent event;
                        event.set_type(protocol::ParameterChangeEvent_Type_SetLensParameters);
                        *event.mutable_lens_parameters() = request.lens_parameters();
                        zmq_protobuf_send(event, socket_events);

                        set_needs_render();

                    } break;
                }

                protocol::ControllerResponse response;
                response.set_status("success");
                zmq_protobuf_send(response, socket_commands);
            }
        }
    }
};

int main(int argc, char* argv[]) {
    zmq_context = zmq_ctx_new();
    config.parseFile("allovolume.server.yaml");
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
        if(args[0] == "tf-gaussian" && args.size() == 5) {

            protocol::ControllerRequest req;
            req.set_type(protocol::ControllerRequest_Type_SetTransferFunction);
            protocol::TransferFunction& tf = *req.mutable_transfer_function();
            tf.set_scale(args[1] == "log" ? protocol::TransferFunction_Scale_Log : protocol::TransferFunction_Scale_Linear);
            tf.set_domain_min(::atof(args[2].c_str()));
            tf.set_domain_max(::atof(args[3].c_str()));
            TransferFunction* tf_real = TransferFunction::CreateGaussianTicks(0, 1, TransferFunction::kLogScale, ::atoi(args[4].c_str()));
            // tf.set_size(tf_real->getSize());
            tf.set_content(tf_real->getContent(), sizeof(Color) * tf_real->getSize());
            delete tf_real;

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
