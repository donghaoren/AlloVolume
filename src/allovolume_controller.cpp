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

#include "allovolume_protocol.pb.h"

#include "allovolume_common.h"

using namespace std;
using namespace allovolume;


class Controller : public SyncSystem::Delegate {
public:
    Controller(SyncSystem* sync_) {
        waiting_for_barrier = false;
        sync = sync_;
        sync->setDelegate(this);
        sync->start();
    }

    void _send_message(const protocol::RendererBroadcast& msg) {
        size_t size = msg.ByteSize();
        void *buffer = malloc(size);
        msg.SerializeToArray(buffer, size);
        sync->sendMessage(buffer, size);
        free(buffer);
    }

    void loadVolume(VolumeBlocks* volume) {
        protocol::RendererBroadcast msg;

        msg.set_type(protocol::RendererBroadcast_RequestType_LoadVolume);

        size_t length = VolumeBlocks::WriteToBufferSize(volume);
        unsigned char* volume_data = (unsigned char*)malloc(length);
        VolumeBlocks::WriteToBuffer(volume, volume_data, length);
        msg.set_volume_data(volume_data, length);
        free(volume_data);

        _send_message(msg);
    }

    void render() {
        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_RequestType_Render);
        _send_message(msg);
    }

    void present() {
        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_RequestType_Present);
        _send_message(msg);
    }

    void barrier() {
        waiting_for_barrier = true;
        sync->sendBarrier();
        while(waiting_for_barrier) { usleep(1000); }
    }

    void setPose(Vector origin) {
        protocol::RendererBroadcast msg;
        msg.set_type(protocol::RendererBroadcast_RequestType_SetPose);
        protocol::Pose& pose = *msg.mutable_pose();
        pose.set_x(origin.x); pose.set_y(origin.y); pose.set_z(origin.z);
        pose.set_qx(0); pose.set_qy(0); pose.set_qz(0); pose.set_qw(1);
        _send_message(msg);
    }

    virtual void onBarrierClear(SyncSystem* sync, size_t sequence_id) {
        waiting_for_barrier = false;
    }

    volatile bool waiting_for_barrier;

    SyncSystem* sync;
};

int main(int argc, char* argv[]) {
    SyncSystem* sync = SyncSystem::Create("allovolume.server.yaml");
    Controller controller(sync);
    ConfigParser config("allovolume.server.yaml");

    void* controller_socket = zmq_socket(sync->getZMQContext(), ZMQ_REP);
    zmq_bind(controller_socket, config.get<string>("allovolume.controller_endpoint").c_str());

    while(1) {
        int r;
        protocol::RendererCommand cmd;
        zmq_msg_t msg;
        zmq_msg_init(&msg);
        r = zmq_msg_recv(&msg, controller_socket, 0);
        if(r < 0) {
            fprintf(stderr, "zmq_msg_recv: %s\n", zmq_strerror(zmq_errno()));
            break;
        }
        cmd.ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg));
        zmq_msg_close(&msg);

        protocol::RendererReply reply;
        reply.set_status("success");

        printf("Request: %s\n", RendererCommand_RequestType_Name(cmd.type()).c_str());

        switch(cmd.type()) {
            case protocol::RendererCommand_RequestType_LoadVolume: {
                VolumeBlocks* vol = VolumeBlocks::LoadFromFile(cmd.volume_filename().c_str());
                controller.loadVolume(vol);
                delete vol;
            } break;
            case protocol::RendererCommand_RequestType_LoadVolumeFromFile: {
            } break;
            case protocol::RendererCommand_RequestType_SetPose: {
                Vector origin(cmd.pose().x(), cmd.pose().y(), cmd.pose().z());
                controller.setPose(origin);
                controller.render();
                controller.present();
            } break;
            case protocol::RendererCommand_RequestType_SetTransferFunction: {
            } break;
        }

        zmq_msg_init_size(&msg, reply.ByteSize());
        reply.SerializeToArray(zmq_msg_data(&msg), zmq_msg_size(&msg));
        r = zmq_msg_send(&msg, controller_socket, 0);
        if(r < 0) {
            fprintf(stderr, "zmq_msg_send: %s\n", zmq_strerror(zmq_errno()));
            break;
        }
    }

    // while(1) {
    //     char* input = readline("> ");
    //     if(!input) break;
    //     string in(input);
    //     free(input);
    //     if(in == "volume") {
    //         VolumeBlocks* vol = VolumeBlocks::LoadFromFile("super3d_hdf5_plt_cnt_0122.volume");
    //         controller.loadVolume(vol);
    //         delete vol;
    //     }
    //     if(in == "setpose") {
    //         char* input = readline("Pose: ");
    //         Vector origin;
    //         sscanf(input, "%f %f %f", &origin.x, &origin.y, &origin.z);
    //         controller.setPose(origin);
    //     }
    //     if(in == "render") {
    //         controller.render();
    //     }
    //     if(in == "present") {
    //         controller.barrier();
    //         controller.present();
    //     }
    // }
}
