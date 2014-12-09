#ifndef ALLOVOLUME_COMMON_H_INCLUDED
#define ALLOVOLUME_COMMON_H_INCLUDED

#include "dataset.h"
#include "renderer.h"
#include "allosphere/allosphere_calibration.h"
#include <zmq.h>
#include <boost/shared_ptr.hpp>
#include <vector>

#include "allovolume_protocol.pb.h"

// The renderer has 4 state varaibles:
//  volume: The volume to be rendered.
//  pose: The pose of the viewer. (x, y, z, qx, qy, qz, qw)
//  lens: The information for the lens. (eye_separation, focal_distance)
//  transfer_function: The transfer function to use.
//  RGB curve: The rgb curve for final output.
// Volume is set only, others are get/set.

struct AlloVolumeState {
    allovolume::protocol::TransferFunction transfer_function;
    allovolume::protocol::Pose pose;
    allovolume::protocol::RGBCurve rgb_curve;
    allovolume::protocol::LensParameters lens_parameters;
    allovolume::protocol::RendererParameters renderer_parameters;
};

template<typename ProtobufT>
int zmq_protobuf_send(const ProtobufT& message, void* socket) {
    zmq_msg_t msg;
    zmq_msg_init_size(&msg, message.ByteSize());
    message.SerializeToArray(zmq_msg_data(&msg), zmq_msg_size(&msg));
    return zmq_msg_send(&msg, socket, 0);
}

template<typename ProtobufT>
int zmq_protobuf_recv(ProtobufT& message, void* socket) {
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int r = zmq_msg_recv(&msg, socket, 0);
    message.ParseFromArray(zmq_msg_data(&msg), zmq_msg_size(&msg));
    zmq_msg_close(&msg);
    return r;
}

template<typename T>
void zmq_setsockopt_ez(void* socket, int name, T value) {
    zmq_setsockopt(socket, name, &value, sizeof(T));
}

#endif
