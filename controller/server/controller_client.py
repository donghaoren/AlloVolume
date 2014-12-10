import allovolume_protocol_pb2 as protocol

import zmq
import struct

zmq_context = zmq.Context()

controller = zmq.Socket(zmq_context, zmq.REQ)
controller.connect("tcp://127.0.0.1:55556")


def HDRendering(filename, lens, width = 3000, height = 2000):
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.HDRendering

