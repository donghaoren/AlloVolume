import allovolume_protocol_pb2 as protocol

import zmq
import struct
import math

zmq_context = zmq.Context()

controller = zmq.Socket(zmq_context, zmq.REQ)
controller.connect("tcp://127.0.0.1:55556")


def HDRendering(filename, width = 3000, height = 2000, lens = "perspective", fovx = 90):
    msg = protocol.ControllerRequest()

    msg.type = protocol.ControllerRequest.HDRendering

    if lens == "perspective":
        msg.hd_rendering_task.lens_type = protocol.HDRenderingTask.Perspective
        msg.hd_rendering_task.perspective_fovx = fovx / 180.0 * math.pi
    if lens == "equirectangular":
        msg.hd_rendering_task.lens_type = protocol.HDRenderingTask.Equirectangular

    msg.hd_rendering_task.total_width = width
    msg.hd_rendering_task.total_height = height

    msg.hd_rendering_task.output_filename = filename

    msg.hd_rendering_task.lens_parameters.eye_separation = 0
    msg.hd_rendering_task.lens_parameters.focal_distance = 1

    controller.send(msg.SerializeToString())

    result = protocol.ControllerResponse()
    result.ParseFromString(controller.recv())
    print result
