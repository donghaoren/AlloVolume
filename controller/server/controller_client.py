import allovolume_protocol_pb2 as protocol

import zmq
import struct
import math
import json

import yaml

config = yaml.load(open("allovolume.yaml").read().decode("utf-8"))

zmq_context = None
controller = None

def Initialize(zmq_context_):
    global controller, zmq_context
    zmq_context = zmq_context_
    controller = zmq.Socket(zmq_context, zmq.REQ)
    controller.connect(config['allovolume']['controller'])

def ListenEvents():
    events = zmq.Socket(zmq_context, zmq.SUB)
    events.connect(config['allovolume']['events'])
    events.setsockopt(zmq.SUBSCRIBE, "")
    while True:
        try:
            msg = events.recv()
            event = protocol.ParameterChangeEvent()
            event.ParseFromString(msg)
            yield event
        except zmq.error.ContextTerminated:
            break

def RequestResponse(request):
    global controller
    controller.send(request.SerializeToString())
    response = protocol.ControllerResponse()
    response.ParseFromString(controller.recv())
    return response

def SetVolume(filename):
    # Serialize the transfer function
    msg = protocol.RendererCommand()
    msg.type = RendererCommand.LoadVolume
    msg.volume_filename = filename
    return RequestResponse(msg).status == "success"

def SetTransferFunction(tf):
    # Serialize the transfer function
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.SetTransferFunction
    msg.transfer_function.domain_min = tf['domain'][0];
    msg.transfer_function.domain_max = tf['domain'][1];
    if tf['scale'] == 'linear':
        msg.transfer_function.scale = protocol.TransferFunction.Linear
    if tf['scale'] == 'log':
        msg.transfer_function.scale = protocol.TransferFunction.Log
    msg.transfer_function.size = 1600
    msg.transfer_function.layers = json.dumps(tf['layers'])
    return RequestResponse(msg).status == "success"

def GetTransferFunction():
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetTransferFunction
    response = RequestResponse(msg)
    tf = response.transfer_function
    result = { }
    result['domain'] = [ tf.domain_min, tf.domain_max ]
    if tf.scale == protocol.TransferFunction.Linear:
        result['scale'] = 'linear'
    if tf.scale == protocol.TransferFunction.Log:
        result['scale'] = 'log'
    return result

def SetLensParameters(focal_distance, eye_separation):
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.SetLensParameters
    msg.lens_parameters.focal_distance = float(focal_distance)
    msg.lens_parameters.eye_separation = float(eye_separation)
    return RequestResponse(msg).status == "success"

def GetLensParameters():
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetLensParameters
    response = RequestResponse(msg)
    return {
        'focal_distance': response.lens_parameters.focal_distance,
        'eye_separation': response.lens_parameters.eye_separation
    }

def SetRendererParameters(method, blending_coefficient):
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.SetRendererParameters
    msg.renderer_parameters.blending_coefficient = float(blending_coefficient)
    method_map = {
        'RK4': protocol.RendererParameters.RK4,
        'AdaptiveRKV': protocol.RendererParameters.AdaptiveRKV,
        'BasicBlending': protocol.RendererParameters.BasicBlending
    }
    msg.renderer_parameters.method = protocol.RendererParameters.RenderingMethod.Value(method)
    return RequestResponse(msg).status == "success"

def GetRendererParameters(method, blending_coefficient):
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetRendererParameters
    response = RequestResponse(msg)
    return {
        'method': protocol.RendererParameters.RenderingMethod.Value(response.renderer_parameters.method),
        'blending_coefficient': response.renderer_parameters.blending_coefficient
    }

def SetPose(x, y, z, qw, qx, qy, qz):
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.SetPose
    msg.pose.x = float(x)
    msg.pose.y = float(y)
    msg.pose.z = float(z)
    msg.pose.qw = float(qw)
    msg.pose.qx = float(qx)
    msg.pose.qy = float(qy)
    msg.pose.qz = float(qz)
    return RequestResponse(msg).status == "success"

def GetPose():
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetPose
    response = RequestResponse(msg)
    return {
        'x': response.pose.x,
        'y': response.pose.y,
        'z': response.pose.z,
        'qw': response.pose.qw,
        'qx': response.pose.qx,
        'qy': response.pose.qy,
        'qz': response.pose.qz
    }

def SetRGBLevels(min, max, pow):
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.SetRGBLevels
    msg.rgb_levels.min = float(min)
    msg.rgb_levels.max = float(max)
    msg.rgb_levels.pow = float(pow)
    return RequestResponse(msg).status == "success"

def GetRGBLevels():
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetRGBLevels
    response = RequestResponse(msg)
    return {
        'min': response.rgb_levels.min,
        'max': response.rgb_levels.max,
        'pow': response.rgb_levels.pow
    }

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

    print RequestResponse(msg)
