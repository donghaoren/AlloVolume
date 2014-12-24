import allovolume_protocol_pb2 as protocol

import zmq
import struct
import math
import json
import base64
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
            parsed_event = {
                "sender": event.sender,
                "type": protocol.ParameterChangeEvent.Type.Name(event.type)
            }

            if event.HasField("transfer_function"):
                parsed_event['transfer_function'] = ParseTransferFunction(event.transfer_function)

            if event.HasField("pose"):
                parsed_event['pose'] = ParsePose(event.pose)

            if event.HasField("lens_parameters"):
                parsed_event['lens_parameters'] = ParseLensParameters(event.lens_parameters)

            if event.HasField("renderer_parameters"):
                parsed_event['renderer_parameters'] = ParseRendererParameters(event.renderer_parameters)

            if event.HasField("rgb_levels"):
                parsed_event['rgb_levels'] = ParseRGBLevels(event.rgb_levels)

            if event.HasField("hd_rendering_filename"):
                parsed_event['hd_rendering_filename'] = event.hd_rendering_filename

            yield parsed_event

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

def ParseTransferFunction(tf):
    result = { }
    result['domain'] = [ tf.domain_min, tf.domain_max ]
    if tf.scale == protocol.TransferFunction.Linear:
        result['scale'] = 'linear'
    if tf.scale == protocol.TransferFunction.Log:
        result['scale'] = 'log'
    result['layers'] = json.loads(tf.layers)
    return result

def SetTransferFunction(sender, tf):
    # Serialize the transfer function
    msg = protocol.ControllerRequest()
    msg.sender = sender
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
    return ParseTransferFunction(tf)

def SetLensParameters(sender, focal_distance, eye_separation):
    msg = protocol.ControllerRequest()
    msg.sender = sender
    msg.type = protocol.ControllerRequest.SetLensParameters
    msg.lens_parameters.focal_distance = float(focal_distance)
    msg.lens_parameters.eye_separation = float(eye_separation)
    return RequestResponse(msg).status == "success"

def ParseLensParameters(lens_parameters):
    return {
        'focal_distance': lens_parameters.focal_distance,
        'eye_separation': lens_parameters.eye_separation
    }

def GetLensParameters():
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetLensParameters
    response = RequestResponse(msg)
    return ParseLensParameters(response.lens_parameters)

def SetRendererParameters(sender, method, blending_coefficient):
    msg = protocol.ControllerRequest()
    msg.sender = sender
    msg.type = protocol.ControllerRequest.SetRendererParameters
    msg.renderer_parameters.blending_coefficient = float(blending_coefficient)
    msg.renderer_parameters.method = protocol.RendererParameters.RenderingMethod.Value(method)
    return RequestResponse(msg).status == "success"

def ParseRendererParameters(renderer_parameters):
    return {
        'method': protocol.RendererParameters.RenderingMethod.Name(renderer_parameters.method),
        'blending_coefficient': renderer_parameters.blending_coefficient
    }

def GetRendererParameters():
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetRendererParameters
    response = RequestResponse(msg)
    return ParseRendererParameters(response.renderer_parameters)

def SetPose(sender, x, y, z, qw, qx, qy, qz):
    msg = protocol.ControllerRequest()
    msg.sender = sender
    msg.type = protocol.ControllerRequest.SetPose
    msg.pose.x = float(x)
    msg.pose.y = float(y)
    msg.pose.z = float(z)
    msg.pose.qw = float(qw)
    msg.pose.qx = float(qx)
    msg.pose.qy = float(qy)
    msg.pose.qz = float(qz)
    return RequestResponse(msg).status == "success"

def ParsePose(pose):
    return {
        'x': pose.x,
        'y': pose.y,
        'z': pose.z,
        'qw': pose.qw,
        'qx': pose.qx,
        'qy': pose.qy,
        'qz': pose.qz
    }

def GetPose():
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetPose
    response = RequestResponse(msg)
    return ParsePose(response.pose)

def SetRGBLevels(sender, min, max, pow):
    msg = protocol.ControllerRequest()
    msg.sender = sender
    msg.type = protocol.ControllerRequest.SetRGBLevels
    msg.rgb_levels.min = float(min)
    msg.rgb_levels.max = float(max)
    msg.rgb_levels.pow = float(pow)
    return RequestResponse(msg).status == "success"

def ParseRGBLevels(rgb_levels):
    return {
        'min': rgb_levels.min,
        'max': rgb_levels.max,
        'pow': rgb_levels.pow
    }

def GetRGBLevels():
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetRGBLevels
    response = RequestResponse(msg)
    return ParseRGBLevels(response.rgb_levels)

def SavePreset(name, description):
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.SavePreset
    msg.preset_name = name
    msg.preset_description = description
    return RequestResponse(msg).status == "success"

def LoadPreset(name):
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.LoadPreset
    msg.preset_name = name
    return RequestResponse(msg).status == "success"

def ListPresets():
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.ListPresets
    response = RequestResponse(msg)
    return list(response.preset_list)

def GetImage(filename):
    msg = protocol.ControllerRequest()
    msg.type = protocol.ControllerRequest.GetImage
    msg.image_filename = filename
    response = RequestResponse(msg)
    return base64.b64encode(response.binary_data)

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
