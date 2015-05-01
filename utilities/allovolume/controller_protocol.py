import allovolume_protocol_pb2 as protocol
import os
import zmq
import struct
import math
import json
import base64
import yaml

class AlloVolumeController:
    """Connects to the allovolume controller."""

    def __init__(self, zmq_context = None, path = None):
        """Initialize the connection.

        Args:
            zmq_context: ZeroMQ context.
            path: Path to allovolume.yaml.
        """

        if zmq_context == None: zmq_context = zmq.Context()
        if path == None: path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "allovolume.yaml")

        self.config = yaml.load(open(path).read().decode("utf-8"))
        self.zmq_context = zmq_context
        self.controller = zmq.Socket(self.zmq_context, zmq.REQ)
        self.controller.connect(self.config['allovolume']['controller'])

    def ListenEvents(self):
        """Listen to controller events.

        Yields:
           Dicts as event.
        """
        events = zmq.Socket(zmq_context, zmq.SUB)
        events.connect(self.config['allovolume']['events'])
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
                    parsed_event['transfer_function'] = self.ParseTransferFunction(event.transfer_function)

                if event.HasField("pose"):
                    parsed_event['pose'] = ParsePose(event.pose)

                if event.HasField("lens_parameters"):
                    parsed_event['lens_parameters'] = self.ParseLensParameters(event.lens_parameters)

                if event.HasField("renderer_parameters"):
                    parsed_event['renderer_parameters'] = self.ParseRendererParameters(event.renderer_parameters)

                if event.HasField("rgb_levels"):
                    parsed_event['rgb_levels'] = self.ParseRGBLevels(event.rgb_levels)

                if event.HasField("hd_rendering_filename"):
                    parsed_event['hd_rendering_filename'] = event.hd_rendering_filename

                yield parsed_event

            except zmq.error.ContextTerminated:
                break

    def RequestResponse(self, request):
        """Send a protobuf request and receive response from the controller.

        Args:
            request: The request.

        Returns:
            The response from the controller.
        """
        self.controller.send(request.SerializeToString())
        response = protocol.ControllerResponse()
        response.ParseFromString(self.controller.recv())
        return response

    def Render(self):
        """Trigger a render of the volume.

        Returns:
            True if command succeed, False otherwise.
        """

        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.Render
        return self.RequestResponse(msg).status == "success"

    def LoadVolume(self, filename, dataset, description):
        """Load volume from file on controller.

        Args:
            filename: The path of the volume, which should be accessible by the controller machine.
            dataset: The name of the dataset.
            description: The description of the dataset.

        Returns:
            True if command succeed, False otherwise.
        """
        # Serialize the transfer function
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.LoadVolume
        msg.volume_filename = filename
        msg.volume_dataset = dataset
        msg.volume_description = description
        return self.RequestResponse(msg).status == "success"

    def LoadVolumeFromFile(self, filename, dataset, description):
        """Load volume from file on renderers.

        Args:
            filename: The path of the volume, which should be accessible by each of the rendering machines.
            dataset: The name of the dataset.
            description: The description of the dataset.

        Returns:
            True if command succeed, False otherwise.
        """
        # Serialize the transfer function
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.LoadVolumeFromFile
        msg.volume_filename = filename
        msg.volume_dataset = dataset
        msg.volume_description = description
        return self.RequestResponse(msg).status == "success"

    def LoadVolumeFromData(self, data, dataset, description):
        """Load volume from data.

        Args:
            data: The binary volume data, in AlloVolume's data format.
            dataset: The name of the dataset.
            description: The description of the dataset.

        Returns:
            True if command succeed, False otherwise.
        """
        # Serialize the transfer function
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.LoadVolumeFromData
        msg.volume_data = data
        msg.volume_dataset = dataset
        msg.volume_description = description
        return self.RequestResponse(msg).status == "success"

    def ParseTransferFunction(self, tf):
        result = { }
        result['domain'] = [ tf.domain_min, tf.domain_max ]
        if tf.scale == protocol.TransferFunction.Linear:
            result['scale'] = 'linear'
        if tf.scale == protocol.TransferFunction.Log:
            result['scale'] = 'log'
        result['layers'] = json.loads(tf.layers)
        return result

    def SetTransferFunction(self, sender, tf):
        """Set transfer function.

        Args:
            sender: The name of the sender.
            tf: The transfer function description.

        Returns:
            True if command succeed, False otherwise.
        """
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
        return self.RequestResponse(msg).status == "success"

    def GetTransferFunction(self):
        """Get current transfer function.

        Returns:
            The transfer function description.
        """
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.GetTransferFunction
        response = self.RequestResponse(msg)
        tf = response.transfer_function
        return self.ParseTransferFunction(tf)

    def SetLensParameters(self, sender, focal_distance, eye_separation):
        """Set lens parameters.

        Args:
            sender: The name of the sender.
            focal_distance: The focal distance (aka, the radius of the sphere).
            eye_separation: The eye separation.

        Returns:
            True if command succeed, False otherwise.
        """
        msg = protocol.ControllerRequest()
        msg.sender = sender
        msg.type = protocol.ControllerRequest.SetLensParameters
        msg.lens_parameters.focal_distance = float(focal_distance)
        msg.lens_parameters.eye_separation = float(eye_separation)
        return self.RequestResponse(msg).status == "success"

    def ParseLensParameters(self, lens_parameters):
        return {
            'focal_distance': lens_parameters.focal_distance,
            'eye_separation': lens_parameters.eye_separation
        }

    def GetLensParameters(self):
        """Get lens parameters.

        Returns:
            The current lens parameters.
        """
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.GetLensParameters
        response = self.RequestResponse(msg)
        return self.ParseLensParameters(response.lens_parameters)

    def SetRendererParameters(self, sender, method, blending_coefficient, step_size = 1.0, internal_format = "Float32", enable_z_index = False):
        """Set renderer parameters.

        Args:
            sender: The name of the sender.
            method: Raycasting method.
            blending_coefficient: Blending coefficient.
            step_size: Step size multiplier.
            internal_format: Internal data format.
            enable_z_index: Use Z-indexing or not.

        Returns:
            True if command succeed, False otherwise.
        """
        msg = protocol.ControllerRequest()
        msg.sender = sender
        msg.type = protocol.ControllerRequest.SetRendererParameters
        msg.renderer_parameters.blending_coefficient = float(blending_coefficient)
        msg.renderer_parameters.step_size = float(step_size)
        msg.renderer_parameters.method = protocol.RendererParameters.RenderingMethod.Value(method)
        msg.renderer_parameters.internal_format = protocol.RendererParameters.InternalFormat.Value(internal_format)
        msg.renderer_parameters.enable_z_index = enable_z_index
        return self.RequestResponse(msg).status == "success"

    def ParseRendererParameters(self, renderer_parameters):
        return {
            'method': protocol.RendererParameters.RenderingMethod.Name(renderer_parameters.method),
            'blending_coefficient': renderer_parameters.blending_coefficient,
            'step_size': renderer_parameters.step_size,
            'internal_format': protocol.RendererParameters.InternalFormat.Name(renderer_parameters.internal_format),
            'enable_z_index': renderer_parameters.enable_z_index
        }

    def GetRendererParameters(self):
        """Get renderer parameters.

        Returns:
            The current renderer parameters.
        """
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.GetRendererParameters
        response = self.RequestResponse(msg)
        return self.ParseRendererParameters(response.renderer_parameters)

    def SetPose(self, sender, x, y, z, qw, qx, qy, qz):
        """Set pose.

        Args:
            sender: The name of the sender.
            x, y, z: The position.
            qw, qx, qy, qz: The quaternion for rotation.

        Returns:
            True if command succeed, False otherwise.
        """
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
        return self.RequestResponse(msg).status == "success"

    def ParsePose(self, pose):
        return {
            'x': pose.x,
            'y': pose.y,
            'z': pose.z,
            'qw': pose.qw,
            'qx': pose.qx,
            'qy': pose.qy,
            'qz': pose.qz
        }

    def GetPose(self):
        """Get pose.

        Returns:
            The current pose.
        """
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.GetPose
        response = self.RequestResponse(msg)
        return self.ParsePose(response.pose)

    def SetRGBLevels(self, sender, min, max, pow):
        """Set RGB levels.

        The mathematical operation on R, G, B is: x' = ((x - min) / (max - min)) ^ pow.
        This only works in the standalone renderer.

        Args:
            sender: The name of the sender.
            min: Minimum RGB.
            max: Maximum RGB.
            pow: Power.

        Returns:
            True if command succeed, False otherwise.
        """
        msg = protocol.ControllerRequest()
        msg.sender = sender
        msg.type = protocol.ControllerRequest.SetRGBLevels
        msg.rgb_levels.min = float(min)
        msg.rgb_levels.max = float(max)
        msg.rgb_levels.pow = float(pow)
        return self.RequestResponse(msg).status == "success"

    def ParseRGBLevels(self, rgb_levels):
        return {
            'min': rgb_levels.min,
            'max': rgb_levels.max,
            'pow': rgb_levels.pow
        }

    def GetRGBLevels(self):
        """Get RGB levels.

        Returns:
            The current RGB levels.
        """
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.GetRGBLevels
        response = self.RequestResponse(msg)
        return self.ParseRGBLevels(response.rgb_levels)

    def SavePreset(self, name, description):
        """Save preset.

        Args:
            name: Preset name.
            description: Preset description.

        Returns:
            True if command succeed, False otherwise.
        """
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.SavePreset
        msg.preset_name = name
        msg.preset_description = description
        return self.RequestResponse(msg).status == "success"

    def LoadPreset(self, name):
        """Load preset.

        Args:
            name: Preset name.

        Returns:
            True if command succeed, False otherwise.
        """
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.LoadPreset
        msg.preset_name = name
        return self.RequestResponse(msg).status == "success"

    def ListPresets(self):
        """Get a list of presets.

        Returns:
            List of preset names.
        """
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.ListPresets
        response = self.RequestResponse(msg)
        return list(response.preset_list)

    def GetImage(self, filename):
        """Retrieve image file.

        Args:
            filename: Image filename.

        Returns:
            Binary image data.
        """
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.GetImage
        msg.image_filename = filename
        response = self.RequestResponse(msg)
        return response.binary_data

    def HDRendering(self, filename, width = 3000, height = 2000, lens = "perspective", fovx = 90):
        """Perform HD Rendering.

        Args:
            filename: Output filename.
            width: Output image width.
            height: Output image height.
            lens: Lens type: 'perspective' / 'equirectangular'.
            fovx: Perspective lens parameter: horizontal field of view in degrees.

        Returns:
            True if command succeed, False otherwise.
        """
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

        return self.RequestResponse(msg).status == "success"
