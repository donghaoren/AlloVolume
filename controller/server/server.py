from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner, RouterFactory, RouterSessionFactory
from autobahn.wamp import types
from autobahn.twisted.websocket import WampWebSocketServerFactory
from autobahn.twisted.resource import WebSocketResource
from twisted.internet.defer import inlineCallbacks
from twisted.web.server import Site
from twisted.web.static import File
from twisted.internet import reactor

import allovolume_protocol_pb2 as protocol

import zmq
import struct

zmq_context = zmq.Context()

class ControllerServer(ApplicationSession):
    @inlineCallbacks
    def onJoin(self, details):
        print("Session starting up...")

        server = zmq.Socket(zmq_context, zmq.REQ)
        server.connect("tcp://127.0.0.1:55556")

        def set_volume(filename):
            # Serialize the transfer function
            msg = protocol.RendererCommand()
            msg.type = RendererCommand.LoadVolume
            msg.volume_filename = "super3d_hdf5_plt_cnt_0122.volume"
            server.send(msg.SerializeToString())
            return True

        def set_transfer_function(tf):
            # Serialize the transfer function
            msg = protocol.ControllerRequest()
            msg.type = protocol.ControllerRequest.SetTransferFunction
            msg.transfer_function.domain_min = tf['domain'][0];
            msg.transfer_function.domain_max = tf['domain'][1];
            if tf['scale'] == 'linear':
                msg.transfer_function.scale = protocol.TransferFunction.Linear
            if tf['scale'] == 'log':
                msg.transfer_function.scale = protocol.TransferFunction.Log
            content = map(lambda x: struct.pack("ffff", x[0], x[1], x[2], x[3]), tf['content'])
            msg.transfer_function.content = "".join(content)
            server.send(msg.SerializeToString())
            response = protocol.ControllerResponse()
            response.ParseFromString(server.recv())
            return True

        def set_lens_parameters(focal_distance, eye_separation):
            msg = protocol.ControllerRequest()
            msg.type = protocol.ControllerRequest.SetLensParameters
            msg.lens_parameters.focal_distance = float(focal_distance)
            msg.lens_parameters.eye_separation = float(eye_separation)
            server.send(msg.SerializeToString())
            response = protocol.ControllerResponse()
            response.ParseFromString(server.recv())
            return True

        def set_renderer_parameters(method, blending_coefficient):
            msg = protocol.ControllerRequest()
            msg.type = protocol.ControllerRequest.SetRendererParameters
            msg.renderer_parameters.blending_coefficient = float(blending_coefficient)
            method_map = {
                'RK4': protocol.RendererParameters.RK4,
                'AdaptiveRKV': protocol.RendererParameters.AdaptiveRKV
            }
            msg.renderer_parameters.method = method_map[method]
            server.send(msg.SerializeToString())
            response = protocol.ControllerResponse()
            response.ParseFromString(server.recv())
            return True

        def set_pose(x, y, z, qw, qx, qy, qz):
            msg = protocol.ControllerRequest()
            msg.type = protocol.ControllerRequest.setPose
            msg.pose.x = float(x)
            msg.pose.y = float(y)
            msg.pose.z = float(z)
            msg.pose.qw = float(qw)
            msg.pose.qx = float(qx)
            msg.pose.qy = float(qy)
            msg.pose.qz = float(qz)
            server.send(msg.SerializeToString())
            response = protocol.ControllerResponse()
            response.ParseFromString(server.recv())
            return True

        def set_rgb_curve(curve):
            return True

        yield self.register(set_volume, u"allovolume.renderer.set_volume")
        yield self.register(set_transfer_function, u"allovolume.renderer.set_transfer_function")
        yield self.register(set_pose, u"allovolume.renderer.set_pose")
        yield self.register(set_lens_parameters, u"allovolume.renderer.set_lens_parameters")
        yield self.register(set_renderer_parameters, u"allovolume.renderer.set_renderer_parameters")
        print("Session startup success!")

root = File(".")
session_factory = RouterSessionFactory(RouterFactory())
component_config = types.ComponentConfig(realm = "anonymous")
session_factory.add(ControllerServer(component_config))
factory = WampWebSocketServerFactory(session_factory)
#factory.protocol = YourServerProtocolClass
factory.startFactory()
resource = WebSocketResource(factory)
root.putChild("ws", resource)

site = Site(root)
reactor.listenTCP(8080, site, interface = "0.0.0.0")

reactor.run()
