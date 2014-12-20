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

from controller_client import config, Initialize, ListenEvents, RequestResponse
from controller_client import SetVolume
from controller_client import SetPose, SetTransferFunction, SetRGBLevels, SetRendererParameters, SetLensParameters
from controller_client import GetPose, GetTransferFunction, GetRGBLevels, GetRendererParameters, GetLensParameters

zmq_context = zmq.Context()

Initialize(zmq_context)

class ControllerServer(ApplicationSession):
    @inlineCallbacks
    def onJoin(self, details):
        def listen_for_parameter_change():
            for event in ListenEvents():
                print event
                import sys
                sys.stdout.flush()

        reactor.addSystemEventTrigger('before', 'shutdown', zmq_context.destroy)
        reactor.callInThread(listen_for_parameter_change)

        yield self.register(SetVolume, u"allovolume.renderer.set_volume")
        yield self.register(SetPose, u"allovolume.renderer.set_pose")
        yield self.register(GetPose, u"allovolume.renderer.get_pose")
        yield self.register(SetTransferFunction, u"allovolume.renderer.set_transfer_function")
        yield self.register(GetTransferFunction, u"allovolume.renderer.get_transfer_function")
        yield self.register(SetRGBLevels, u"allovolume.renderer.set_rgb_levels")
        yield self.register(GetRGBLevels, u"allovolume.renderer.get_rgb_levels")
        yield self.register(SetLensParameters, u"allovolume.renderer.set_lens_parameters")
        yield self.register(GetLensParameters, u"allovolume.renderer.get_lens_parameters")
        yield self.register(SetRendererParameters, u"allovolume.renderer.set_renderer_parameters")
        yield self.register(GetRendererParameters, u"allovolume.renderer.get_renderer_parameters")

root = File(".")
session_factory = RouterSessionFactory(RouterFactory())
component_config = types.ComponentConfig(realm = "anonymous")
session_factory.add(ControllerServer(component_config))
factory = WampWebSocketServerFactory(session_factory)
factory.startFactory()
resource = WebSocketResource(factory)
root.putChild("ws", resource)

site = Site(root)
reactor.listenTCP(int(config["allovolume"]["webserver"]["port"]), site, interface = config["allovolume"]["webserver"]["bind"])

reactor.run()
