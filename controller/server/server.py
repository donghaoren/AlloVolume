from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner, RouterFactory, RouterSessionFactory
from autobahn.wamp import types
from autobahn.twisted.websocket import WampWebSocketServerFactory
from autobahn.twisted.resource import WebSocketResource
from twisted.internet.defer import inlineCallbacks
from twisted.web.server import Site
from twisted.web.static import File
from twisted.internet import reactor

import renderer_pb2 as protocol

import zmq

zmq_context = zmq.Context()

class ControllerServer(ApplicationSession):
    @inlineCallbacks
    def onJoin(self, details):
        print("Session starting up...")

        server = zmq.Socket(zmq_context, zmq.REQ)
        server.connect("tcp://127.0.0.1:5555")

        def set_volume(filename):
            # Serialize the transfer function
            msg = protocol.RendererCommand()
            msg.type = RendererCommand.LoadVolume
            msg.volume_filename = "super3d_hdf5_plt_cnt_0122.volume"
            server.send(msg.SerializeToString())
            return True

        def set_transfer_function(tf):
            # Serialize the transfer function
            return True

        yield self.register(set_volume, u"allovolume.renderer.set_volume")
        yield self.register(set_transfer_function, u"allovolume.renderer.set_transfer_function")
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
reactor.listenTCP(8080, site, interface = "localhost")

reactor.run()
