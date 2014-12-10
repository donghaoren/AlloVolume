import re
import struct
import socket
import zmq
import allovolume_protocol_pb2 as protocol

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 4111))

pose = (
  (1, 0, 0, 0),
  (0, 0, 0)
)
zmq_context = zmq.Context()
server = zmq.Socket(zmq_context, zmq.REQ)
server.connect("tcp://192.168.10.80:55560")

while True:
    msg = s.recv(65535)

    new_pose = pose

    try:
        quat = struct.unpack('>ffff', re.search(r'/as_view_quat\0+,ffff\0{3}(.{16})', msg).group(1))
        new_pose = ( quat, new_pose[1] )
    except: pass

    try:
        pos = struct.unpack('>fff', (re.search(r'/as_view_pos\0+,fff\0{4}(.{12})', msg).group(1)))
        pos = ( pos[0] * 1e9, pos[1] *  1e9, pos[2] * 1e9 )
        new_pose = ( new_pose[0], pos )
    except: pass

    if pose != new_pose:
        pose = new_pose
        msg = protocol.ControllerRequest()
        msg.type = protocol.ControllerRequest.SetPose
        msg.pose.x = float(pose[1][0])
        msg.pose.y = float(pose[1][1])
        msg.pose.z = float(pose[1][2])
        msg.pose.qw = float(pose[0][3])
        msg.pose.qx = float(pose[0][0])
        msg.pose.qy = float(pose[0][1])
        msg.pose.qz = float(pose[0][2])
        server.send(msg.SerializeToString())
        response = protocol.ControllerResponse()
        response.ParseFromString(server.recv())
        print "Change pose: ", pose