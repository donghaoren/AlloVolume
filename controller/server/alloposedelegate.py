import re
import struct
import socket
import allovolume_protocol_pb2

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 4111))

pose = (
  (1, 0, 0, 0),
  (0, 0, 0)
)

while True:
    msg = s.recv(65535)

    new_pose = pose

    try:
        quat = struct.unpack('ffff', re.search(r'/as_view_quat\0+,ffff\0{3}(.{16})', msg).group(1))
        new_pose = ( quat, new_pose[1] )
    except: pass

    try:
        pos = struct.unpack('>fff', (re.search(r'/as_view_pos\0+,fff\0{3}(.{12})', msg).group(1)))
        pos = ( pos[0] * 1e10, pos[1] *  1e10, pos[2] * 1e10 )
        new_pose = ( new_pose[0], pos )
    except: pass

    if pose != new_pose:
        pose = new_pose
        print "Change pose: ", pose
