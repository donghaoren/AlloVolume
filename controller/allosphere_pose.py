import OSC
import threading
import time
import zmq
from server.controller_client import SetPose, GetPose, Initialize

zmq_context = zmq.Context()

Initialize(zmq_context)

client = OSC.OSCClient()
client.connect(("192.168.0.3", 12000))

server = OSC.OSCServer(("0.0.0.0", 12001))
server.timeout = 1

msg = OSC.OSCMessage()
msg.setAddress("/handshake")
msg.append("AlloVolume")
msg.append(12001)
client.send(msg)

ctrl_state = {
    "lx": 0, "ly": 0, "rx": 0, "ry": 0
}

def user_callback(path, tags, args, source):
    global ctrl_state
    if path == "/lx": ctrl_state["lx"] = args[0]
    if path == "/ly": ctrl_state["ly"] = args[0]
    if path == "/rx": ctrl_state["rx"] = args[0]
    if path == "/ry": ctrl_state["ry"] = args[0]

def timer_thread():
    rotation_speed = [0, 0]
    rotation_momentum = 0.7

    translation_speed = [0, 0, 0]
    translation_momentum = 0.7

    scale = 1e10

    delta = 0.1 * scale

    pose = ( (0, 0, 0), (0, 0, 0, 1) )

    while True:
        time.sleep(0.005)

        rotation_speed[0] = rotation_speed[0] * rotation_momentum + ctrl_state['lx'] * (1 - rotation_momentum)
        rotation_speed[1] = rotation_speed[1] * rotation_momentum + ctrl_state['ly'] * (1 - rotation_momentum)
        translation_speed[0] = translation_speed[0] * translation_momentum + ctrl_state['ry'] * (1 - translation_momentum)
        translation_speed[1] = translation_speed[1] * translation_momentum + ctrl_state['rx'] * (1 - translation_momentum)

        ((x, y, z), (qx, qy, qz, qw)) = pose
        x += translation_speed[0] * delta
        y += translation_speed[1] * delta
        z += translation_speed[2] * delta
        new_pose = ((x, y, z), (qx, qy, qz, qw))

        print new_pose
        if new_pose != pose:
            pose = new_pose
            SetPose(sender = "allosphere_pose.py", x = x, y = y, z = z, qw = qw, qx = qx, qy = qy, qz = qz)

server.addMsgHandler("/handshake", user_callback)
server.addMsgHandler("/lx", user_callback)
server.addMsgHandler("/ly", user_callback)
server.addMsgHandler("/rx", user_callback)
server.addMsgHandler("/ry", user_callback)
server.addMsgHandler("/hat", user_callback)
server.addMsgHandler("/agents", user_callback)
server.addMsgHandler("/b2", user_callback)
server.addMsgHandler("/b3", user_callback)
server.addMsgHandler("/b4", user_callback)
server.addMsgHandler("/b9", user_callback)
server.addMsgHandler("/b10", user_callback)
server.addMsgHandler("/b5", user_callback)
server.addMsgHandler("/b6", user_callback)
server.addMsgHandler("/b7", user_callback)
server.addMsgHandler("/b8", user_callback)

th = threading.Thread(target = timer_thread)
th.daemon = True
th.start()

while True:
    server.handle_request()



# Joystick mappings:
# /hat: Left cross. Up: 0, Right: 2, Down: 4, Left: 6, Release: 8. Odd number means in between.
# lx: -1 ~ +1 (left ~ right)
# ly: -1 ~ +1 (top ~ bottom)
# rx: -1 ~ +1 (left ~ right)
# ry: -1 ~ +1 (top ~ bottom)
# agents, b1, b2, b3, b4: 1=pressed, 0=released
# b9, b10: Button in the center (top two)
# b5, b6: Back top row (left / right).
# b7, b8: Back top row (left / right).
