import OSC
import threading
import time
import zmq
import math
from server.controller_client import SetPose, GetPose, Initialize, ListenEvents, GetLensParameters

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

# Global parameters.
scale = 1e10
pose = ( (0, 0, 0), (1, 0, 0, 0) )

p = GetPose()
pose = ( (p['x'], p['y'], p['z']), (p['qw'], p['qx'], p['qy'], p['qz']) )
rp = GetLensParameters()
scale = rp['focal_distance']

ctrl_state = {
    "lx": 0, "ly": 0, "rx": 0, "ry": 0, "rotate_mode": "viewer"
}

def QuaternionMul((w1, x1, y1, z1), (w2, x2, y2, z2)):
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
def QuaternionConj((w1, x1, y1, z1)):
    return (w1, -x1, -y1, -z1)

def QuaternionRotation(q, (vx, vy, vz)):
    vq = (0, vx, vy, vz)
    return QuaternionMul(QuaternionMul(q, vq), QuaternionConj(q))[1:]
def RotationQuaternion((vx, vy, vz), angle):
    length = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    s = math.sin(angle / 2) / length
    return (
        math.cos(angle / 2),
        vx * s, vy * s, vz * s
    )
def VectorCross((x1, y1, z1), (x2, y2, z2)):
    return (
        y1 * z2 - y2 * z1,
        z1 * x2 - z2 * x1,
        x1 * y2 - x2 * y1
    )
def VectorDot((x1, y1, z1), (x2, y2, z2)):
    return x1 * x2 + y1 * y2 + z1 * z2
def VectorNormalize((x, y, z)):
    s = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    return ( x / s, y / s, z / s )

def user_callback(path, tags, args, source):
    global ctrl_state
    if path == "/lx": ctrl_state["lx"] = args[0]
    if path == "/ly": ctrl_state["ly"] = args[0]
    if path == "/rx": ctrl_state["rx"] = args[0]
    if path == "/ry": ctrl_state["ry"] = args[0]
    if path == "/b5": ctrl_state['rotate_mode'] = "model" if args[0] > 0.5 else "viewer"

def event_thread():
    global pose, scale
    for event in ListenEvents():
        if event['sender'] != "allosphere_pose.py":
            if event['type'] == "SetPose":
                p = event['pose']
                pose = ( (p['x'], p['y'], p['z']), (p['qw'], p['qx'], p['qy'], p['qz']) )
            if event['type'] == "SetLensParameters":
                rp = event['lens_parameters']
                scale = rp['focal_distance']

def timer_thread():
    global pose, scale

    rotation_speed = [0, 0]
    rotation_momentum = 0.7

    translation_speed = [0, 0, 0]
    translation_momentum = 0.7

    while True:
        time.sleep(0.005)

        delta = 0.0005 * scale

        rotation_speed[0] = rotation_speed[0] * rotation_momentum + ctrl_state['lx'] * (1 - rotation_momentum)
        rotation_speed[1] = rotation_speed[1] * rotation_momentum + ctrl_state['ly'] * (1 - rotation_momentum)
        translation_speed[0] = translation_speed[0] * translation_momentum - ctrl_state['ry'] * (1 - translation_momentum)
        translation_speed[1] = translation_speed[1] * translation_momentum - ctrl_state['rx'] * (1 - translation_momentum)

        if rotation_speed[0] != 0 or rotation_speed[1] != 0:
            rp1 = (1, 0, 0)
            rs = 0.002
            rp2 = VectorNormalize((1, -rotation_speed[0] * rs, -rotation_speed[1] * rs))
            rp = VectorCross(rp1, rp2)
            if math.sqrt(VectorDot(rp, rp)) < 1e-5:
                rot = (1, 0, 0, 0)
            else:
                alpha = math.acos(VectorDot(rp1, rp2))
                rot = RotationQuaternion(rp, alpha)
        else:
            rot = (1, 0, 0, 0)

        ((x, y, z), (qw, qx, qy, qz)) = pose
        if ctrl_state['rotate_mode'] == 'viewer':
            (dx, dy, dz) = QuaternionRotation(pose[1], ( translation_speed[0] * delta,  translation_speed[1] * delta,  translation_speed[2] * delta))
            x += dx; y += dy; z += dz
            (qw, qx, qy, qz) = QuaternionMul(pose[1], rot)
        else:
            rot = QuaternionMul(QuaternionMul(pose[1], rot), QuaternionConj(pose[1]))
            (x, y, z) = QuaternionRotation(rot, (x, y, z))
            (qw, qx, qy, qz) = QuaternionMul(rot, pose[1])

        qlen = math.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)
        qw /= qlen; qx /= qlen; qy /= qlen; qz /= qlen

        new_pose = ((x, y, z), (qw, qx, qy, qz))

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

th_event = threading.Thread(target = event_thread)
th_event.daemon = True
th_event.start()

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
