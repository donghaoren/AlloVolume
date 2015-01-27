import zmq
from server import controller_client
import subprocess
import readline
import shlex
import traceback
from twisted.internet import reactor
from twisted.internet.task import LoopingCall
import thread, threading

help_string = """
ls            : List available datasets.
select <name> : Select a dataset.
goto <frame>  : Goto a particular frame.
next          : Goto the next frame.
prev          : Goto the previous frame.
play <fps>    : Playback (negative fps means backwards).
step <step>   : Set step size for playback.
stop          : Stop playback.
""".strip()

def globFiles(glob_expr):
    import glob
    return sorted(glob.glob(glob_expr))

class Application:
    def __init__(self):
        zmq_context = zmq.Context()
        controller_client.Initialize(zmq_context)
        config = controller_client.config
        datasets = config['allovolume']['datasets']
        name_map = { }

        for volume in datasets:
            volume['files'] = globFiles(volume['glob'])

            print "Dataset: %s" % volume['name']
            print "    %d frame(s)" % len(volume['files'])
            print "    " + volume['description'].replace("\n", "\n    ")

            name_map[volume['name'].lower()] = volume

        self.name_map = name_map
        self.datasets = datasets
        self.config = config
        self.volume_filename = None
        self.current_volume = None
        self.timer = None
        self.step_size = 1

    def setCurrentFrame(self, frame_index):
        self.current_frame = frame_index % len(self.current_volume['files'])
        volume_filename = self.current_volume['files'][self.current_frame]
        if self.volume_filename != volume_filename:
            self.volume_filename = volume_filename
            print volume_filename
            #controller_client.LoadVolume(volume_filename, self.current_volume['name'], self.current_volume['description'])

    def setCurrentVolume(self, volume):
        self.current_volume = volume
        self.current_frame = None

    def tick(self, reverse):
        if reverse:
            self.setCurrentFrame(self.current_frame - self.step_size)
        else:
            self.setCurrentFrame(self.current_frame + self.step_size)

    def playback(self, fps, reverse = False):
        if self.timer:
            self.timer.stop()
        self.timer = LoopingCall(self.tick, reverse = reverse)
        self.timer.start(1.0 / fps)

    def stopPlayback(self):
        if self.timer:
            self.timer.stop()
            self.timer = None

    def process_command(self, cmd, line):
        try:
            if cmd == "ls":
                if len(line) == 2 and self.current_volume:
                    print "\n".join(self.current_volume['files'])
                else:
                    print "\n".join(sorted(self.name_map.keys()))

            if cmd == "help":
                print help_string

            if cmd == "select":
                name, = line[1:]
                if not name in self.name_map:
                    print "Invalid dataset '%s'" % name
                else:
                    self.setCurrentVolume(self.name_map[name])

            if cmd == "goto":
                if not self.current_volume: self.setCurrentVolume(self.datasets[0])
                frame_index, = line[1:]
                frame_index = int(frame_index)
                if frame_index >= 0 and frame_index < len(self.current_volume['files']):
                    self.setCurrentFrame(frame_index)
                else:
                    print "Frame index out of range (0 - %d)" % (len(self.current_volume['files']) - 1)

            if cmd == "stop":
                self.stopPlayback()

            if cmd == "step":
                step, = line[1:]
                self.step_size = int(step)

            if cmd == "next":
                if not self.current_volume: self.setCurrentVolume(self.datasets[0])
                self.setCurrentFrame(self.current_frame + 1)

            if cmd == "prev":
                if not self.current_volume: self.setCurrentVolume(self.datasets[0])
                self.setCurrentFrame(self.current_frame - 1)

            if cmd == "play":
                if not self.current_volume: self.setCurrentVolume(self.datasets[0])
                fps, = line[1:]
                fps = float(fps)
                if fps > 0:
                    self.playback(fps)
                if fps < 0:
                    self.playback(-fps, reverse = True)

        except:
            traceback.print_exc()

thread.start_new_thread(reactor.run, (), { "installSignalHandlers": False })

def initialize(waiter):
    global app
    app = Application()
    waiter.set()

def process_command(cmd, line, waiter):
    global app
    app.process_command(cmd, line)
    waiter.set()

waiter = threading.Event()
reactor.callFromThread(initialize, waiter)
waiter.wait()

while True:
    try:
        line = shlex.split(raw_input('[allovolume.dataset]: '))
    except (KeyboardInterrupt, IOError, EOFError):
        break
    if len(line) == 0: continue
    cmd = line[0].lower()
    if cmd == 'exit' or cmd == 'quit': exit()

    waiter = threading.Event()
    reactor.callFromThread(process_command, cmd, line, waiter)
    if not waiter.wait(1):
        print "Warning: the command is still running..."

print ""
exit()
