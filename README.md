AlloVolume
====

Author: Donghao Ren, Four Eyes Lab and AlloSphere Research Group, UCSB.

AlloVolume is a CUDA-based application to support full-surround stereo volume rendering with interactive navigation for the UCSB Allosphere.
It supports AMR (Adaptive Mesh Refinement) datasets.

AlloVolume's components:

- CUDA-based AMR Ray-casting Volume Renderer.
    - KD-Tree for fast ray-block intersection.
    - Customizable lens.
    - High definintion rendering using adaptive Runge-Kutta-Fehlberg method.

- HTML5-based Parameter Editor.
    - Transfer function editing.
    - Set rendering parameters and lens parameters.
    - Works across multiple devices (iPad / Laptop).

- Renderer / Controller for the UCSB Allosphere infstructure.
    - Read Allosphere calibration data, perform warping and blending (mono/stereo).
    - Multi-threaded rendering using the two available GPUs in each rendering machine.
    - Synchronizing renderers.
    - Interactive navigation in the volume.
    - ZeroMQ's EPGM-based broadcasting of datasets and rendering messages.
    - Websocket server for the HTML5-based parameter editor.

Building
----

Install dependencies first:

- CUDA
- Boost
- ZeroMQ
- yaml-cpp
- Protobuf
- FreeImage
- freeglut (linux)

Build:

    cd proto
    make
    cd ..
    cd controller
    make
    cd ..
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make

Launching
----

### Launch in the Allosphere:

1. Copy `renderer` and `controller` to all rendering machines (or to the shared filesystem).
2. Write `allovolume.yaml` in the same directory.
3. Copy the datasets to somewhere accessible to the `controller`.
4. Launch `controller` on GR01.
5. Launch `renderer` on GR02-GR14 (with multishell or other methods).
6. Launch `controller/server/server.py` with python for the HTML5-based interface.
   - `cd controller` then `python server/server.py`, the same `allovolume.yaml` should present in the `controller` directory.
7. Launch `controller/server/osc_pose.py` to delegate pose messages from the DeviceServer and MaxMSP (currently parses PanoView's messages).

### allovolume.yaml:

    sync:
      # Broadcast endpoint and feedback endpoint.
      broadcast: epgm://[endpoint address];[multicast address (224.0.0.1)]:[port]
      feedback: tcp://[ip-address]:[port]

      # ZeroMQ parameters.
      zmq:
        rcvhwm: 50000
        sndhwm: 50000
        rcvbuf: 100000000
        sndbuf: 100000000
        rate: 10000000
        delay: 150000

    allovolume:
      # Endpoint where controller listen for requests.
      controller: tcp://[ip-address]:[port]
      # Endpoint where controller send parameter update events.
      events: tcp://[ip-address]:[port]

      fullscreen: true
      stereo: true
      # Rendering resolution for each projector.
      resolution:
        width: 600
        height: 400

    [hostname]:
      # Machine-specific overrides here.
      sync:
        braodcast: ...

License
----

BSDv3 License.
