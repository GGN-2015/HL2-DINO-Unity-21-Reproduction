# HL2-DINO-Unity-21-Reproduction
Reproduction project HL2-DINO with Unity 2021 platform。

Original Project
- https://github.com/HL2-DINO
- https://github.com/HL2-DINO/DINO-Unity/tree/unity-21

## Steps

> [!IMPORTANT]
> This project only works for Hololens2 and MRTK3

### Preparation of Tool Chains

> [!WARNING]
> You can only use `Unity Editor 2021.x` to reproduce this project.

- Download Unity Hub
- Install Unity Editor 2021.x LTS via Unity Hub
    - Install module `Visual Studio 2019` of Unity Editor 2021.x
    - Install module `Universal Windows Platform Build Support` of Unity Editor 2021.x
- Open Project `DINO-Unity-21` with Unity Editor 2021.x LTS

### Prepare Hololens2

- Configure Device Portal:
    - https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/using-the-windows-device-portal
- Configure Research Mode and Sensor Streaming
    - https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/research-mode

### Configure Project

- Checkout Sence `Scences\SampleSceneMRTK.unity`
- Configure project, see: https://github.com/HL2-DINO/DINO-Unity/tree/unity-21#getting-started

### Raw Sensor TCP Streaming

This project can stream the HoloLens 2 raw 16-bit depth image and raw 16-bit infrared image to a Python TCP server in real time. Each frame contains two `512 x 512` `uint16` images. The Unity client sends frames from a background thread so the main Unity update loop is not blocked by TCP transfer. The C# client uses the `simple_tcp_server` L framing mode: one `L` negotiation byte when the socket connects, then `4-byte big-endian payload length + raw payload` for each request and response.

The default server address is:

```text
169.254.83.86:8888
```

The provided scenes already enable this stream on `Managers/RM_Manager`, which contains the `ResearchModeController` component. The relevant Inspector fields are:

- `Stream Raw Sensor Images Over Tcp`: enable or disable the TCP stream.
- `Sensor Tcp Host`: the Python server IP address. Default: `169.254.83.86`.
- `Sensor Tcp Port`: the Python server port. Default: `8888`.
- `Sensor Tcp Frame Interval Seconds`: target send interval. Default: `0.033333335`, about 30 FPS (24 FPS in real world).
- `Sensor Tcp Reconnect Interval Seconds`: reconnect delay after a failed connection attempt.

Start the Python server from the repository root before launching the HoloLens app:

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade numpy opencv-python simple-tcp-server
venv\Scripts\python.exe DINO-Unity-21\Assets\SampleServer\RawSensorImageServer.py
```

When the server starts successfully, it prints:

```text
Raw sensor server listening on 169.254.83.86:8888
```

When the HoloLens client connects, it prints a line like:

```text
[2026-06-06 22:00:00.000000] Client connected from <client-ip>:<client-port>
```

For every received frame, the server converts the depth and infrared payloads to NumPy arrays with shape `(512, 512)` and prints the receive timestamp, frame sequence number, client timestamp, FPS, and array shapes. It also opens an OpenCV visualization window. The left image is depth, where near pixels are bright and far pixels are dark. The right image is infrared, normalized per frame with min-max scaling. Press `Q`, `Esc`, or `Ctrl+C` in the terminal to stop the server.

You can also import the server module from your own Python code and read the latest raw images directly:

```python
import sys
import time
from pathlib import Path

sys.path.append(str(Path("DINO-Unity-21/Assets/SampleServer").resolve()))

from RawSensorImageServer import RawSensorImageReceiver

receiver = RawSensorImageReceiver(host="169.254.83.86", port=8888)
receiver.start()

try:
    while True:
        images = receiver.wait_for_next_images(timeout=1.0)
        if images is None:
            print("Waiting for HoloLens frames...")
            continue

        depth, infrared = images
        print(depth.shape, depth.dtype, infrared.shape, infrared.dtype)
        time.sleep(0.01)
finally:
    receiver.stop()
```

The receiver API returns NumPy arrays with dtype `uint16`:

- `receiver.get_current_depth_image()`: returns the latest depth image, or `None` before the first frame arrives.
- `receiver.get_current_infrared_image()`: returns the latest infrared image, or `None` before the first frame arrives.
- `receiver.get_current_images()`: returns `(depth, infrared)`, or `None` before the first frame arrives.
- `receiver.wait_for_images(timeout=1.0)`: waits until at least one frame is available and returns `(depth, infrared)`, or `None` on timeout.
- `receiver.wait_for_next_images(timeout=1.0)`: waits for a newer frame than the current one and returns `(depth, infrared)`, or `None` on timeout.
- `receiver.get_current_frame()`: returns metadata plus the two images. The metadata includes `sequence`, `client_timestamp`, and `received_timestamp`.

By default the getter methods return copies so caller code cannot accidentally mutate the receiver's live frame. Pass `copy=False` if you need lower overhead and will treat the arrays as read-only.

If the server cannot bind to `169.254.83.86`, make sure that this IP exists on the PC network adapter connected to the HoloLens. If you use another server IP, update both places:

1. `HOST` in `DINO-Unity-21/Assets/SampleServer/RawSensorImageServer.py`
1. `Sensor Tcp Host` on `Managers/RM_Manager` in the Unity scene

If no client connection appears in the server terminal:

1. Confirm the HoloLens app was rebuilt and redeployed after the TCP changes.
1. Confirm `Stream Raw Sensor Images Over Tcp` is enabled on `Managers/RM_Manager`.
1. Confirm the HoloLens can reach the PC IP and port.
1. Allow `python.exe` through Windows Firewall on the active network.
1. Check the in-app debug text for messages such as `TCP connecting...` or `TCP connection failed...`.

> [!TIP]
> If you see compile error like:
>
> ```
> InvalidOperationException: Certificate Assets\WSATestCertificate.pfx is expired
> and cannot be used for a UWP build. To fix this, either delete it or select a
> different certificate in the player settings.
> ```
> 
> Delete file `Assets\WSATestCertificate.pfx` and then select `None` as Certificate in:
> - Edit -> Project Settings -> Player -> UWP -> Publishing Settings

> [!IMPORTANT]
> After first build, if failed with a very long compiler output, try to run `fix_win_mobile.py` in the root folder of the current project.

- Configure device portal in Build Settings, keep Hololens2 on, and then click `Build and Run`.
- The project will be packed and then send to your device, it may take about 6min to compile the project.
