# HL2-DINO-Unity-21-Reproduction
A reproduction of the HL2-DINO project on the Unity 2021 platform.

Original projects:
- https://github.com/HL2-DINO
- https://github.com/HL2-DINO/DINO-Unity/tree/unity-21

## Steps

> [!IMPORTANT]
> This project only works with HoloLens 2 and MRTK3.

### Prepare Toolchains

> [!WARNING]
> Use only `Unity Editor 2021.x` to reproduce this project.

- Download Unity Hub
- Install Unity Editor 2021.x LTS via Unity Hub
    - Install the `Visual Studio 2019` module for Unity Editor 2021.x
    - Install the `Universal Windows Platform Build Support` module for Unity Editor 2021.x
- Open the `DINO-Unity-21` project with Unity Editor 2021.x LTS

### Prepare HoloLens 2

- Configure Device Portal:
    - https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/using-the-windows-device-portal
- Configure Research Mode and Sensor Streaming:
    - https://learn.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/research-mode

### Configure Project

- Check out the scene `Scenes\SampleSceneMRTK.unity`
- Configure the project. See: https://github.com/HL2-DINO/DINO-Unity/tree/unity-21#getting-started

### Raw Sensor TCP Streaming

This project can stream the HoloLens 2 raw 16-bit depth image and raw 16-bit infrared image to a Python TCP server in real time. Each frame contains two `512 x 512` `uint16` images. The Unity client sends frames from a background thread so the main Unity update loop is not blocked by TCP transfer. The C# client uses the `simple_tcp_server` L framing mode: one `L` negotiation byte when the socket connects, then `4-byte big-endian payload length + raw payload` for each request and response.

The default server address is:

```text
169.254.83.86:8888
```

The Python server reads its default bind address and image size from `DINO-Unity-21/Assets/SampleServer/RawSensorImageServerConfig.json`:

```json
{
  "host": "169.254.83.86",
  "port": 8888,
  "width": 512,
  "height": 512
}
```

Use `host` and `port` for the PC network adapter address and TCP port. `width` and `height` must match the raw sensor frame size sent by the HoloLens app. Command-line `--host` and `--port` values still override the JSON defaults for one server run.

In the `SampleSceneMRTK` scene, the TCP streaming script is bound to the `Managers -> RM_Manager (Research Mode Controller)` object. You can turn raw sensor streaming on or off in this script and configure the IP address and port.

The provided scenes already enable this stream on `Managers/RM_Manager`, which contains the `ResearchModeController` component. The relevant Inspector fields are:

- `Stream Raw Sensor Images Over Tcp`: enable or disable the TCP stream.
- `Sensor Tcp Host`: the Python server IP address. Default: `169.254.83.86`.
- `Sensor Tcp Port`: the Python server port. Default: `8888`.
- `Sensor Tcp Frame Interval Seconds`: target send interval. Default: `0.033333335`, about 30 FPS (24 FPS in real-world use).
- `Sensor Tcp Reconnect Interval Seconds`: reconnect delay after a failed connection attempt.

Start the Python server from the repository root before launching the HoloLens app:

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade numpy opencv-python simple-tcp-server
venv\Scripts\python.exe DINO-Unity-21\Assets\SampleServer\RawSensorImageServer.py
```

To save every received raw depth and infrared image, start the server with `--save-raw-images`:

```powershell
venv\Scripts\python.exe DINO-Unity-21\Assets\SampleServer\RawSensorImageServer.py --save-raw-images
```

You can also enable saving by setting `SAVE_RAW_IMAGES_TO_PICKLE = True` in `DINO-Unity-21/Assets/SampleServer/RawSensorImageServer.py`, or by passing `save_raw_images=True` when creating `RawSensorImageReceiver`.

When the server starts successfully, it prints:

```text
Raw sensor server listening on 169.254.83.86:8888
```

When the HoloLens client connects, it prints a line like:

```text
[2026-06-06 22:00:00.000000] Client connected from <client-ip>:<client-port>
```

For every received frame, the server converts the depth and infrared payloads to NumPy arrays with shape `(512, 512)`. Per-frame terminal logging is disabled by default; pass `--print-frame-log` to print the receive timestamp, frame sequence number, client timestamp, FPS, processing time, and array shapes. The server also opens an OpenCV visualization window. The left image is depth, where near pixels are bright and far pixels are dark. The right image is infrared, normalized per frame with min-max scaling. Press `Q`, `Esc`, or `Ctrl+C` in the terminal to stop the server.

When raw image saving is enabled, the server creates the `IRData` folder in the project root if it does not already exist. Saving runs on a background writer thread so disk I/O does not block the TCP server receive loop. Depth images are saved under `IRData/depth`, and infrared images are saved under `IRData/infrared`. Each file contains one NumPy `uint16` array with shape `(512, 512)`, serialized with Python `pickle`. File names use a 7-digit decimal counter with leading zeros, such as `0000001.pickle`, `0000002.pickle`, and so on. The depth and infrared files with the same number belong to the same received frame.

You can load a saved image like this:

```python
import pickle

with open("IRData/depth/0000001.pickle", "rb") as file:
    depth = pickle.load(file)

print(depth.shape, depth.dtype)
```

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

By default, the getter methods return copies so caller code cannot accidentally mutate the receiver's live frame. Pass `copy=False` if you need lower overhead and will treat the arrays as read-only.

If the server cannot bind to `169.254.83.86`, make sure that this IP exists on the PC network adapter connected to the HoloLens. If you use another server IP, update both places:

1. `host` in `DINO-Unity-21/Assets/SampleServer/RawSensorImageServerConfig.json`, or pass `--host <ip>` when starting the server.
1. `Sensor Tcp Host` on `Managers/RM_Manager` in the Unity scene

If no client connection appears in the server terminal:

1. Confirm the HoloLens app was rebuilt and redeployed after the TCP changes.
1. Confirm `Stream Raw Sensor Images Over Tcp` is enabled on `Managers/RM_Manager`.
1. Confirm the HoloLens can reach the PC IP and port.
1. Allow `python.exe` through Windows Firewall on the active network.
1. Check the in-app debug text for messages such as `TCP connecting...` or `TCP connection failed...`.

> [!TIP]
> If you see a compile error like:
>
> ```
> InvalidOperationException: Certificate Assets\WSATestCertificate.pfx is expired
> and cannot be used for a UWP build. To fix this, either delete it or select a
> different certificate in the player settings.
> ```
>
> Delete the file `Assets\WSATestCertificate.pfx`, and then select `None` as the certificate in:
> - Edit -> Project Settings -> Player -> UWP -> Publishing Settings

> [!IMPORTANT]
> After the first build, if the build fails with a very long compiler output, try running `fix_win_mobile.py` in the root folder of the current project.

- Configure Device Portal in Build Settings, keep HoloLens 2 on, and then click `Build and Run`.
- The project will be packaged and sent to your device. It may take about 6 minutes to compile the project.
