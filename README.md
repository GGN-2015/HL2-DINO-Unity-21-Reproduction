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

This project can stream the HoloLens 2 raw 16-bit depth image and raw 16-bit infrared image to a Python TCP server in real time. Each frame contains two `512 x 512` `uint16` images and the depth-to-world matrix. The Unity client sends frames from a background thread so the main Unity update loop is not blocked by TCP transfer. The C# client uses the `simple_tcp_server` L framing mode: one `L` negotiation byte when the socket connects, then `4-byte big-endian payload length + raw payload` for each request and response.

The Unity client detects marker balls locally from the same 16-bit infrared frame using a C# port of `ir_yolo_tracker.ThresholdMarkerDetector`. It then converts those local 2D image coordinates to 3D Unity positions with the same-frame depth image, depth-to-world matrix, and per-marker `MapImagePointToCameraUnitPlane` calls. The full per-pixel unit-plane lookup table is not generated or transmitted.

The TCP stream is still sent to the Python server so the server can show the raw sensor visualization, run its own marker detection for diagnostics, and report server-side response latency. The server still returns a compact `DINOUV01` marker response, but the Unity client intentionally ignores those marker positions. Red semi-transparent marker spheres are driven by Unity-side detection and projection only.

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
  "height": 512,
  "detector": "threshold",
  "threshold_confidence_threshold": 0.25,
  "threshold_percentile": 99.7,
  "threshold_min_threshold": 0,
  "threshold_min_area": 6,
  "threshold_max_area": 3000,
  "threshold_min_circularity": 0.15,
  "threshold_min_aspect_ratio": 0.25,
  "threshold_max_aspect_ratio": 4.0,
  "threshold_min_width": 1,
  "threshold_min_height": 1,
  "threshold_morphology_kernel_size": 0,
  "threshold_morphology_open_iterations": 0,
  "threshold_max_markers": 16
}
```

Use `host` and `port` for the PC network adapter address and TCP port. `width` and `height` must match the raw sensor frame size sent by the HoloLens app. Command-line `--host` and `--port` values still override the JSON defaults for one server run.

For lowest latency, `detector` defaults to `threshold`, which uses the `ir_yolo_tracker.ThresholdMarkerDetector` pure threshold pipeline. Use `--detector yolo` or set `"detector": "yolo"` to switch back to the YOLO detector. The `threshold_*` values tune the fast detector's confidence, threshold, shape, area, and maximum marker count. Existing configs that still use `"detector": "blob"` and `blob_*` keys are accepted as aliases for the threshold detector.

In the `SampleSceneMRTK` scene, the TCP streaming script is bound to the `Managers -> RM_Manager (Research Mode Controller)` object. You can turn raw sensor streaming on or off in this script and configure the IP address and port.

The provided scenes already enable this stream on `Managers/RM_Manager`, which contains the `ResearchModeController` component. The relevant Inspector fields are:

- `Stream Raw Sensor Images Over Tcp`: enable or disable the TCP stream.
- `Sensor Tcp Host`: the Python server IP address. Default: `169.254.83.86`.
- `Sensor Tcp Port`: the Python server port. Default: `8888`.
- `Sensor Tcp Frame Interval Seconds`: target send interval. Default: `0.033333335`, about 30 FPS (24 FPS in real-world use).
- `Sensor Tcp Reconnect Interval Seconds`: reconnect delay after a failed connection attempt.
- `Detect Markers In Unity`: run the local threshold marker detector. Default: enabled.
- `Local Marker Detection Interval Seconds`: target local detection interval. Default: `0.033333335`, about 30 FPS.
- `Local Threshold *`: Unity-side threshold detector parameters. Defaults mirror the server's threshold config.

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

For every received frame, the server converts the depth and infrared payloads to NumPy arrays with shape `(512, 512)`, runs marker detection for diagnostics, and returns marker image coordinates in the TCP response. Unity ignores those returned marker coordinates. Per-frame terminal logging is disabled by default; pass `--print-frame-log` to print the receive timestamp, frame sequence number, client timestamp, FPS, processing time, and array shapes. The server also opens an OpenCV visualization window. The left image is depth, where near pixels are bright and far pixels are dark. The right image is infrared, normalized per frame with min-max scaling. Press `Q`, `Esc`, or `Ctrl+C` in the terminal to stop the server.

The visualization overlay shows average server response latency in milliseconds. This latency is measured from the moment the Python server has received a complete request frame to the moment `sendall()` successfully returns after writing the response to the socket. Pass `--print-response-latency` to print one latency line immediately after each response is sent.

### AimTool Model Tracking

AimTool source files live under the repository-level `AimTools` folder. Treat every `.aimtool` marker coordinate and every matching `.stl` vertex as original right-handed millimetre data; do not edit those source files to make them look like Unity coordinates.

Use `DINO Unity > Import AimTools` in the Unity editor to regenerate runtime assets. The importer parses each `.aimtool`, converts the marker points from right-handed millimetres to Unity left-handed metres, converts the matching `.stl` to an `.obj` under `DINO-Unity-21/Assets/Resources/AimTools`, and compensates for Unity's OBJ import axis handling so the imported mesh lands in the same left-handed coordinate frame as the generated marker JSON. Runtime matching and rendering only use those generated `Resources/AimTools` assets.

Do not manually rotate or re-coordinate the generated AimTool `.obj` files in Unity. The importer has already applied the coordinate conversion needed for AimTool models. The manual OBJ rotation note in the tutorials is only for manually imported custom models, not for AimTool assets generated by this importer.

To add a new model, copy both source files into `AimTools` with the same filename prefix:

```text
AimTools/<model-name>.aimtool
AimTools/<model-name>.stl
```

The `.aimtool` parser ignores the first two lines, reads the third line as the marker count `n`, then reads the next `n` lines as `x y z` marker coordinates in millimetres. A fourth value on those coordinate lines is ignored. Any content after those `n` marker lines is also ignored.

After adding or replacing source files, run `DINO Unity > Import AimTools`. This regenerates `<model-name>.obj`, `<model-name>.markers.json`, and their `.meta` files under `DINO-Unity-21/Assets/Resources/AimTools`. `AimToolModelTracker` loads all marker JSON/model pairs from that Resources folder at runtime, so no scene wiring is needed for each new model.

Runtime AimTool model tracking is separate from the old `StreamingAssets` tool config and from the DINO-DLL tool dictionary. `ResearchModeController` runs Unity-side marker detection on the raw infrared frame, resolves detected marker pixels into 3D world positions using the matching depth frame, and passes those world-space marker positions to `AimToolModelTracker`.

AimTool rendering depends on these runtime conditions:

- `Detect Markers In Unity` must be enabled before the app starts, because the local marker detection thread is created during `ResearchModeController.Start()`.
- The sensor-image update path must be running. If the palm-menu `Toggle Sensor Data` button turns sensor data off, new raw frames are not queued for local marker detection and AimTool poses will stop updating.
- A Python TCP server connection is not required for AimTool pose matching or rendering. The server stream is useful for visualization and diagnostics, but Unity ignores the marker positions returned by the server.
- Each generated marker JSON/model pair must contain at least 3 markers. A model is displayed only when all marker points in that model's template can be matched to the currently observed 3D marker set. Extra observed markers are allowed.
- Multiple AimTool models can be displayed at the same time, but selected matches cannot reuse the same observed marker points.

Useful `AimToolModelTracker` tuning fields are `maxMarkerErrorMetres`, `distanceToleranceMetres`, `maxObservedMarkers`, `maxSearchNodesPerTool`, `lostVisibilityTimeoutSeconds`, `jitterSmoothingDistanceMetres`, `jitterSmoothingFactor`, `hideModelsWhenUnmatched`, and `logMatches`. By default, `distanceToleranceMetres = 0` means the matcher uses `maxMarkerErrorMetres * 2`, and unmatched models are hidden after `lostVisibilityTimeoutSeconds`.

### Saving Raw Sensor Images

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
