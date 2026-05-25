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

> [!IMPORTANT]
> After first build, if failed with a very long compiler output, try to run `fix_win_mobile.py` in the root folder of the current project.

- Configure device portal in Build Settings, keep Hololens2 on, and then click `Build and Run`.
- The project will be packed and then send to your device, it may take about 4min to compile the project.
