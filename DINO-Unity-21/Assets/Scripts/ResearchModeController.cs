using UnityEngine;
using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using UnityEngine.XR.WSA;
using Newtonsoft.Json.Linq;

#if !UNITY_EDITOR && UNITY_WSA
using Windows.Perception.Spatial;
using Windows.Storage;
using System.Threading.Tasks;
#endif

#if ENABLE_WINMD_SUPPORT
using HL2DinoPlugin;
#endif

/** @file           ResearchModeController.cs
 *  @brief          Main controller/interaction point in Unity for the HL2-DINO .dll
 *  
 *  @details        This will grab sensor images and tool-pose data from the C++ side and use it to pass
 *                  along to any interested parties in Unity
 * 
 *  @note           The logic for setting up the grayscale sensor image structure and grabbing image data from 
 *                  the C++ DLL is adapted from petergu684's HoloLens2-ResearchMode-Unity on GitHub.
 *                  Check it out for completeness!
 *
 *  @author         Hisham Iqbal
 *  @copyright      &copy; 2023 Hisham Iqbal
 */

public class ResearchModeController : MonoBehaviour
{
#if ENABLE_WINMD_SUPPORT
    /// <summary>
    /// The C# interface to the HL2-DINO DLL interface, generated with WinRT. Functions exposed by this
    /// mirror the contents of the .idl file in the DLL project
    /// </summary>
    HL2ResearchModeController researchMode;
#endif

    //! @name Unity variables for visualising sensor images
    //!@{
    public GameObject depthPreviewPlane = null;
    private Material depthMediaMaterial = null;
    private Texture2D depthMediaTexture = null;

    public GameObject abImagePreviewPlane = null;
    private Material abImageMediaMaterial = null;
    private Texture2D abMediaTexture = null;

    public TMPro.TextMeshProUGUI ConsoleDebugTextMesh;

    [Header("Raw Sensor TCP Stream")]
    public bool streamRawSensorImagesOverTcp = true;
    public string sensorTcpHost = "169.254.83.86";
    public int sensorTcpPort = 8888;
    public float sensorTcpFrameIntervalSeconds = 1f / 30f;
    public float sensorTcpReconnectIntervalSeconds = 1.0f;

    private const int SensorImageWidth = 512;
    private const int SensorImageHeight = 512;
    private const int SensorImagePixelCount = SensorImageWidth * SensorImageHeight;
    private const int SensorTcpHeaderV4BaseBytes = 48;
    private const int SensorTcpDepthToWorldMatrixValues = 16;
    private const int SensorTcpUnitPlaneValuesPerPixel = 2;
    private const float Raw16MaxValue = 65535f;
    private const float DepthDisplayBlackMm = 1f;
    private const float DepthDisplayWhiteMm = 4090f;
    private const float SensorImageUploadIntervalSeconds = 1f / 30f;
    private static readonly byte[] SensorTcpRawStreamPrefix = Encoding.ASCII.GetBytes("raw_stream:");
    private static readonly byte[] SensorTcpPayloadMagic = Encoding.ASCII.GetBytes("DINOIMG4");
    private static readonly byte[] MarkerWorldResponseMagic = Encoding.ASCII.GetBytes("DINOXYZ1");
    private const float MarkerSphereDiameterMetres = 0.01f;

    private static readonly int SourceChannelId = Shader.PropertyToID("_SourceChannel");
    private static readonly int InputMaxValueId = Shader.PropertyToID("_InputMaxValue");
    private static readonly int DisplayBlackValueId = Shader.PropertyToID("_DisplayBlackValue");
    private static readonly int DisplayWhiteValueId = Shader.PropertyToID("_DisplayWhiteValue");
    private static readonly int ClampMaxValueId = Shader.PropertyToID("_ClampMaxValue");
    private static readonly int OutputBlackValueId = Shader.PropertyToID("_OutputBlackValue");
    private static readonly int InvertOutputId = Shader.PropertyToID("_InvertOutput");

    private float nextSensorImageUploadTime = 0f;
    private bool sensorImageStreamReady = false;
    private float nextSensorTcpQueueTime = 0f;
    private ulong sensorTcpFrameSequence = 0;
    private volatile bool sensorTcpRunning = false;
    private Thread sensorTcpThread = null;
    private SimpleTcpClient sensorTcpClient = null;
    private byte[] sensorTcpPayloadBuffer = null;
    private float[] sensorTcpUnitPlaneBuffer = null;
    private bool sensorTcpUnitPlaneBufferReady = false;
    private bool sensorTcpUnitPlaneMapSent = false;
    private RawSensorTcpFrame latestSensorTcpFrame = null;
    private readonly object sensorTcpFrameLock = new object();
    private readonly object sensorTcpStatusLock = new object();
    private string sensorTcpStatusMessage = string.Empty;
    private bool sensorTcpStatusDirty = false;
    private readonly object markerWorldPositionsLock = new object();
    private List<Vector3> latestMarkerWorldPositions = new List<Vector3>();
    private bool newMarkerWorldPositionsReceived = false;
    private readonly List<GameObject> markerWorldSpheres = new List<GameObject>();
    private Material markerSphereMaterial = null;

    /// <summary>
    /// Use to internally track if sensor images are updated, but also pass this into \p HL2ResearchMode to 
    /// tell it if it should continue/stop processing sensor images for display purposes.
    /// </summary>
    bool SensorImagesDisplaying = true;
    //!@}

    /// <summary>
    /// Should be set from inspector. Class is responsible for using tool pose data obtained from 
    /// the DLL to set GameObject transforms in Unity.
    /// </summary>
    public UnityToolManager ToolManagerScript;

    public string JSONFilename = "toolConfig.json";
    string JSONStorageFolder = "";

    // Start is called before the first frame update
    void Start()
    {
        // (1) Caching/attaching Unity objects
        ImageTexturesSetup();
        StartSensorTcpThread();

        // (2) Reading tool config data from file
        string toolConfigJSONString = ToolConfigJSONSetup();

        // (3) Launching HL2-DINO DLL if all is ok
        if (!string.IsNullOrEmpty(toolConfigJSONString))
        {
            // only set up with a valid string to avoid crashing
            ResearchModeSetup(toolConfigJSONString);
        }
    }

    /// <summary>
    /// Update class member vars to point to correct locations for where the tool config JSON file is
    /// </summary>
    private string ToolConfigJSONSetup()
    {
        // using this as a stand-in location, as StreamingAssets is compiled into the app.
        // Future TODO: explore reading directly from some headset location so you can change
        // tool config data at runtime, and without re-compiling
        JSONStorageFolder = Application.streamingAssetsPath;

        string toolConfigJSONString = ToolConfigUtilities.JSONUtils.GetJSONToolStringHL2(JSONStorageFolder + "/" + JSONFilename);
        if (!string.IsNullOrEmpty(toolConfigJSONString))
        {
            // if we get here, then the string should be properly JSON formatted
            // but it still will need to pass the same checks on the CPP side
#if UNITY_EDITOR
            print(toolConfigJSONString);
#endif
            return toolConfigJSONString;
        }
        else
        {
            ConsoleDebugTextMesh.text = $"{JSONFilename} not a valid JSON construct for this app";
            return string.Empty;
        }
    }

    /// <summary>
    /// Function which initialises DLL functions and instructs DLL which tools to track
    /// </summary>
    /// <param name="toolsetString">Ideally a JSON-formatted string of marker/tool triplet locations</param>
    private void ResearchModeSetup(string toolsetString)
    {
#if ENABLE_WINMD_SUPPORT
        researchMode = new HL2ResearchModeController(toolsetString, true);
        researchMode.InitialiseDepthSensor();

        if (!SetupLocator()) // call will try to sync the world frame of Unity to the DLL
        {
            // if we're here, we failed to grab/pass locator information
            ConsoleDebugTextMesh.text = "App could not find a SpatialLocator";
            return;
        }

        // all good, so launch the DLL's depth sensor tool-detection loop
        researchMode.StartDepthSensorLoop();
#endif
    }

    /// <summary>
    /// Attaching Unity GameObjects to their corresponding textures used to visualise images passed out of the DLL
    /// </summary>
    private void ImageTexturesSetup()
    {
        depthMediaMaterial = depthPreviewPlane.GetComponent<MeshRenderer>().material;
        depthMediaTexture = new Texture2D(SensorImageWidth, SensorImageHeight, TextureFormat.R16, false, true);
        depthMediaMaterial.mainTexture = depthMediaTexture;
        ConfigureRaw16GrayscaleMaterial(depthMediaMaterial, DepthDisplayBlackMm, DepthDisplayWhiteMm, DepthDisplayWhiteMm, 0f, true);

        abImageMediaMaterial = abImagePreviewPlane.GetComponent<MeshRenderer>().material;
        abMediaTexture = new Texture2D(SensorImageWidth, SensorImageHeight, TextureFormat.R16, false, true);
        abImageMediaMaterial.mainTexture = abMediaTexture;
        ConfigureRaw16GrayscaleMaterial(abImageMediaMaterial, 0f, Raw16MaxValue, Raw16MaxValue, 0f, false);

    }

    private void ConfigureRaw16GrayscaleMaterial(Material material, float displayBlackValue, float displayWhiteValue, float clampMaxValue, float outputBlackValue, bool invertOutput)
    {
        material.SetFloat(SourceChannelId, 0f);
        material.SetFloat(InputMaxValueId, Raw16MaxValue);
        material.SetFloat(DisplayBlackValueId, displayBlackValue);
        material.SetFloat(DisplayWhiteValueId, displayWhiteValue);
        material.SetFloat(ClampMaxValueId, clampMaxValue);
        material.SetFloat(OutputBlackValueId, outputBlackValue);
        material.SetFloat(InvertOutputId, invertOutput ? 1f : 0f);
    }

    private bool LoadRaw16TextureData(Texture2D texture, ushort[] frameData)
    {
        if (frameData == null || frameData.Length < SensorImagePixelCount) return false;

        texture.SetPixelData(frameData, 0);
        texture.Apply(false);
        return true;
    }

    private void UpdateInfraredDisplayRange(ushort[] frameData)
    {
        if (frameData == null || frameData.Length < SensorImagePixelCount) return;

        ushort minValue = ushort.MaxValue;
        ushort maxValue = ushort.MinValue;

        for (int i = 0; i < SensorImagePixelCount; ++i)
        {
            ushort value = frameData[i];
            if (value < minValue) minValue = value;
            if (value > maxValue) maxValue = value;
        }

        if (maxValue <= minValue)
        {
            if (maxValue < ushort.MaxValue) maxValue++;
            else minValue--;
        }

        ConfigureRaw16GrayscaleMaterial(abImageMediaMaterial, minValue, maxValue, maxValue, 0f, false);
    }

    private void StartSensorTcpThread()
    {
        if (!streamRawSensorImagesOverTcp || sensorTcpThread != null) return;

        sensorTcpRunning = true;
        sensorTcpThread = new Thread(SensorTcpBackgroundLoop)
        {
            IsBackground = true,
            Name = "DINO Raw Sensor TCP Sender"
        };
        sensorTcpThread.Start();
    }

    private void StopSensorTcpThread()
    {
        sensorTcpRunning = false;

        lock (sensorTcpFrameLock)
        {
            Monitor.PulseAll(sensorTcpFrameLock);
        }

        try { sensorTcpClient?.Close(); }
        catch { }

        if (sensorTcpThread != null && sensorTcpThread.IsAlive)
        {
            sensorTcpThread.Join(1000);
        }

        sensorTcpThread = null;
        latestSensorTcpFrame = null;
        sensorTcpPayloadBuffer = null;
        sensorTcpUnitPlaneBufferReady = false;
        sensorTcpUnitPlaneMapSent = false;
    }

    private void SetSensorTcpStatus(string message)
    {
        lock (sensorTcpStatusLock)
        {
            if (sensorTcpStatusMessage == message) return;

            sensorTcpStatusMessage = message;
            sensorTcpStatusDirty = true;
        }
    }

    private void ApplySensorTcpStatusToScreen()
    {
        string message = null;

        lock (sensorTcpStatusLock)
        {
            if (!sensorTcpStatusDirty) return;

            message = sensorTcpStatusMessage;
            sensorTcpStatusDirty = false;
        }

        if (ConsoleDebugTextMesh != null && !string.IsNullOrEmpty(message))
        {
            ConsoleDebugTextMesh.text = message;
        }
    }

    private void QueueRawSensorFrameForTcp(ushort[] depthFrame, ushort[] infraredFrame, double[] depthToWorldMatrix)
    {
        if (!streamRawSensorImagesOverTcp || !sensorTcpRunning) return;
        if (depthFrame == null || infraredFrame == null) return;
        if (depthFrame.Length < SensorImagePixelCount || infraredFrame.Length < SensorImagePixelCount) return;
        if (depthToWorldMatrix == null || depthToWorldMatrix.Length < SensorTcpDepthToWorldMatrixValues) return;
        if (!EnsureSensorTcpUnitPlaneBuffer()) return;
        if (Time.unscaledTime < nextSensorTcpQueueTime) return;

        float frameInterval = Mathf.Clamp(sensorTcpFrameIntervalSeconds, 0f, SensorImageUploadIntervalSeconds);
        nextSensorTcpQueueTime = Time.unscaledTime + frameInterval;

        double[] matrixCopy = new double[SensorTcpDepthToWorldMatrixValues];
        Array.Copy(depthToWorldMatrix, matrixCopy, SensorTcpDepthToWorldMatrixValues);

        RawSensorTcpFrame frame = new RawSensorTcpFrame
        {
            Depth = depthFrame,
            Infrared = infraredFrame,
            DepthToWorldMatrix = matrixCopy,
            UnitPlaneMap = sensorTcpUnitPlaneBuffer,
            IncludeUnitPlaneMap = !sensorTcpUnitPlaneMapSent,
            Sequence = ++sensorTcpFrameSequence,
            TimestampUnixSeconds = GetUnixTimeSeconds()
        };

        lock (sensorTcpFrameLock)
        {
            latestSensorTcpFrame = frame;
            Monitor.Pulse(sensorTcpFrameLock);
        }
    }

    private RawSensorTcpFrame WaitForLatestSensorTcpFrame()
    {
        lock (sensorTcpFrameLock)
        {
            while (sensorTcpRunning && latestSensorTcpFrame == null)
            {
                Monitor.Wait(sensorTcpFrameLock, 250);
                if (sensorTcpClient != null && !sensorTcpClient.IsConnected)
                {
                    return null;
                }
            }

            if (!sensorTcpRunning) return null;

            RawSensorTcpFrame frame = latestSensorTcpFrame;
            latestSensorTcpFrame = null;
            return frame;
        }
    }

    private bool EnsureSensorTcpUnitPlaneBuffer()
    {
        if (sensorTcpUnitPlaneBufferReady && sensorTcpUnitPlaneBuffer != null) return true;

#if ENABLE_WINMD_SUPPORT
        if (researchMode == null) return false;

        int valueCount = SensorImagePixelCount * SensorTcpUnitPlaneValuesPerPixel;
        if (sensorTcpUnitPlaneBuffer == null || sensorTcpUnitPlaneBuffer.Length != valueCount)
        {
            sensorTcpUnitPlaneBuffer = new float[valueCount];
        }

        int offset = 0;
        for (int y = 0; y < SensorImageHeight; ++y)
        {
            for (int x = 0; x < SensorImageWidth; ++x)
            {
                float[] unitPlane = researchMode.MapImagePointToCameraUnitPlane((float)x, (float)y);
                if (unitPlane == null || unitPlane.Length < SensorTcpUnitPlaneValuesPerPixel) return false;

                sensorTcpUnitPlaneBuffer[offset++] = unitPlane[0];
                sensorTcpUnitPlaneBuffer[offset++] = unitPlane[1];
            }
        }

        sensorTcpUnitPlaneBufferReady = true;
        return true;
#else
        return false;
#endif
    }

    private void SensorTcpBackgroundLoop()
    {
        while (sensorTcpRunning)
        {
            if (!EnsureSensorTcpConnected())
            {
                continue;
            }

            RawSensorTcpFrame frame = WaitForLatestSensorTcpFrame();
            if (frame == null) continue;

            byte[] payload = BuildRawSensorTcpPayload(frame, ref sensorTcpPayloadBuffer);
            byte[] response = sensorTcpClient.Request(payload);
            if (response == null)
            {
                string reason = GetSensorTcpErrorReason(sensorTcpClient);
                DisconnectSensorTcpClient();
                SetSensorTcpStatus($"TCP connection failed: {reason}. Retrying {sensorTcpHost}:{sensorTcpPort} in {GetSensorTcpReconnectIntervalSeconds():0.0}s.");
                SleepSensorTcpReconnectInterval();
                continue;
            }

            if (IsErrorSensorTcpResponse(response))
            {
                string responseText = Encoding.ASCII.GetString(response);
                DisconnectSensorTcpClient();
                SetSensorTcpStatus($"TCP connection failed: server returned unexpected response '{responseText}'. Retrying {sensorTcpHost}:{sensorTcpPort} in {GetSensorTcpReconnectIntervalSeconds():0.0}s.");
                SleepSensorTcpReconnectInterval();
                continue;
            }

            try
            {
                List<Vector3> markerWorldPositions = ParseServerMarkerWorldResponse(response);
                lock (markerWorldPositionsLock)
                {
                    latestMarkerWorldPositions = markerWorldPositions;
                    newMarkerWorldPositionsReceived = true;
                }
                sensorTcpUnitPlaneMapSent = true;
            }
            catch (Exception ex)
            {
                DisconnectSensorTcpClient();
                SetSensorTcpStatus($"TCP marker response parse failed: {ex.Message}. Retrying {sensorTcpHost}:{sensorTcpPort} in {GetSensorTcpReconnectIntervalSeconds():0.0}s.");
                SleepSensorTcpReconnectInterval();
            }
        }

        DisconnectSensorTcpClient();
        sensorTcpPayloadBuffer = null;
    }

    private bool EnsureSensorTcpConnected()
    {
        if (sensorTcpClient != null && sensorTcpClient.IsConnected) return true;

        DisconnectSensorTcpClient();

        string host = sensorTcpHost;
        int port = sensorTcpPort;
        float retrySeconds = GetSensorTcpReconnectIntervalSeconds();

        SetSensorTcpStatus($"TCP connecting to {host}:{port}...");

        SimpleTcpClient client = new SimpleTcpClient(host, port);
        if (!client.IsConnected)
        {
            string reason = GetSensorTcpErrorReason(client);
            client.Dispose();
            SetSensorTcpStatus($"TCP connection failed: {reason}. Retrying {host}:{port} in {retrySeconds:0.0}s.");
            SleepSensorTcpReconnectInterval();
            return false;
        }

        sensorTcpClient = client;
        SetSensorTcpStatus($"TCP connected to {host}:{port}.");
        return true;
    }

    private float GetSensorTcpReconnectIntervalSeconds()
    {
        return Math.Max(0.1f, sensorTcpReconnectIntervalSeconds);
    }

    private void SleepSensorTcpReconnectInterval()
    {
        SleepSensorTcpThread((int)(GetSensorTcpReconnectIntervalSeconds() * 1000f));
    }

    private void DisconnectSensorTcpClient()
    {
        try { sensorTcpClient?.Dispose(); }
        catch { }
        sensorTcpClient = null;
        sensorTcpUnitPlaneMapSent = false;
    }

    private static string GetSensorTcpErrorReason(SimpleTcpClient client)
    {
        if (client == null || string.IsNullOrEmpty(client.LastError)) return "Unknown error";
        return client.LastError;
    }

    private static bool IsErrorSensorTcpResponse(byte[] response)
    {
        return response != null && response.Length == 5
            && response[0] == (byte)'e'
            && response[1] == (byte)'r'
            && response[2] == (byte)'r'
            && response[3] == (byte)'o'
            && response[4] == (byte)'r';
    }

    private void SleepSensorTcpThread(int milliseconds)
    {
        int remaining = Math.Max(0, milliseconds);
        while (sensorTcpRunning && remaining > 0)
        {
            int sleepNow = Math.Min(remaining, 100);
            Thread.Sleep(sleepNow);
            remaining -= sleepNow;
        }
    }

    private static List<Vector3> ParseServerMarkerWorldResponse(byte[] response)
    {
        if (IsBinaryMarkerWorldResponse(response))
        {
            return ParseBinaryMarkerWorldResponse(response);
        }

        return ParseJsonMarkerWorldResponse(response);
    }

    private static bool IsBinaryMarkerWorldResponse(byte[] response)
    {
        if (response == null || response.Length < MarkerWorldResponseMagic.Length) return false;
        for (int i = 0; i < MarkerWorldResponseMagic.Length; ++i)
        {
            if (response[i] != MarkerWorldResponseMagic[i]) return false;
        }

        return true;
    }

    private static List<Vector3> ParseBinaryMarkerWorldResponse(byte[] response)
    {
        const int headerBytes = 12;
        const int pointBytes = 12;
        if (response.Length < headerBytes) throw new ArgumentException("Marker response is too short.");

        int offset = MarkerWorldResponseMagic.Length;
        uint count = ReadUInt32LittleEndian(response, ref offset);
        int expectedLength = headerBytes + checked((int)count * pointBytes);
        if (response.Length != expectedLength)
        {
            throw new ArgumentException($"Marker response size mismatch: got {response.Length}, expected {expectedLength}.");
        }

        List<Vector3> markerWorldPositions = new List<Vector3>((int)count);
        for (int i = 0; i < count; ++i)
        {
            float x = ReadFloatLittleEndian(response, ref offset);
            float y = ReadFloatLittleEndian(response, ref offset);
            float z = ReadFloatLittleEndian(response, ref offset);
            markerWorldPositions.Add(new Vector3(x, y, -z));
        }

        return markerWorldPositions;
    }

    private static List<Vector3> ParseJsonMarkerWorldResponse(byte[] response)
    {
        string json = Encoding.UTF8.GetString(response);
        JArray markerArray = JArray.Parse(json);
        List<Vector3> markerWorldPositions = new List<Vector3>();

        foreach (var markerToken in markerArray)
        {
            if (!(markerToken is JArray marker) || marker.Count < 3) continue;

            markerWorldPositions.Add(new Vector3(
                marker[0].ToObject<float>(),
                marker[1].ToObject<float>(),
                -marker[2].ToObject<float>()));
        }

        return markerWorldPositions;
    }

    private static byte[] BuildRawSensorTcpPayload(RawSensorTcpFrame frame, ref byte[] payload)
    {
        int depthByteLength = SensorImagePixelCount * sizeof(ushort);
        int infraredByteLength = SensorImagePixelCount * sizeof(ushort);
        int matrixByteLength = SensorTcpDepthToWorldMatrixValues * sizeof(double);
        int unitPlaneCount = frame.IncludeUnitPlaneMap ? SensorImagePixelCount : 0;
        int unitPlaneByteLength = unitPlaneCount * SensorTcpUnitPlaneValuesPerPixel * sizeof(float);
        int payloadLength = SensorTcpRawStreamPrefix.Length + SensorTcpHeaderV4BaseBytes + matrixByteLength + unitPlaneByteLength + depthByteLength + infraredByteLength;
        if (payload == null || payload.Length != payloadLength)
        {
            payload = new byte[payloadLength];
        }

        int offset = 0;

        WriteBytes(payload, ref offset, SensorTcpRawStreamPrefix);
        WriteBytes(payload, ref offset, SensorTcpPayloadMagic);
        WriteInt32(payload, ref offset, SensorImageWidth);
        WriteInt32(payload, ref offset, SensorImageHeight);
        WriteUInt64(payload, ref offset, frame.Sequence);
        WriteDouble(payload, ref offset, frame.TimestampUnixSeconds);
        WriteInt32(payload, ref offset, SensorImagePixelCount);
        WriteInt32(payload, ref offset, SensorImagePixelCount);
        WriteInt32(payload, ref offset, SensorTcpDepthToWorldMatrixValues);
        WriteInt32(payload, ref offset, unitPlaneCount);

        if (BitConverter.IsLittleEndian)
        {
            Buffer.BlockCopy(frame.DepthToWorldMatrix, 0, payload, offset, matrixByteLength);
            offset += matrixByteLength;
        }
        else
        {
            foreach (double matrixValue in frame.DepthToWorldMatrix)
            {
                WriteDouble(payload, ref offset, matrixValue);
            }
        }

        if (frame.IncludeUnitPlaneMap)
        {
            if (BitConverter.IsLittleEndian)
            {
                Buffer.BlockCopy(frame.UnitPlaneMap, 0, payload, offset, unitPlaneByteLength);
                offset += unitPlaneByteLength;
            }
            else
            {
                foreach (float unitPlaneValue in frame.UnitPlaneMap)
                {
                    WriteFloat(payload, ref offset, unitPlaneValue);
                }
            }
        }

        Buffer.BlockCopy(frame.Depth, 0, payload, offset, depthByteLength);
        offset += depthByteLength;
        Buffer.BlockCopy(frame.Infrared, 0, payload, offset, infraredByteLength);

        return payload;
    }

    private static void WriteBytes(byte[] target, ref int offset, byte[] value)
    {
        Buffer.BlockCopy(value, 0, target, offset, value.Length);
        offset += value.Length;
    }

    private static void WriteInt32(byte[] target, ref int offset, int value)
    {
        unchecked
        {
            target[offset++] = (byte)value;
            target[offset++] = (byte)(value >> 8);
            target[offset++] = (byte)(value >> 16);
            target[offset++] = (byte)(value >> 24);
        }
    }

    private static void WriteUInt64(byte[] target, ref int offset, ulong value)
    {
        target[offset++] = (byte)value;
        target[offset++] = (byte)(value >> 8);
        target[offset++] = (byte)(value >> 16);
        target[offset++] = (byte)(value >> 24);
        target[offset++] = (byte)(value >> 32);
        target[offset++] = (byte)(value >> 40);
        target[offset++] = (byte)(value >> 48);
        target[offset++] = (byte)(value >> 56);
    }

    private static void WriteDouble(byte[] target, ref int offset, double value)
    {
        WriteUInt64(target, ref offset, (ulong)BitConverter.DoubleToInt64Bits(value));
    }

    private static void WriteFloat(byte[] target, ref int offset, float value)
    {
        byte[] valueBytes = BitConverter.GetBytes(value);
        if (BitConverter.IsLittleEndian)
        {
            target[offset++] = valueBytes[0];
            target[offset++] = valueBytes[1];
            target[offset++] = valueBytes[2];
            target[offset++] = valueBytes[3];
        }
        else
        {
            target[offset++] = valueBytes[3];
            target[offset++] = valueBytes[2];
            target[offset++] = valueBytes[1];
            target[offset++] = valueBytes[0];
        }
    }

    private static uint ReadUInt32LittleEndian(byte[] source, ref int offset)
    {
        uint value = (uint)(
            source[offset]
            | (source[offset + 1] << 8)
            | (source[offset + 2] << 16)
            | (source[offset + 3] << 24));
        offset += 4;
        return value;
    }

    private static float ReadFloatLittleEndian(byte[] source, ref int offset)
    {
        if (BitConverter.IsLittleEndian)
        {
            float value = BitConverter.ToSingle(source, offset);
            offset += 4;
            return value;
        }

        byte[] valueBytes = new byte[]
        {
            source[offset + 3],
            source[offset + 2],
            source[offset + 1],
            source[offset]
        };
        offset += 4;
        return BitConverter.ToSingle(valueBytes, 0);
    }

    private static double GetUnixTimeSeconds()
    {
        DateTime unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        return (DateTime.UtcNow - unixEpoch).TotalSeconds;
    }

    private void ApplyLatestMarkerWorldPositions()
    {
        List<Vector3> markerWorldPositions = null;
        lock (markerWorldPositionsLock)
        {
            if (!newMarkerWorldPositionsReceived) return;

            markerWorldPositions = new List<Vector3>(latestMarkerWorldPositions);
            newMarkerWorldPositionsReceived = false;
        }

        ReplaceMarkerWorldSpheres(markerWorldPositions);
    }

    private void ReplaceMarkerWorldSpheres(List<Vector3> markerWorldPositions)
    {
        foreach (var sphere in markerWorldSpheres)
        {
            if (sphere != null) Destroy(sphere);
        }

        markerWorldSpheres.Clear();

        foreach (var position in markerWorldPositions)
        {
            GameObject markerSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            markerSphere.name = "DetectedIRMarker";
            markerSphere.transform.position = position;
            markerSphere.transform.localScale = Vector3.one * MarkerSphereDiameterMetres;

            Collider collider = markerSphere.GetComponent<Collider>();
            if (collider != null) Destroy(collider);

            Renderer renderer = markerSphere.GetComponent<Renderer>();
            if (renderer != null) renderer.material = GetMarkerSphereMaterial();

            markerWorldSpheres.Add(markerSphere);
        }
    }

    private Material GetMarkerSphereMaterial()
    {
        if (markerSphereMaterial != null) return markerSphereMaterial;

        markerSphereMaterial = new Material(Shader.Find("Standard"));
        markerSphereMaterial.color = new Color(1f, 0f, 0f, 0.45f);
        markerSphereMaterial.SetFloat("_Mode", 3f);
        markerSphereMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        markerSphereMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        markerSphereMaterial.SetInt("_ZWrite", 0);
        markerSphereMaterial.DisableKeyword("_ALPHATEST_ON");
        markerSphereMaterial.EnableKeyword("_ALPHABLEND_ON");
        markerSphereMaterial.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        markerSphereMaterial.renderQueue = 3000;
        return markerSphereMaterial;
    }

    /// <summary>
    /// Function for receiving the latest raw 16-bit sensor images from the HL2 for visualisation.
    /// Display scaling happens in the grayscale shader, with a per-frame min/max range for the infrared image.
    /// </summary>
    void GrabLatestSensorImages()
    {
#   if ENABLE_WINMD_SUPPORT
        if (!SensorImagesDisplaying) return;

        if (!sensorImageStreamReady)
        {
            sensorImageStreamReady = researchMode.Depth8BitImageUpdated() || researchMode.AB8BitImageUpdated();
            if (!sensorImageStreamReady) return;
        }

        if (Time.unscaledTime < nextSensorImageUploadTime) return;

        ushort[] depthFrameTexture = researchMode.GetRawDepthImageBuffer();
        LoadRaw16TextureData(depthMediaTexture, depthFrameTexture);

        ushort[] abFrameTexture = researchMode.GetRawABImageBuffer();
        if (LoadRaw16TextureData(abMediaTexture, abFrameTexture))
        {
            UpdateInfraredDisplayRange(abFrameTexture);
        }

        double[] depthToWorldMatrix = researchMode.GetDepthToWorldMatrix();
        QueueRawSensorFrameForTcp(depthFrameTexture, abFrameTexture, depthToWorldMatrix);

        nextSensorImageUploadTime = Time.unscaledTime + SensorImageUploadIntervalSeconds;
#endif
    }

    /// <summary>
    /// Function for receiving tool pose matrices from the IR tracking class running on board the HL2
    /// </summary>
    void GrabLatestToolDictionary()
    {
#if ENABLE_WINMD_SUPPORT
        try
        {
            if (researchMode.ToolDictionaryUpdated()) // true each time a new set of tool poses are updated on the C++ side
            {
                // grabs an encoded double arrays
                double[] toolsTransform = researchMode.GetTrackedToolsPoseMatrices();
                // pass this information onto our tool manager
                if (toolsTransform != null) ToolManagerScript.EnqueueTrackingData(toolsTransform);
            }
        }
        catch (Exception ex)
        {
            ConsoleDebugTextMesh.text = ex.StackTrace;
        }
#endif
    }

    void LateUpdate()
    {
        // main loop of this class
#if ENABLE_WINMD_SUPPORT
        GrabLatestSensorImages();
        GrabLatestToolDictionary();
#endif
        ApplyLatestMarkerWorldPositions();
        ApplySensorTcpStatusToScreen();
    }

#if WINDOWS_UWP
    /// <summary>
    ///  Function tries to grab coordinate frame details from Unity's side, to pass into the HL2-DINO DLL.
    ///  This will allow the C++ DLL to try and track tools in the same coordinate frame being used in Unity.
    /// </summary>
    /// <returns></returns>
    bool SetupLocator()
    {
        // original source, instructive:
        // https://github.com/microsoft/MixedRealityToolkit-Unity/issues/10082#issuecomment-905865993

        SpatialLocatability headset_init = SpatialLocatability.Unavailable;
        int i;
        for (i = 0; i < 20; i++) // 20 attempts before cutting out? (this isn't threaded, so it could be fatal...)
        {
            SpatialLocator locator = SpatialLocator.GetDefault();
            headset_init = locator.Locatability;
            if (headset_init == SpatialLocatability.PositionalTrackingActive) { break; }
        }

        if (headset_init != SpatialLocatability.PositionalTrackingActive) { return false; }

        var unityCoordinateSystem = Microsoft.MixedReality.OpenXR.PerceptionInterop.GetSceneCoordinateSystem(UnityEngine.Pose.identity) 
            as SpatialCoordinateSystem;

        if (unityCoordinateSystem == null) return false;

        if (researchMode != null) { researchMode.SetReferenceCoordinateSystem(unityCoordinateSystem); return true; }
        else return false;
    }

    /// <summary>
    /// Helper function which will save the profiler string to the app's local folder
    /// Has to be retrieved from the Device Portal
    /// </summary>
    /// <param name="content"></param>
    /// <param name="filename">Filepath (with proper extension)</param>
    private async Task SaveProfilerString(string content, string filename)
    {
        try
        {
            string localFolder = ApplicationData.Current.LocalFolder.Path;
            string profiler_path = localFolder + "/" + filename;

            using (StreamWriter file = new StreamWriter(profiler_path))
            {
                await file.WriteAsync(content);
            }
        }
        catch (Exception ex)
        {
            ConsoleDebugTextMesh.text = $"Error: {ex.Message}";
        }
    }

#endif


    private void OnApplicationFocus(bool focus)
    {
        // if app is shutdown, try to stop the sensor loop running
        if (!focus) StopDepthSensor();
    }

    private void OnDestroy()
    {
        StopSensorTcpThread();
    }

    /// <summary>
    /// Function to flag a shutdown for the depth sensor loop on the DLL side
    /// </summary>
    public void StopDepthSensor()
    {
#if ENABLE_WINMD_SUPPORT
        researchMode.StopSensorLoop();
#endif
    }

    /// <summary>
    /// Public toggle function to tell the DLL side if we should be stashing sensor images for display
    /// or not
    /// </summary>
    public void SwitchSensorDisplayOnOff()
    {
        SensorImagesDisplaying = !SensorImagesDisplaying;
#if ENABLE_WINMD_SUPPORT
        researchMode.ToggleDisplaySensorImages(SensorImagesDisplaying);
#endif
    }

    /// <summary>
    /// Profiler data as recorded by the Shiny library on the C++ DLL.
    /// 
    /// Proper thread safety has not been robustly tested, would suggest avoiding
    /// numerous calls from multiple sources, especially as you're dumping to a file.
    /// </summary>

#if ENABLE_WINMD_SUPPORT
    async
#endif
    public void FetchProfilerString()
    {
#if ENABLE_WINMD_SUPPORT
        string profileString = researchMode.GetProfilerString();
        ConsoleDebugTextMesh.text = profileString;
        await SaveProfilerString(profileString, $"DINO-AR_Profile_Unity{Application.unityVersion}.txt");
#endif
    }

    private class RawSensorTcpFrame
    {
        public ushort[] Depth;
        public ushort[] Infrared;
        public double[] DepthToWorldMatrix;
        public float[] UnitPlaneMap;
        public bool IncludeUnitPlaneMap;
        public ulong Sequence;
        public double TimestampUnixSeconds;
    }

}
