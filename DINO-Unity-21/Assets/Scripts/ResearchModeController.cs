using UnityEngine;
using System;
using System.Runtime.InteropServices;
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
    private const int SensorTcpHeaderBytes = 40;
    private const float Raw16MaxValue = 65535f;
    private const float DepthDisplayBlackMm = 1f;
    private const float DepthDisplayWhiteMm = 4090f;
    private const float SensorImageUploadIntervalSeconds = 1f / 30f;
    private static readonly byte[] SensorTcpRawStreamPrefix = Encoding.ASCII.GetBytes("raw_stream:");
    private static readonly byte[] SensorTcpIrMarkersRequest = Encoding.ASCII.GetBytes("ir_markers:");
    private static readonly byte[] SensorTcpReal3DCoordPrefix = Encoding.ASCII.GetBytes("real_3d_coord:");
    private static readonly byte[] SensorTcpPayloadMagic = Encoding.ASCII.GetBytes("DINOIMG1");
    private const float MarkerPollingIntervalSeconds = 1f / 15f;
    private const float MarkerSphereDiameterMetres = 0.01f;
    private const float MarkerSphereRadiusMetres = MarkerSphereDiameterMetres * 0.5f;

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
    private RawSensorTcpFrame latestSensorTcpFrame = null;
    private readonly object sensorTcpFrameLock = new object();
    private readonly object sensorTcpStatusLock = new object();
    private string sensorTcpStatusMessage = string.Empty;
    private bool sensorTcpStatusDirty = false;
    private Thread markerTcpThread = null;
    private SimpleTcpClient markerTcpClient = null;
    private volatile bool markerTcpRunning = false;
    private readonly object markerPixelsLock = new object();
    private List<Vector2> latestMarkerPixels = new List<Vector2>();
    private bool newMarkerPixelsReceived = false;
    private readonly object markerWorldCoordinatesUploadLock = new object();
    private List<Vector3> pendingMarkerWorldCoordinatesUpload = new List<Vector3>();
    private bool markerWorldCoordinatesUploadPending = false;
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
        StartMarkerTcpThread();

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
    }

    private void StartMarkerTcpThread()
    {
        if (markerTcpThread != null) return;

        markerTcpRunning = true;
        markerTcpThread = new Thread(MarkerTcpBackgroundLoop)
        {
            IsBackground = true,
            Name = "DINO IR Marker TCP Receiver"
        };
        markerTcpThread.Start();
    }

    private void StopMarkerTcpThread()
    {
        markerTcpRunning = false;

        try { markerTcpClient?.Close(); }
        catch { }

        if (markerTcpThread != null && markerTcpThread.IsAlive)
        {
            markerTcpThread.Join(1000);
        }

        markerTcpThread = null;
        markerTcpClient = null;
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

    private void QueueRawSensorFrameForTcp(ushort[] depthFrame, ushort[] infraredFrame)
    {
        if (!streamRawSensorImagesOverTcp || !sensorTcpRunning) return;
        if (depthFrame == null || infraredFrame == null) return;
        if (depthFrame.Length < SensorImagePixelCount || infraredFrame.Length < SensorImagePixelCount) return;
        if (Time.unscaledTime < nextSensorTcpQueueTime) return;

        float frameInterval = Mathf.Clamp(sensorTcpFrameIntervalSeconds, 0f, SensorImageUploadIntervalSeconds);
        nextSensorTcpQueueTime = Time.unscaledTime + frameInterval;

        RawSensorTcpFrame frame = new RawSensorTcpFrame
        {
            Depth = depthFrame,
            Infrared = infraredFrame,
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
            }

            if (!sensorTcpRunning) return null;

            RawSensorTcpFrame frame = latestSensorTcpFrame;
            latestSensorTcpFrame = null;
            return frame;
        }
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
                sensorTcpClient.Dispose();
                sensorTcpClient = null;
                SetSensorTcpStatus($"TCP connection failed: {reason}. Reconnecting to {sensorTcpHost}:{sensorTcpPort}.");
                SleepSensorTcpThread(250);
                continue;
            }

            if (!IsOkSensorTcpResponse(response))
            {
                string responseText = Encoding.ASCII.GetString(response);
                sensorTcpClient.Dispose();
                sensorTcpClient = null;
                SetSensorTcpStatus($"TCP connection failed: server returned unexpected response '{responseText}'. Reconnecting to {sensorTcpHost}:{sensorTcpPort}.");
                SleepSensorTcpThread(250);
            }
        }

        sensorTcpClient?.Dispose();
        sensorTcpClient = null;
        sensorTcpPayloadBuffer = null;
    }

    private void MarkerTcpBackgroundLoop()
    {
        while (markerTcpRunning)
        {
            if (!EnsureMarkerTcpConnected())
            {
                continue;
            }

            byte[] response = markerTcpClient.Request(SensorTcpIrMarkersRequest);
            if (response == null)
            {
                string reason = GetSensorTcpErrorReason(markerTcpClient);
                markerTcpClient.Dispose();
                markerTcpClient = null;
                SetSensorTcpStatus($"IR marker TCP failed: {reason}. Reconnecting to {sensorTcpHost}:{sensorTcpPort}.");
                SleepMarkerTcpThread(250);
                continue;
            }

            try
            {
                List<Vector2> markerPixels = ParseMarkerPixelResponse(response);
                lock (markerPixelsLock)
                {
                    latestMarkerPixels = markerPixels;
                    newMarkerPixelsReceived = true;
                }
            }
            catch (Exception ex)
            {
                SetSensorTcpStatus($"IR marker response parse failed: {ex.Message}");
            }

            try
            {
                UploadPendingMarkerWorldCoordinates();
            }
            catch (Exception ex)
            {
                SetSensorTcpStatus(ex.Message);
            }

            SleepMarkerTcpThread((int)(MarkerPollingIntervalSeconds * 1000f));
        }

        markerTcpClient?.Dispose();
        markerTcpClient = null;
    }

    private bool EnsureSensorTcpConnected()
    {
        if (sensorTcpClient != null && sensorTcpClient.IsConnected) return true;

        sensorTcpClient?.Dispose();
        sensorTcpClient = null;

        string host = sensorTcpHost;
        int port = sensorTcpPort;
        float retrySeconds = Math.Max(0.1f, sensorTcpReconnectIntervalSeconds);

        SetSensorTcpStatus($"TCP connecting to {host}:{port}...");

        SimpleTcpClient client = new SimpleTcpClient(host, port);
        if (!client.IsConnected)
        {
            string reason = GetSensorTcpErrorReason(client);
            client.Dispose();
            SetSensorTcpStatus($"TCP connection failed: {reason}. Retrying {host}:{port} in {retrySeconds:0.0}s.");
            SleepSensorTcpThread((int)(retrySeconds * 1000f));
            return false;
        }

        sensorTcpClient = client;
        SetSensorTcpStatus($"TCP connected to {host}:{port}.");
        return true;
    }

    private bool EnsureMarkerTcpConnected()
    {
        if (markerTcpClient != null && markerTcpClient.IsConnected) return true;

        markerTcpClient?.Dispose();
        markerTcpClient = null;

        string host = sensorTcpHost;
        int port = sensorTcpPort;
        float retrySeconds = Math.Max(0.1f, sensorTcpReconnectIntervalSeconds);

        SimpleTcpClient client = new SimpleTcpClient(host, port);
        if (!client.IsConnected)
        {
            string reason = GetSensorTcpErrorReason(client);
            client.Dispose();
            SetSensorTcpStatus($"IR marker TCP failed: {reason}. Retrying {host}:{port} in {retrySeconds:0.0}s.");
            SleepMarkerTcpThread((int)(retrySeconds * 1000f));
            return false;
        }

        markerTcpClient = client;
        return true;
    }

    private static string GetSensorTcpErrorReason(SimpleTcpClient client)
    {
        if (client == null || string.IsNullOrEmpty(client.LastError)) return "Unknown error";
        return client.LastError;
    }

    private static bool IsOkSensorTcpResponse(byte[] response)
    {
        return response != null && response.Length == 2 && response[0] == (byte)'o' && response[1] == (byte)'k';
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

    private void SleepMarkerTcpThread(int milliseconds)
    {
        int remaining = Math.Max(0, milliseconds);
        while (markerTcpRunning && remaining > 0)
        {
            int sleepNow = Math.Min(remaining, 100);
            Thread.Sleep(sleepNow);
            remaining -= sleepNow;
        }
    }

    private static List<Vector2> ParseMarkerPixelResponse(byte[] response)
    {
        string json = Encoding.UTF8.GetString(response);
        JArray markerArray = JArray.Parse(json);
        List<Vector2> markerPixels = new List<Vector2>();

        foreach (var markerToken in markerArray)
        {
            if (!(markerToken is JArray marker) || marker.Count < 2) continue;

            markerPixels.Add(new Vector2(
                marker[0].ToObject<float>(),
                marker[1].ToObject<float>()));
        }

        return markerPixels;
    }

    private void QueueMarkerWorldCoordinatesUpload(List<Vector3> markerWorldPositions)
    {
        lock (markerWorldCoordinatesUploadLock)
        {
            pendingMarkerWorldCoordinatesUpload = new List<Vector3>(markerWorldPositions);
            markerWorldCoordinatesUploadPending = true;
        }
    }

    private void UploadPendingMarkerWorldCoordinates()
    {
        List<Vector3> markerWorldPositions;
        lock (markerWorldCoordinatesUploadLock)
        {
            if (!markerWorldCoordinatesUploadPending) return;

            markerWorldPositions = new List<Vector3>(pendingMarkerWorldCoordinatesUpload);
            markerWorldCoordinatesUploadPending = false;
        }

        byte[] payload = BuildMarkerWorldCoordinatesPayload(markerWorldPositions);
        byte[] response = markerTcpClient.Request(payload);
        if (!IsOkSensorTcpResponse(response))
        {
            lock (markerWorldCoordinatesUploadLock)
            {
                pendingMarkerWorldCoordinatesUpload = markerWorldPositions;
                markerWorldCoordinatesUploadPending = true;
            }

            string reason = response == null ? GetSensorTcpErrorReason(markerTcpClient) : Encoding.ASCII.GetString(response);
            throw new InvalidOperationException($"real_3d_coord upload failed: {reason}");
        }
    }

    private static byte[] BuildMarkerWorldCoordinatesPayload(List<Vector3> markerWorldPositions)
    {
        JArray coordinatesJson = new JArray();
        foreach (var position in markerWorldPositions)
        {
            coordinatesJson.Add(new JArray(position.x, position.y, position.z));
        }

        byte[] jsonPayload = Encoding.UTF8.GetBytes(coordinatesJson.ToString(Newtonsoft.Json.Formatting.None));
        byte[] payload = new byte[SensorTcpReal3DCoordPrefix.Length + jsonPayload.Length];
        Buffer.BlockCopy(SensorTcpReal3DCoordPrefix, 0, payload, 0, SensorTcpReal3DCoordPrefix.Length);
        Buffer.BlockCopy(jsonPayload, 0, payload, SensorTcpReal3DCoordPrefix.Length, jsonPayload.Length);
        return payload;
    }

    private static byte[] BuildRawSensorTcpPayload(RawSensorTcpFrame frame, ref byte[] payload)
    {
        int depthByteLength = SensorImagePixelCount * sizeof(ushort);
        int infraredByteLength = SensorImagePixelCount * sizeof(ushort);
        int payloadLength = SensorTcpRawStreamPrefix.Length + SensorTcpHeaderBytes + depthByteLength + infraredByteLength;
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
        WriteLittleEndianBytes(target, ref offset, BitConverter.GetBytes(value));
    }

    private static void WriteUInt64(byte[] target, ref int offset, ulong value)
    {
        WriteLittleEndianBytes(target, ref offset, BitConverter.GetBytes(value));
    }

    private static void WriteDouble(byte[] target, ref int offset, double value)
    {
        WriteLittleEndianBytes(target, ref offset, BitConverter.GetBytes(value));
    }

    private static void WriteLittleEndianBytes(byte[] target, ref int offset, byte[] value)
    {
        if (!BitConverter.IsLittleEndian) Array.Reverse(value);
        Buffer.BlockCopy(value, 0, target, offset, value.Length);
        offset += value.Length;
    }

    private static double GetUnixTimeSeconds()
    {
        DateTime unixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        return (DateTime.UtcNow - unixEpoch).TotalSeconds;
    }

    private void ApplyLatestMarkerPixels()
    {
        List<Vector2> markerPixels = null;
        lock (markerPixelsLock)
        {
            if (!newMarkerPixelsReceived) return;

            markerPixels = new List<Vector2>(latestMarkerPixels);
            newMarkerPixelsReceived = false;
        }

        List<Vector3> markerWorldPositions = ResolveMarkerWorldPositions(markerPixels);
        QueueMarkerWorldCoordinatesUpload(markerWorldPositions);
        ReplaceMarkerWorldSpheres(markerWorldPositions);
    }

    private List<Vector3> ResolveMarkerWorldPositions(List<Vector2> markerPixels)
    {
        List<Vector3> markerWorldPositions = new List<Vector3>();
#if ENABLE_WINMD_SUPPORT
        if (researchMode == null || markerPixels == null) return markerWorldPositions;

        ushort[] depthFrame = researchMode.GetRawDepthImageBuffer();
        if (depthFrame == null || depthFrame.Length < SensorImagePixelCount) return markerWorldPositions;

        foreach (var markerPixel in markerPixels)
        {
            int pixelX = Mathf.Clamp(Mathf.RoundToInt(markerPixel.x), 0, SensorImageWidth - 1);
            int pixelY = Mathf.Clamp(Mathf.RoundToInt(markerPixel.y), 0, SensorImageHeight - 1);
            ushort depthValue = depthFrame[pixelY * SensorImageWidth + pixelX];
            if (depthValue == 0) continue;

            double[] worldCoordinate = researchMode.GetDepthPixelWorldCoordinate(markerPixel.x, markerPixel.y, depthValue);
            if (worldCoordinate == null || worldCoordinate.Length < 3) continue;

            Vector3 markerSurfacePosition = new Vector3(
                (float)worldCoordinate[0],
                (float)worldCoordinate[1],
                -(float)worldCoordinate[2]);

            markerWorldPositions.Add(OffsetMarkerSurfacePointToCenter(markerSurfacePosition));
        }
#endif
        return markerWorldPositions;
    }

    private static Vector3 OffsetMarkerSurfacePointToCenter(Vector3 markerSurfacePosition)
    {
        Camera mainCamera = Camera.main;
        if (mainCamera == null) return markerSurfacePosition;

        Vector3 outwardDirection = markerSurfacePosition - mainCamera.transform.position;
        if (outwardDirection.sqrMagnitude <= Mathf.Epsilon) return markerSurfacePosition;

        return markerSurfacePosition + outwardDirection.normalized * MarkerSphereRadiusMetres;
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

        QueueRawSensorFrameForTcp(depthFrameTexture, abFrameTexture);

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
        ApplyLatestMarkerPixels();
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
        StopMarkerTcpThread();
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
        public ulong Sequence;
        public double TimestampUnixSeconds;
    }

}
