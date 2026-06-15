using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

public static class AimToolAssetImporter
{
    private const string MenuPath = "DINO Unity/Import AimTools";
    private const string AimToolsFolderName = "AimTools";
    private const string ResourcesAimToolsPath = "Assets/Resources/AimTools";
    private static readonly CultureInfo InvariantCulture = CultureInfo.InvariantCulture;

    [MenuItem(MenuPath)]
    public static void ImportAimTools()
    {
        string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        string repoRoot = Path.GetFullPath(Path.Combine(projectRoot, ".."));
        string sourceFolder = Path.Combine(repoRoot, AimToolsFolderName);

        if (!Directory.Exists(sourceFolder))
        {
            Debug.LogError(string.Format("AimTools source folder not found: {0}", sourceFolder));
            return;
        }

        string targetFolderAbsolute = Path.Combine(Application.dataPath, "Resources", AimToolsFolderName);
        Directory.CreateDirectory(targetFolderAbsolute);

        int importedCount = 0;
        string[] aimtoolFiles = Directory.GetFiles(sourceFolder, "*.aimtool", SearchOption.TopDirectoryOnly);
        Array.Sort(aimtoolFiles, StringComparer.OrdinalIgnoreCase);

        foreach (string aimtoolPath in aimtoolFiles)
        {
            string modelName = Path.GetFileNameWithoutExtension(aimtoolPath);
            string stlPath = Path.Combine(sourceFolder, modelName + ".stl");
            if (!File.Exists(stlPath))
            {
                Debug.LogWarning(string.Format("Skipping {0}: matching STL not found.", modelName));
                continue;
            }

            List<Vector3> markers;
            if (!TryReadAimToolMarkers(aimtoolPath, out markers))
            {
                Debug.LogWarning(string.Format("Skipping {0}: failed to parse aimtool marker coordinates.", modelName));
                continue;
            }

            string objPath = Path.Combine(targetFolderAbsolute, modelName + ".obj");
            string markerJsonPath = Path.Combine(targetFolderAbsolute, modelName + ".markers.json");

            try
            {
                ConvertStlToObj(stlPath, objPath, modelName);
                WriteMarkerJson(markerJsonPath, modelName, aimtoolPath, stlPath, markers);
                importedCount++;
            }
            catch (Exception ex)
            {
                Debug.LogError(string.Format("Failed to import AimTool {0}: {1}", modelName, ex.Message));
            }
        }

        AssetDatabase.Refresh();
        ConfigureImportedModels();
        Debug.Log(string.Format("Imported {0} AimTool model(s) into {1}.", importedCount, ResourcesAimToolsPath));
    }

    private static bool TryReadAimToolMarkers(string path, out List<Vector3> markers)
    {
        markers = new List<Vector3>();
        string[] lines = File.ReadAllLines(path);
        if (lines.Length < 3) return false;

        int markerCount;
        if (!int.TryParse(lines[2].Trim(), NumberStyles.Integer, InvariantCulture, out markerCount)) return false;
        if (markerCount < 3 || lines.Length < 3 + markerCount) return false;

        for (int i = 0; i < markerCount; ++i)
        {
            string line = lines[3 + i];
            string[] parts = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 3) return false;

            float x;
            float y;
            float z;
            if (!float.TryParse(parts[0], NumberStyles.Float, InvariantCulture, out x)) return false;
            if (!float.TryParse(parts[1], NumberStyles.Float, InvariantCulture, out y)) return false;
            if (!float.TryParse(parts[2], NumberStyles.Float, InvariantCulture, out z)) return false;

            markers.Add(ConvertRightHandedPointToUnityMetres(x, y, z));
        }

        return true;
    }

    private static Vector3 ConvertRightHandedPointToUnityMetres(float x, float y, float z)
    {
        const float millimetresToMetres = 0.001f;
        return new Vector3(x * millimetresToMetres, y * millimetresToMetres, -z * millimetresToMetres);
    }

    private static Vector3 ConvertRightHandedStlVertexToObjMetres(float x, float y, float z)
    {
        const float millimetresToMetres = 0.001f;
        // Unity's OBJ importer applies its own handedness compensation on import.
        // Pre-flip X here so the imported mesh lands in the same Unity left-handed
        // coordinates as the marker JSON above.
        return new Vector3(-x * millimetresToMetres, y * millimetresToMetres, -z * millimetresToMetres);
    }

    private static void ConvertStlToObj(string stlPath, string objPath, string modelName)
    {
        using (FileStream stream = File.OpenRead(stlPath))
        using (StreamWriter writer = new StreamWriter(objPath, false, Encoding.ASCII))
        {
            writer.WriteLine("# Generated from {0}", Path.GetFileName(stlPath));
            writer.WriteLine("o {0}", SanitizeObjName(modelName));

            if (LooksLikeBinaryStl(stream))
            {
                WriteBinaryStlAsObj(stream, writer);
            }
            else
            {
                WriteAsciiStlAsObj(stream, writer);
            }
        }
    }

    private static bool LooksLikeBinaryStl(FileStream stream)
    {
        if (stream.Length < 84)
        {
            stream.Position = 0;
            return false;
        }

        stream.Position = 80;
        byte[] countBytes = new byte[4];
        ReadExactly(stream, countBytes, 0, 4);
        uint triangleCount = BitConverter.ToUInt32(countBytes, 0);
        bool lengthMatchesBinary = 84L + triangleCount * 50L == stream.Length;
        stream.Position = 0;
        return lengthMatchesBinary;
    }

    private static void WriteBinaryStlAsObj(FileStream stream, StreamWriter writer)
    {
        stream.Position = 80;
        byte[] countBytes = new byte[4];
        ReadExactly(stream, countBytes, 0, 4);
        uint triangleCount = BitConverter.ToUInt32(countBytes, 0);
        byte[] triangleBytes = new byte[50];
        int vertexIndex = 1;

        for (uint triangle = 0; triangle < triangleCount; ++triangle)
        {
            ReadExactly(stream, triangleBytes, 0, triangleBytes.Length);
            Vector3 a = ReadStlVertex(triangleBytes, 12);
            Vector3 b = ReadStlVertex(triangleBytes, 24);
            Vector3 c = ReadStlVertex(triangleBytes, 36);

            WriteObjVertex(writer, a);
            WriteObjVertex(writer, b);
            WriteObjVertex(writer, c);
            writer.WriteLine("f {0} {1} {2}", vertexIndex, vertexIndex + 1, vertexIndex + 2);
            vertexIndex += 3;
        }
    }

    private static void WriteAsciiStlAsObj(FileStream stream, StreamWriter writer)
    {
        stream.Position = 0;
        int vertexIndex = 1;
        List<Vector3> triangle = new List<Vector3>(3);

        using (StreamReader reader = new StreamReader(stream, Encoding.ASCII, true, 4096, leaveOpen: true))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                string trimmed = line.TrimStart();
                if (!trimmed.StartsWith("vertex ", StringComparison.OrdinalIgnoreCase)) continue;

                string[] parts = trimmed.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 4) continue;

                float x;
                float y;
                float z;
                if (!float.TryParse(parts[1], NumberStyles.Float, InvariantCulture, out x)) continue;
                if (!float.TryParse(parts[2], NumberStyles.Float, InvariantCulture, out y)) continue;
                if (!float.TryParse(parts[3], NumberStyles.Float, InvariantCulture, out z)) continue;

                triangle.Add(ConvertRightHandedStlVertexToObjMetres(x, y, z));
                if (triangle.Count == 3)
                {
                    WriteObjVertex(writer, triangle[0]);
                    WriteObjVertex(writer, triangle[1]);
                    WriteObjVertex(writer, triangle[2]);
                    writer.WriteLine("f {0} {1} {2}", vertexIndex, vertexIndex + 1, vertexIndex + 2);
                    vertexIndex += 3;
                    triangle.Clear();
                }
            }
        }
    }

    private static Vector3 ReadStlVertex(byte[] bytes, int offset)
    {
        float x = BitConverter.ToSingle(bytes, offset);
        float y = BitConverter.ToSingle(bytes, offset + 4);
        float z = BitConverter.ToSingle(bytes, offset + 8);
        return ConvertRightHandedStlVertexToObjMetres(x, y, z);
    }

    private static void WriteObjVertex(StreamWriter writer, Vector3 vertex)
    {
        writer.WriteLine(
            "v {0} {1} {2}",
            vertex.x.ToString("G9", InvariantCulture),
            vertex.y.ToString("G9", InvariantCulture),
            vertex.z.ToString("G9", InvariantCulture));
    }

    private static void WriteMarkerJson(string path, string modelName, string aimtoolPath, string stlPath, List<Vector3> markers)
    {
        using (StreamWriter writer = new StreamWriter(path, false, Encoding.UTF8))
        {
            writer.WriteLine("{");
            writer.WriteLine("  \"name\": \"{0}\",", JsonEscape(modelName));
            writer.WriteLine("  \"sourceAimTool\": \"{0}\",", JsonEscape(Path.GetFileName(aimtoolPath)));
            writer.WriteLine("  \"sourceStl\": \"{0}\",", JsonEscape(Path.GetFileName(stlPath)));
            writer.WriteLine("  \"modelResourcePath\": \"{0}/{1}\",", AimToolsFolderName, JsonEscape(modelName));
            writer.WriteLine("  \"markers\": [");
            for (int i = 0; i < markers.Count; ++i)
            {
                Vector3 marker = markers[i];
                string suffix = i + 1 == markers.Count ? string.Empty : ",";
                writer.WriteLine(
                    "    {{ \"x\": {0}, \"y\": {1}, \"z\": {2} }}{3}",
                    marker.x.ToString("G9", InvariantCulture),
                    marker.y.ToString("G9", InvariantCulture),
                    marker.z.ToString("G9", InvariantCulture),
                    suffix);
            }

            writer.WriteLine("  ]");
            writer.WriteLine("}");
        }
    }

    private static void ConfigureImportedModels()
    {
        string[] objGuids = AssetDatabase.FindAssets("t:Model", new[] { ResourcesAimToolsPath });
        for (int i = 0; i < objGuids.Length; ++i)
        {
            string assetPath = AssetDatabase.GUIDToAssetPath(objGuids[i]);
            if (!assetPath.EndsWith(".obj", StringComparison.OrdinalIgnoreCase)) continue;

            ModelImporter modelImporter = AssetImporter.GetAtPath(assetPath) as ModelImporter;
            if (modelImporter == null) continue;

            modelImporter.globalScale = 1f;
            modelImporter.useFileScale = true;
            modelImporter.importAnimation = false;
            modelImporter.addCollider = false;
            modelImporter.materialImportMode = ModelImporterMaterialImportMode.None;
            modelImporter.SaveAndReimport();
        }
    }

    private static void ReadExactly(Stream stream, byte[] buffer, int offset, int count)
    {
        int readTotal = 0;
        while (readTotal < count)
        {
            int read = stream.Read(buffer, offset + readTotal, count - readTotal);
            if (read <= 0) throw new EndOfStreamException("Unexpected end of STL file.");
            readTotal += read;
        }
    }

    private static string SanitizeObjName(string value)
    {
        if (string.IsNullOrEmpty(value)) return "AimToolModel";

        char[] chars = value.ToCharArray();
        for (int i = 0; i < chars.Length; ++i)
        {
            if (!char.IsLetterOrDigit(chars[i]) && chars[i] != '_' && chars[i] != '-')
            {
                chars[i] = '_';
            }
        }

        return new string(chars);
    }

    private static string JsonEscape(string value)
    {
        return string.IsNullOrEmpty(value)
            ? string.Empty
            : value.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }
}
