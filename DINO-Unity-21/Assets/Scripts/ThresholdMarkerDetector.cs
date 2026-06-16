using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Unity-side port of ir_yolo_tracker.ThresholdMarkerDetector.
/// Runs directly on the 16-bit infrared frame and returns marker centers in image pixels.
/// </summary>
public sealed class ThresholdMarkerDetector
{
    public const float DefaultConfidenceThreshold = 0.25f;
    public const float DefaultThresholdPercentile = 99.7f;
    public const int DefaultMinimumThreshold = 0;
    public const int DefaultMinArea = 6;
    public const int DefaultMaxArea = 3000;
    public const float DefaultMinCircularity = 0.15f;
    public const float DefaultMinAspectRatio = 0.25f;
    public const float DefaultMaxAspectRatio = 4.0f;
    public const int DefaultMinWidth = 1;
    public const int DefaultMinHeight = 1;
    public const int DefaultMaxWidth = 80;
    public const int DefaultMaxHeight = 80;
    public const int DefaultMorphologyKernelSize = 0;
    public const int DefaultMorphologyOpenIterations = 0;
    public const int DefaultMorphologyCloseIterations = 0;
    public const float DefaultMaxForegroundFraction = 0.05f;
    public const float DefaultConfidenceSaturationRatio = 3.0f;
    public const int DefaultMaxDetections = 16;

    private const int HistogramBins = ushort.MaxValue + 1;

    private readonly int width;
    private readonly int height;
    private readonly int pixelCount;
    private readonly float confidenceThreshold;
    private readonly float thresholdPercentile;
    private readonly int minimumThreshold;
    private readonly int minArea;
    private readonly int maxArea;
    private readonly float minCircularity;
    private readonly float minAspectRatio;
    private readonly float maxAspectRatio;
    private readonly int minWidth;
    private readonly int minHeight;
    private readonly int maxWidth;
    private readonly int maxHeight;
    private readonly int morphologyKernelSize;
    private readonly int morphologyOpenIterations;
    private readonly int morphologyCloseIterations;
    private readonly float maxForegroundFraction;
    private readonly float confidenceSaturationRatio;
    private readonly int maxDetections;

    private readonly int[] histogram = new int[HistogramBins];
    private readonly byte[] binary;
    private readonly int[] labels;
    private readonly int[] floodQueue;
    private readonly byte[] morphologyBufferA;
    private readonly byte[] morphologyBufferB;
    private readonly List<Detection> detections = new List<Detection>(DefaultMaxDetections);

    public ThresholdMarkerDetector(
        int width,
        int height,
        float confidenceThreshold = DefaultConfidenceThreshold,
        float thresholdPercentile = DefaultThresholdPercentile,
        int minimumThreshold = DefaultMinimumThreshold,
        int minArea = DefaultMinArea,
        int maxArea = DefaultMaxArea,
        float minCircularity = DefaultMinCircularity,
        float minAspectRatio = DefaultMinAspectRatio,
        float maxAspectRatio = DefaultMaxAspectRatio,
        int minWidth = DefaultMinWidth,
        int minHeight = DefaultMinHeight,
        int maxWidth = DefaultMaxWidth,
        int maxHeight = DefaultMaxHeight,
        int morphologyKernelSize = DefaultMorphologyKernelSize,
        int morphologyOpenIterations = DefaultMorphologyOpenIterations,
        int morphologyCloseIterations = DefaultMorphologyCloseIterations,
        float maxForegroundFraction = DefaultMaxForegroundFraction,
        float confidenceSaturationRatio = DefaultConfidenceSaturationRatio,
        int maxDetections = DefaultMaxDetections)
    {
        if (width <= 0 || height <= 0) throw new ArgumentException("Image size must be positive.");
        if (confidenceThreshold < 0f || confidenceThreshold > 1f) throw new ArgumentException("confidenceThreshold must be between 0 and 1.");
        if (thresholdPercentile < 0f || thresholdPercentile > 100f) throw new ArgumentException("thresholdPercentile must be between 0 and 100.");
        if (minimumThreshold < 0 || minimumThreshold > ushort.MaxValue) throw new ArgumentException("minimumThreshold must fit in uint16.");
        if (minArea <= 0 || maxArea < minArea) throw new ArgumentException("Area bounds must satisfy 0 < minArea <= maxArea.");
        if (minCircularity < 0f) throw new ArgumentException("minCircularity must be non-negative.");
        if (minAspectRatio <= 0f || maxAspectRatio < minAspectRatio) throw new ArgumentException("Aspect ratio bounds must satisfy 0 < min <= max.");
        if (minWidth <= 0 || minHeight <= 0) throw new ArgumentException("Minimum dimensions must be positive.");
        if (maxWidth < minWidth || maxHeight < minHeight) throw new ArgumentException("Maximum dimensions must be greater than minimum dimensions.");
        if (morphologyKernelSize < 0) throw new ArgumentException("morphologyKernelSize must be non-negative.");
        if (morphologyOpenIterations < 0 || morphologyCloseIterations < 0) throw new ArgumentException("Morphology iterations must be non-negative.");
        if (maxForegroundFraction <= 0f || maxForegroundFraction > 1f) throw new ArgumentException("maxForegroundFraction must be in the range (0, 1].");
        if (confidenceSaturationRatio <= 1f) throw new ArgumentException("confidenceSaturationRatio must be greater than 1.");
        if (maxDetections < 0) throw new ArgumentException("maxDetections must be non-negative.");

        this.width = width;
        this.height = height;
        pixelCount = checked(width * height);
        this.confidenceThreshold = confidenceThreshold;
        this.thresholdPercentile = thresholdPercentile;
        this.minimumThreshold = minimumThreshold;
        this.minArea = minArea;
        this.maxArea = maxArea;
        this.minCircularity = minCircularity;
        this.minAspectRatio = minAspectRatio;
        this.maxAspectRatio = maxAspectRatio;
        this.minWidth = minWidth;
        this.minHeight = minHeight;
        this.maxWidth = maxWidth;
        this.maxHeight = maxHeight;
        this.morphologyKernelSize = morphologyKernelSize;
        this.morphologyOpenIterations = morphologyOpenIterations;
        this.morphologyCloseIterations = morphologyCloseIterations;
        this.maxForegroundFraction = maxForegroundFraction;
        this.confidenceSaturationRatio = confidenceSaturationRatio;
        this.maxDetections = maxDetections;

        binary = new byte[pixelCount];
        labels = new int[pixelCount];
        floodQueue = new int[pixelCount];
        morphologyBufferA = new byte[pixelCount];
        morphologyBufferB = new byte[pixelCount];
    }

    public List<Vector2> DetectCenters(ushort[] frame)
    {
        if (frame == null) throw new ArgumentNullException(nameof(frame));
        if (frame.Length < pixelCount) throw new ArgumentException("Frame is smaller than the configured image size.");

        detections.Clear();

        ushort frameMin;
        ushort frameMax;
        int threshold = ResolveThreshold(frame, out frameMin, out frameMax);
        int compareThreshold = Math.Max(threshold, 1);
        if (frameMax <= 0 || frameMax == frameMin || frameMax < compareThreshold)
        {
            return new List<Vector2>();
        }

        int foregroundCount = BuildBinary(frame, compareThreshold, includeEqual: true);
        int maxForegroundPixels = (int)(pixelCount * maxForegroundFraction);
        if (foregroundCount > maxForegroundPixels)
        {
            BuildBinary(frame, compareThreshold, includeEqual: false);
        }

        ApplyMorphologyIfNeeded();
        FindDetections(frame, threshold, frameMax);

        detections.Sort((left, right) => right.Confidence.CompareTo(left.Confidence));
        if (maxDetections > 0 && detections.Count > maxDetections)
        {
            detections.RemoveRange(maxDetections, detections.Count - maxDetections);
        }

        List<Vector2> centers = new List<Vector2>(detections.Count);
        for (int i = 0; i < detections.Count; ++i)
        {
            centers.Add(detections[i].Center);
        }

        return centers;
    }

    private int ResolveThreshold(ushort[] frame, out ushort frameMin, out ushort frameMax)
    {
        Array.Clear(histogram, 0, histogram.Length);

        frameMin = ushort.MaxValue;
        frameMax = ushort.MinValue;
        for (int i = 0; i < pixelCount; ++i)
        {
            ushort value = frame[i];
            histogram[value]++;
            if (value < frameMin) frameMin = value;
            if (value > frameMax) frameMax = value;
        }

        int threshold;
        if (thresholdPercentile <= 0f)
        {
            threshold = frameMin;
        }
        else if (thresholdPercentile >= 100f)
        {
            threshold = frameMax;
        }
        else
        {
            int targetIndex = (int)Math.Floor((thresholdPercentile / 100.0) * (pixelCount - 1));
            int cumulative = 0;
            threshold = frameMax;
            for (int value = 0; value < histogram.Length; ++value)
            {
                cumulative += histogram[value];
                if (cumulative > targetIndex)
                {
                    threshold = value;
                    break;
                }
            }
        }

        return Math.Max(threshold, minimumThreshold);
    }

    private int BuildBinary(ushort[] frame, int compareThreshold, bool includeEqual)
    {
        int foregroundCount = 0;
        for (int i = 0; i < pixelCount; ++i)
        {
            bool foreground = includeEqual ? frame[i] >= compareThreshold : frame[i] > compareThreshold;
            byte value = foreground ? (byte)1 : (byte)0;
            binary[i] = value;
            foregroundCount += value;
        }

        return foregroundCount;
    }

    private void ApplyMorphologyIfNeeded()
    {
        if (morphologyKernelSize <= 1) return;

        for (int i = 0; i < morphologyOpenIterations; ++i)
        {
            Erode(binary, morphologyBufferA);
            Dilate(morphologyBufferA, binary);
        }

        for (int i = 0; i < morphologyCloseIterations; ++i)
        {
            Dilate(binary, morphologyBufferB);
            Erode(morphologyBufferB, binary);
        }
    }

    private void FindDetections(ushort[] frame, int threshold, ushort frameMax)
    {
        Array.Clear(labels, 0, labels.Length);

        int componentLabel = 0;
        for (int index = 0; index < pixelCount; ++index)
        {
            if (binary[index] == 0 || labels[index] != 0) continue;

            componentLabel++;
            Component component = FloodFillComponent(frame, index, componentLabel);
            ComponentGeometry geometry = EstimateContourGeometry(component, componentLabel);
            if (geometry.Area < minArea || geometry.Area > maxArea) continue;

            if (geometry.Perimeter <= 0f) continue;

            float circularity = (float)(4.0 * Math.PI * geometry.Area / (geometry.Perimeter * geometry.Perimeter));
            if (circularity < minCircularity) continue;

            int componentWidth = component.MaxX - component.MinX + 1;
            int componentHeight = component.MaxY - component.MinY + 1;
            if (!DimensionsAllowed(componentWidth, componentHeight)) continue;

            int peakIntensity = PeakIntensityInBounds(frame, component.MinX, component.MinY, component.MaxX, component.MaxY);
            float confidence = ScoreCandidate(circularity, peakIntensity, threshold, frameMax);
            if (confidence < confidenceThreshold) continue;

            detections.Add(new Detection(
                new Vector2(component.MinX + componentWidth * 0.5f, component.MinY + componentHeight * 0.5f),
                confidence));
        }
    }

    private Component FloodFillComponent(ushort[] frame, int startIndex, int componentLabel)
    {
        int head = 0;
        int tail = 0;
        floodQueue[tail++] = startIndex;
        labels[startIndex] = componentLabel;

        int startX = startIndex % width;
        int startY = startIndex / width;
        Component component = new Component
        {
            MinX = startX,
            MaxX = startX,
            MinY = startY,
            MaxY = startY
        };

        while (head < tail)
        {
            int index = floodQueue[head++];
            int x = index % width;
            int y = index / width;

            component.Area++;
            if (x < component.MinX) component.MinX = x;
            if (x > component.MaxX) component.MaxX = x;
            if (y < component.MinY) component.MinY = y;
            if (y > component.MaxY) component.MaxY = y;

            for (int dy = -1; dy <= 1; ++dy)
            {
                int ny = y + dy;
                if (ny < 0 || ny >= height) continue;

                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0) continue;

                    int nx = x + dx;
                    if (nx < 0 || nx >= width) continue;

                    int neighborIndex = ny * width + nx;
                    if (binary[neighborIndex] == 0 || labels[neighborIndex] != 0) continue;

                    labels[neighborIndex] = componentLabel;
                    floodQueue[tail++] = neighborIndex;
                }
            }
        }

        return component;
    }

    private ComponentGeometry EstimateContourGeometry(Component component, int componentLabel)
    {
        int startX = -1;
        int startY = -1;
        for (int y = component.MinY; y <= component.MaxY && startX < 0; ++y)
        {
            int rowOffset = y * width;
            for (int x = component.MinX; x <= component.MaxX; ++x)
            {
                if (labels[rowOffset + x] != componentLabel) continue;

                startX = x;
                startY = y;
                break;
            }
        }

        if (startX < 0) return new ComponentGeometry(0f, 0f);

        const float diagonalStepLength = 1.41421356237f;
        int currentX = startX;
        int currentY = startY;
        int backtrackX = startX - 1;
        int backtrackY = startY;
        int guard = Math.Max(32, component.Area * 16);
        double signedAreaTwice = 0.0;
        float perimeter = 0f;

        for (int step = 0; step < guard; ++step)
        {
            int backtrackDirection = DirectionIndex(backtrackX - currentX, backtrackY - currentY);
            if (backtrackDirection < 0) backtrackDirection = 0;

            int nextX = currentX;
            int nextY = currentY;
            int nextBacktrackX = backtrackX;
            int nextBacktrackY = backtrackY;
            bool foundNext = false;

            for (int searchOffset = 1; searchOffset <= 8; ++searchOffset)
            {
                int direction = (backtrackDirection + searchOffset) & 7;
                int candidateX = currentX + DirectionX(direction);
                int candidateY = currentY + DirectionY(direction);
                if (!IsComponentPixel(candidateX, candidateY, componentLabel)) continue;

                int previousDirection = (direction + 7) & 7;
                nextX = candidateX;
                nextY = candidateY;
                nextBacktrackX = currentX + DirectionX(previousDirection);
                nextBacktrackY = currentY + DirectionY(previousDirection);
                foundNext = true;
                break;
            }

            if (!foundNext) return new ComponentGeometry(0f, 0f);

            signedAreaTwice += (double)currentX * nextY - (double)nextX * currentY;
            int deltaX = Math.Abs(nextX - currentX);
            int deltaY = Math.Abs(nextY - currentY);
            perimeter += (deltaX == 1 && deltaY == 1) ? diagonalStepLength : 1f;

            currentX = nextX;
            currentY = nextY;
            backtrackX = nextBacktrackX;
            backtrackY = nextBacktrackY;

            if (step > 0 && currentX == startX && currentY == startY)
            {
                float area = (float)(Math.Abs(signedAreaTwice) * 0.5);
                return new ComponentGeometry(area, perimeter);
            }
        }

        return new ComponentGeometry(0f, 0f);
    }

    private bool IsComponentPixel(int x, int y, int componentLabel)
    {
        return x >= 0
            && x < width
            && y >= 0
            && y < height
            && labels[y * width + x] == componentLabel;
    }

    private static int DirectionIndex(int deltaX, int deltaY)
    {
        for (int i = 0; i < 8; ++i)
        {
            if (DirectionX(i) == deltaX && DirectionY(i) == deltaY) return i;
        }

        return -1;
    }

    private static int DirectionX(int direction)
    {
        switch (direction)
        {
            case 0: return -1;
            case 1: return -1;
            case 2: return 0;
            case 3: return 1;
            case 4: return 1;
            case 5: return 1;
            case 6: return 0;
            case 7: return -1;
            default: return 0;
        }
    }

    private static int DirectionY(int direction)
    {
        switch (direction)
        {
            case 0: return 0;
            case 1: return -1;
            case 2: return -1;
            case 3: return -1;
            case 4: return 0;
            case 5: return 1;
            case 6: return 1;
            case 7: return 1;
            default: return 0;
        }
    }

    private bool DimensionsAllowed(int componentWidth, int componentHeight)
    {
        if (componentWidth < minWidth || componentHeight < minHeight) return false;
        if (componentWidth > maxWidth || componentHeight > maxHeight) return false;

        float aspectRatio = componentWidth / Math.Max((float)componentHeight, 1f);
        return aspectRatio >= minAspectRatio && aspectRatio <= maxAspectRatio;
    }

    private int PeakIntensityInBounds(ushort[] frame, int minX, int minY, int maxX, int maxY)
    {
        int peakIntensity = 0;
        for (int y = minY; y <= maxY; ++y)
        {
            int rowOffset = y * width;
            for (int x = minX; x <= maxX; ++x)
            {
                int value = frame[rowOffset + x];
                if (value > peakIntensity) peakIntensity = value;
            }
        }

        return peakIntensity;
    }

    private float ScoreCandidate(float circularity, int peakIntensity, int threshold, int frameMax)
    {
        if (frameMax <= 0) return 0f;

        float relativePeak = Math.Min(1f, Math.Max(0f, (float)peakIntensity / frameMax));
        float intensityScore = (float)Math.Sqrt(relativePeak);

        if (threshold > 0 && peakIntensity > threshold)
        {
            float saturation = Math.Min(
                frameMax,
                Math.Max(threshold + 1f, threshold * confidenceSaturationRatio));
            float contrastScore = (peakIntensity - threshold) / Math.Max(saturation - threshold, 1f);
            intensityScore = Math.Max(intensityScore, Math.Min(1f, Math.Max(0f, contrastScore)));
        }

        return Math.Min(1f, Math.Max(0f, circularity * intensityScore));
    }

    private void Erode(byte[] source, byte[] target)
    {
        int radius = morphologyKernelSize / 2;
        for (int y = 0; y < height; ++y)
        {
            int rowOffset = y * width;
            for (int x = 0; x < width; ++x)
            {
                bool keep = true;
                for (int ky = -radius; ky <= radius && keep; ++ky)
                {
                    int ny = y + ky;
                    if (ny < 0 || ny >= height)
                    {
                        keep = false;
                        break;
                    }

                    int neighborRowOffset = ny * width;
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        int nx = x + kx;
                        if (nx < 0 || nx >= width || source[neighborRowOffset + nx] == 0)
                        {
                            keep = false;
                            break;
                        }
                    }
                }

                target[rowOffset + x] = keep ? (byte)1 : (byte)0;
            }
        }
    }

    private void Dilate(byte[] source, byte[] target)
    {
        int radius = morphologyKernelSize / 2;
        for (int y = 0; y < height; ++y)
        {
            int rowOffset = y * width;
            for (int x = 0; x < width; ++x)
            {
                bool keep = false;
                for (int ky = -radius; ky <= radius && !keep; ++ky)
                {
                    int ny = y + ky;
                    if (ny < 0 || ny >= height) continue;

                    int neighborRowOffset = ny * width;
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        int nx = x + kx;
                        if (nx < 0 || nx >= width) continue;
                        if (source[neighborRowOffset + nx] != 0)
                        {
                            keep = true;
                            break;
                        }
                    }
                }

                target[rowOffset + x] = keep ? (byte)1 : (byte)0;
            }
        }
    }

    private struct Detection
    {
        public readonly Vector2 Center;
        public readonly float Confidence;

        public Detection(Vector2 center, float confidence)
        {
            Center = center;
            Confidence = confidence;
        }
    }

    private struct Component
    {
        public int Area;
        public int MinX;
        public int MaxX;
        public int MinY;
        public int MaxY;
    }

    private struct ComponentGeometry
    {
        public readonly float Area;
        public readonly float Perimeter;

        public ComponentGeometry(float area, float perimeter)
        {
            Area = area;
            Perimeter = perimeter;
        }
    }
}
