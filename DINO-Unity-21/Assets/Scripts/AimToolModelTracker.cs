using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Loads aimtool marker templates and their matching OBJ assets from Resources,
/// matches each complete template against the currently observed marker subset,
/// then applies the resulting rigid pose to the rendered model.
/// </summary>
public sealed class AimToolModelTracker : MonoBehaviour
{
    [Header("Resources")]
    public string resourcesFolder = "AimTools";
    public bool loadResourcesOnStart = true;

    [Header("Matching")]
    public float maxMarkerErrorMetres = 0.015f;
    public float distanceToleranceMetres = 0f;
    public int maxObservedMarkers = 32;
    public int maxSearchNodesPerTool = 200000;
    public float lostVisibilityTimeoutSeconds = 0.25f;
    public float jitterSmoothingDistanceMetres = 0.01f;
    [Range(0f, 1f)]
    public float jitterSmoothingFactor = 0.35f;
    public bool hideModelsWhenUnmatched = true;
    public bool logMatches = false;

    [Header("Rendering")]
    public bool overrideModelMaterials = true;
    public Color modelColor = new Color(0.15f, 0.75f, 1f, 0.45f);
    public int modelRenderQueue = 2990;

    private readonly List<AimToolRuntimeModel> models = new List<AimToolRuntimeModel>();
    private Material sharedModelMaterial;
    private bool resourcesLoaded;

    private void Start()
    {
        if (loadResourcesOnStart)
        {
            LoadFromResources();
        }
    }

    private void OnDestroy()
    {
        for (int i = 0; i < models.Count; ++i)
        {
            if (models[i].Instance != null)
            {
                Destroy(models[i].Instance);
            }
        }
    }

    public void LoadFromResources()
    {
        if (resourcesLoaded) return;

        TextAsset[] markerAssets = Resources.LoadAll<TextAsset>(resourcesFolder);
        Array.Sort(markerAssets, (left, right) => string.CompareOrdinal(left.name, right.name));

        for (int i = 0; i < markerAssets.Length; ++i)
        {
            TextAsset markerAsset = markerAssets[i];
            if (!markerAsset.name.EndsWith(".markers", StringComparison.OrdinalIgnoreCase)) continue;

            AimToolMarkerSet markerSet;
            try
            {
                markerSet = JsonUtility.FromJson<AimToolMarkerSet>(markerAsset.text);
            }
            catch (Exception ex)
            {
                Debug.LogWarning(string.Format("Failed to parse AimTool marker asset {0}: {1}", markerAsset.name, ex.Message));
                continue;
            }

            if (markerSet == null || markerSet.markers == null || markerSet.markers.Length < 3)
            {
                Debug.LogWarning(string.Format("AimTool marker asset {0} does not contain at least 3 markers.", markerAsset.name));
                continue;
            }

            string modelName = string.IsNullOrEmpty(markerSet.name)
                ? markerAsset.name.Replace(".markers", string.Empty)
                : markerSet.name;
            string modelResourcePath = string.IsNullOrEmpty(markerSet.modelResourcePath)
                ? resourcesFolder + "/" + modelName
                : markerSet.modelResourcePath;

            GameObject prefab = Resources.Load<GameObject>(modelResourcePath);
            if (prefab == null)
            {
                Debug.LogWarning(string.Format("AimTool model prefab not found at Resources/{0}.", modelResourcePath));
                continue;
            }

            AimToolRuntimeModel model = new AimToolRuntimeModel
            {
                Name = modelName,
                Prefab = prefab,
                MarkerLocalPositions = ConvertMarkers(markerSet.markers),
                LastMatchTime = float.NegativeInfinity,
                HasSmoothedPose = false
            };

            models.Add(model);
        }

        resourcesLoaded = true;

        if (models.Count == 0)
        {
            Debug.LogWarning(string.Format("No AimTool marker/model pairs were loaded from Resources/{0}.", resourcesFolder));
        }
    }

    public void UpdateObservedMarkers(IList<Vector3> observedMarkerWorldPositions)
    {
        if (!resourcesLoaded)
        {
            LoadFromResources();
        }

        if (models.Count == 0) return;

        List<Vector3> observed = CopyObservedMarkers(observedMarkerWorldPositions);
        float distanceTolerance = distanceToleranceMetres > 0f
            ? distanceToleranceMetres
            : maxMarkerErrorMetres * 2f;

        List<AimToolModelCandidate> candidates = new List<AimToolModelCandidate>(models.Count);
        for (int i = 0; i < models.Count; ++i)
        {
            AimToolRuntimeModel model = models[i];
            AimToolRigidMatch match;
            bool matched = AimToolRigidMatcher.TryFindBestMatch(
                observed,
                model.MarkerLocalPositions,
                maxMarkerErrorMetres,
                distanceTolerance,
                maxSearchNodesPerTool,
                out match);

            if (matched)
            {
                candidates.Add(new AimToolModelCandidate(i, match));

                if (logMatches)
                {
                    Debug.Log(string.Format(
                        "AimTool {0} matched {1} markers, max error {2:F4} m, RMS {3:F4} m.",
                        model.Name,
                        match.MatchedCount,
                        match.MaxError,
                        match.RmsError));
                }
            }
        }

        List<AimToolModelCandidate> selected = SelectNonOverlappingCandidates(candidates);
        bool[] selectedModels = new bool[models.Count];
        for (int i = 0; i < selected.Count; ++i)
        {
            AimToolModelCandidate candidate = selected[i];
            AimToolRuntimeModel model = models[candidate.ModelIndex];
            selectedModels[candidate.ModelIndex] = true;

            EnsureModelInstance(model);
            if (model.Instance == null) continue;

            if (!model.Instance.activeSelf) model.Instance.SetActive(true);
            ApplyPoseWithJitterSmoothing(model, candidate.Match.Translation, candidate.Match.Rotation);
            model.LastMatchTime = Time.unscaledTime;
        }

        for (int i = 0; i < models.Count; ++i)
        {
            if (!selectedModels[i])
            {
                HideModelIfStale(models[i]);
            }
        }
    }

    private static List<AimToolModelCandidate> SelectNonOverlappingCandidates(List<AimToolModelCandidate> candidates)
    {
        if (candidates == null || candidates.Count == 0) return new List<AimToolModelCandidate>();

        candidates.Sort((left, right) =>
        {
            int scoreCompare = right.Match.Score.CompareTo(left.Match.Score);
            if (scoreCompare != 0) return scoreCompare;

            int rmsCompare = left.Match.RmsError.CompareTo(right.Match.RmsError);
            if (rmsCompare != 0) return rmsCompare;

            return left.Match.MaxError.CompareTo(right.Match.MaxError);
        });

        List<AimToolModelCandidate> best = new List<AimToolModelCandidate>();
        List<AimToolModelCandidate> current = new List<AimToolModelCandidate>();
        float bestScore = float.NegativeInfinity;
        float[] suffixUpper = new float[candidates.Count + 1];
        for (int i = candidates.Count - 1; i >= 0; --i)
        {
            suffixUpper[i] = suffixUpper[i + 1] + Mathf.Max(0f, candidates[i].Match.Score);
        }

        SearchCandidateSelection(candidates, 0, 0UL, 0f, suffixUpper, current, ref best, ref bestScore);
        return best;
    }

    private static void SearchCandidateSelection(
        List<AimToolModelCandidate> candidates,
        int index,
        ulong usedObservedMask,
        float score,
        float[] suffixUpper,
        List<AimToolModelCandidate> current,
        ref List<AimToolModelCandidate> best,
        ref float bestScore)
    {
        if (score + suffixUpper[index] < bestScore - 1e-6f) return;

        if (index == candidates.Count)
        {
            if (SelectionIsBetter(current, score, best, bestScore))
            {
                bestScore = score;
                best = new List<AimToolModelCandidate>(current);
            }

            return;
        }

        AimToolModelCandidate candidate = candidates[index];
        if ((usedObservedMask & candidate.Match.ObservedMask) == 0UL)
        {
            current.Add(candidate);
            SearchCandidateSelection(
                candidates,
                index + 1,
                usedObservedMask | candidate.Match.ObservedMask,
                score + candidate.Match.Score,
                suffixUpper,
                current,
                ref best,
                ref bestScore);
            current.RemoveAt(current.Count - 1);
        }

        SearchCandidateSelection(
            candidates,
            index + 1,
            usedObservedMask,
            score,
            suffixUpper,
            current,
            ref best,
            ref bestScore);
    }

    private static bool SelectionIsBetter(
        List<AimToolModelCandidate> current,
        float currentScore,
        List<AimToolModelCandidate> best,
        float bestScore)
    {
        if (currentScore > bestScore + 1e-6f) return true;
        if (Mathf.Abs(currentScore - bestScore) > 1e-6f) return false;

        float currentRms = 0f;
        float currentMax = 0f;
        for (int i = 0; i < current.Count; ++i)
        {
            currentRms += current[i].Match.RmsError;
            currentMax += current[i].Match.MaxError;
        }

        float bestRms = 0f;
        float bestMax = 0f;
        for (int i = 0; i < best.Count; ++i)
        {
            bestRms += best[i].Match.RmsError;
            bestMax += best[i].Match.MaxError;
        }

        if (currentRms < bestRms - 1e-6f) return true;
        if (currentRms > bestRms + 1e-6f) return false;
        if (currentMax < bestMax - 1e-6f) return true;
        if (currentMax > bestMax + 1e-6f) return false;
        return current.Count > best.Count;
    }

    private List<Vector3> CopyObservedMarkers(IList<Vector3> observedMarkerWorldPositions)
    {
        int sourceCount = observedMarkerWorldPositions == null ? 0 : observedMarkerWorldPositions.Count;
        int cappedCount = Mathf.Min(sourceCount, Mathf.Clamp(maxObservedMarkers, 0, 63));
        List<Vector3> observed = new List<Vector3>(cappedCount);
        for (int i = 0; i < cappedCount; ++i)
        {
            observed.Add(observedMarkerWorldPositions[i]);
        }

        return observed;
    }

    private static List<Vector3> ConvertMarkers(AimToolMarkerPoint[] markers)
    {
        List<Vector3> result = new List<Vector3>(markers.Length);
        for (int i = 0; i < markers.Length; ++i)
        {
            result.Add(new Vector3(markers[i].x, markers[i].y, markers[i].z));
        }

        return result;
    }

    private void EnsureModelInstance(AimToolRuntimeModel model)
    {
        if (model.Instance != null) return;

        model.Instance = Instantiate(model.Prefab);
        model.Instance.name = model.Name + "_AimToolModel";
        model.Instance.transform.localScale = Vector3.one;

        if (overrideModelMaterials)
        {
            Renderer[] renderers = model.Instance.GetComponentsInChildren<Renderer>(true);
            Material material = GetSharedModelMaterial();
            for (int i = 0; i < renderers.Length; ++i)
            {
                renderers[i].sharedMaterial = material;
            }
        }

        if (hideModelsWhenUnmatched)
        {
            model.Instance.SetActive(false);
        }
    }

    private void HideModelIfStale(AimToolRuntimeModel model)
    {
        if (!hideModelsWhenUnmatched || model.Instance == null) return;

        bool neverMatched = float.IsNegativeInfinity(model.LastMatchTime);
        bool stale = Time.unscaledTime - model.LastMatchTime > lostVisibilityTimeoutSeconds;
        if ((neverMatched || stale) && model.Instance.activeSelf)
        {
            model.Instance.SetActive(false);
            model.HasSmoothedPose = false;
        }
    }

    private void ApplyPoseWithJitterSmoothing(AimToolRuntimeModel model, Vector3 targetPosition, Quaternion targetRotation)
    {
        Vector3 outputPosition = targetPosition;
        Quaternion outputRotation = targetRotation;

        if (model.HasSmoothedPose)
        {
            float displacement = Vector3.Distance(model.SmoothedPosition, targetPosition);
            if (displacement <= Mathf.Max(0f, jitterSmoothingDistanceMetres))
            {
                float t = Mathf.Clamp01(jitterSmoothingFactor);
                outputPosition = Vector3.Lerp(model.SmoothedPosition, targetPosition, t);
                outputRotation = Quaternion.Slerp(model.SmoothedRotation, targetRotation, t);
            }
        }

        model.SmoothedPosition = outputPosition;
        model.SmoothedRotation = outputRotation;
        model.HasSmoothedPose = true;
        model.Instance.transform.SetPositionAndRotation(outputPosition, outputRotation);
    }

    private Material GetSharedModelMaterial()
    {
        if (sharedModelMaterial != null) return sharedModelMaterial;

        Shader shader = Shader.Find("Standard");
        sharedModelMaterial = new Material(shader);
        sharedModelMaterial.color = modelColor;

        if (modelColor.a < 0.999f)
        {
            sharedModelMaterial.SetFloat("_Mode", 3f);
            sharedModelMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            sharedModelMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            sharedModelMaterial.SetInt("_ZWrite", 0);
            sharedModelMaterial.DisableKeyword("_ALPHATEST_ON");
            sharedModelMaterial.EnableKeyword("_ALPHABLEND_ON");
            sharedModelMaterial.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            sharedModelMaterial.renderQueue = modelRenderQueue;
        }

        return sharedModelMaterial;
    }

    [Serializable]
    private sealed class AimToolMarkerSet
    {
        public string name;
        public string sourceAimTool;
        public string sourceStl;
        public string modelResourcePath;
        public AimToolMarkerPoint[] markers;
    }

    [Serializable]
    private sealed class AimToolMarkerPoint
    {
        public float x;
        public float y;
        public float z;
    }

    private sealed class AimToolRuntimeModel
    {
        public string Name;
        public GameObject Prefab;
        public GameObject Instance;
        public List<Vector3> MarkerLocalPositions;
        public float LastMatchTime;
        public bool HasSmoothedPose;
        public Vector3 SmoothedPosition;
        public Quaternion SmoothedRotation;
    }

    private struct AimToolModelCandidate
    {
        public readonly int ModelIndex;
        public readonly AimToolRigidMatch Match;

        public AimToolModelCandidate(int modelIndex, AimToolRigidMatch match)
        {
            ModelIndex = modelIndex;
            Match = match;
        }
    }
}

public struct AimToolRigidMatch
{
    public Quaternion Rotation;
    public Vector3 Translation;
    public float MaxError;
    public float RmsError;
    public float Score;
    public int MatchedCount;
    public int[] ObservedIndices;
    public ulong ObservedMask;
}

public static class AimToolRigidMatcher
{
    public static bool TryFindBestMatch(
        IList<Vector3> observedPoints,
        IList<Vector3> templatePoints,
        float maxError,
        float distanceTolerance,
        int maxSearchNodes,
        out AimToolRigidMatch bestMatch)
    {
        bestMatch = new AimToolRigidMatch();

        if (observedPoints == null || templatePoints == null) return false;
        int observedCount = observedPoints.Count;
        int templateCount = templatePoints.Count;
        if (templateCount < 3 || observedCount < templateCount || observedCount > 63) return false;
        if (maxError < 0f || distanceTolerance < 0f) return false;

        float[,] templateDistances = PairwiseDistances(templatePoints);
        float[,] observedDistances = PairwiseDistances(observedPoints);
        int[] templateOrder = BuildTemplateSearchOrder(templatePoints, templateDistances);
        ulong[,,] compatibleMasks = BuildCompatibleMasks(templateCount, observedCount, templateDistances, observedDistances, distanceTolerance);

        SearchState state = new SearchState
        {
            ObservedPoints = observedPoints,
            TemplatePoints = templatePoints,
            TemplateOrder = templateOrder,
            CompatibleMasks = compatibleMasks,
            AssignmentByOrder = CreateFilledArray(templateCount, -1),
            AssignmentByTemplate = CreateFilledArray(templateCount, -1),
            MaxError = maxError,
            MaxSearchNodes = Math.Max(1, maxSearchNodes),
            AllObservedMask = observedCount == 64 ? ulong.MaxValue : ((1UL << observedCount) - 1UL),
            BestScore = float.NegativeInfinity,
            BestMatch = new AimToolRigidMatch()
        };

        Search(0, 0UL, ref state);
        bestMatch = state.BestMatch;
        return state.HasBestMatch;
    }

    private static void Search(int depth, ulong usedObservedMask, ref SearchState state)
    {
        if (state.SearchNodes++ >= state.MaxSearchNodes) return;

        int templateCount = state.TemplatePoints.Count;
        if (depth == templateCount)
        {
            EvaluateAssignment(ref state);
            return;
        }

        int templateNext = state.TemplateOrder[depth];
        ulong allowed = state.AllObservedMask & ~usedObservedMask;

        for (int previousDepth = 0; previousDepth < depth; ++previousDepth)
        {
            int templatePrevious = state.TemplateOrder[previousDepth];
            int observedPrevious = state.AssignmentByOrder[previousDepth];
            allowed &= state.CompatibleMasks[templatePrevious, templateNext, observedPrevious];
            if (allowed == 0UL) return;
        }

        while (allowed != 0UL)
        {
            ulong bit = allowed & (~allowed + 1UL);
            int observedNext = BitIndex(bit);
            allowed ^= bit;

            state.AssignmentByOrder[depth] = observedNext;
            state.AssignmentByTemplate[templateNext] = observedNext;
            Search(depth + 1, usedObservedMask | bit, ref state);
            state.AssignmentByTemplate[templateNext] = -1;
            state.AssignmentByOrder[depth] = -1;

            if (state.SearchNodes >= state.MaxSearchNodes) return;
        }
    }

    private static void EvaluateAssignment(ref SearchState state)
    {
        Quaternion rotation;
        Vector3 translation;
        if (!TryEstimateRigidTransform(
            state.TemplatePoints,
            state.ObservedPoints,
            state.AssignmentByTemplate,
            out rotation,
            out translation))
        {
            return;
        }

        float maxError = 0f;
        float squaredErrorSum = 0f;
        int count = state.TemplatePoints.Count;
        for (int i = 0; i < count; ++i)
        {
            Vector3 transformed = rotation * state.TemplatePoints[i] + translation;
            Vector3 target = state.ObservedPoints[state.AssignmentByTemplate[i]];
            float error = Vector3.Distance(transformed, target);
            if (error > state.MaxError + 1e-6f) return;
            if (error > maxError) maxError = error;
            squaredErrorSum += error * error;
        }

        float rmsError = Mathf.Sqrt(squaredErrorSum / count);
        float score = 1000000f * count - 1000f * rmsError - maxError;
        if (!state.HasBestMatch || score > state.BestScore)
        {
            int[] observedIndices = new int[count];
            Array.Copy(state.AssignmentByTemplate, observedIndices, count);

            state.HasBestMatch = true;
            state.BestScore = score;
            state.BestMatch = new AimToolRigidMatch
            {
                Rotation = rotation,
                Translation = translation,
                MaxError = maxError,
                RmsError = rmsError,
                Score = score,
                MatchedCount = count,
                ObservedIndices = observedIndices,
                ObservedMask = IndicesToMask(observedIndices)
            };
        }
    }

    private static bool TryEstimateRigidTransform(
        IList<Vector3> sourcePoints,
        IList<Vector3> targetPoints,
        int[] targetIndicesBySource,
        out Quaternion rotation,
        out Vector3 translation)
    {
        rotation = Quaternion.identity;
        translation = Vector3.zero;

        int count = sourcePoints.Count;
        if (count <= 0) return false;

        Vector3 sourceCenter = Vector3.zero;
        Vector3 targetCenter = Vector3.zero;
        for (int i = 0; i < count; ++i)
        {
            int targetIndex = targetIndicesBySource[i];
            if (targetIndex < 0 || targetIndex >= targetPoints.Count) return false;

            sourceCenter += sourcePoints[i];
            targetCenter += targetPoints[targetIndex];
        }

        sourceCenter /= count;
        targetCenter /= count;

        double sxx = 0.0;
        double sxy = 0.0;
        double sxz = 0.0;
        double syx = 0.0;
        double syy = 0.0;
        double syz = 0.0;
        double szx = 0.0;
        double szy = 0.0;
        double szz = 0.0;

        for (int i = 0; i < count; ++i)
        {
            Vector3 source = sourcePoints[i] - sourceCenter;
            Vector3 target = targetPoints[targetIndicesBySource[i]] - targetCenter;

            sxx += (double)source.x * target.x;
            sxy += (double)source.x * target.y;
            sxz += (double)source.x * target.z;
            syx += (double)source.y * target.x;
            syy += (double)source.y * target.y;
            syz += (double)source.y * target.z;
            szx += (double)source.z * target.x;
            szy += (double)source.z * target.y;
            szz += (double)source.z * target.z;
        }

        double[,] hornMatrix =
        {
            { sxx + syy + szz, syz - szy, szx - sxz, sxy - syx },
            { syz - szy, sxx - syy - szz, sxy + syx, szx + sxz },
            { szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy },
            { sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz }
        };

        double[] quaternion = LargestEigenVectorSymmetric4(hornMatrix);
        double norm = Math.Sqrt(
            quaternion[0] * quaternion[0]
            + quaternion[1] * quaternion[1]
            + quaternion[2] * quaternion[2]
            + quaternion[3] * quaternion[3]);
        if (norm <= 1e-12) return false;

        double sign = quaternion[0] < 0.0 ? -1.0 : 1.0;
        float w = (float)(sign * quaternion[0] / norm);
        float x = (float)(sign * quaternion[1] / norm);
        float y = (float)(sign * quaternion[2] / norm);
        float z = (float)(sign * quaternion[3] / norm);

        rotation = new Quaternion(x, y, z, w).normalized;
        translation = targetCenter - rotation * sourceCenter;
        return true;
    }

    private static double[] LargestEigenVectorSymmetric4(double[,] matrix)
    {
        double[,] a = (double[,])matrix.Clone();
        double[,] vectors =
        {
            { 1.0, 0.0, 0.0, 0.0 },
            { 0.0, 1.0, 0.0, 0.0 },
            { 0.0, 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 0.0, 1.0 }
        };

        for (int iteration = 0; iteration < 40; ++iteration)
        {
            int p = 0;
            int q = 1;
            double maxOffDiagonal = Math.Abs(a[p, q]);

            for (int row = 0; row < 4; ++row)
            {
                for (int col = row + 1; col < 4; ++col)
                {
                    double value = Math.Abs(a[row, col]);
                    if (value > maxOffDiagonal)
                    {
                        maxOffDiagonal = value;
                        p = row;
                        q = col;
                    }
                }
            }

            if (maxOffDiagonal < 1e-12) break;

            double angle = 0.5 * Math.Atan2(2.0 * a[p, q], a[q, q] - a[p, p]);
            double c = Math.Cos(angle);
            double s = Math.Sin(angle);

            for (int k = 0; k < 4; ++k)
            {
                double akp = a[k, p];
                double akq = a[k, q];
                a[k, p] = c * akp - s * akq;
                a[k, q] = s * akp + c * akq;
            }

            for (int k = 0; k < 4; ++k)
            {
                double apk = a[p, k];
                double aqk = a[q, k];
                a[p, k] = c * apk - s * aqk;
                a[q, k] = s * apk + c * aqk;
            }

            for (int k = 0; k < 4; ++k)
            {
                double vkp = vectors[k, p];
                double vkq = vectors[k, q];
                vectors[k, p] = c * vkp - s * vkq;
                vectors[k, q] = s * vkp + c * vkq;
            }
        }

        int bestIndex = 0;
        double bestValue = a[0, 0];
        for (int i = 1; i < 4; ++i)
        {
            if (a[i, i] > bestValue)
            {
                bestValue = a[i, i];
                bestIndex = i;
            }
        }

        return new[]
        {
            vectors[0, bestIndex],
            vectors[1, bestIndex],
            vectors[2, bestIndex],
            vectors[3, bestIndex]
        };
    }

    private static float[,] PairwiseDistances(IList<Vector3> points)
    {
        int count = points.Count;
        float[,] distances = new float[count, count];
        for (int i = 0; i < count; ++i)
        {
            for (int j = i + 1; j < count; ++j)
            {
                float distance = Vector3.Distance(points[i], points[j]);
                distances[i, j] = distance;
                distances[j, i] = distance;
            }
        }

        return distances;
    }

    private static ulong[,,] BuildCompatibleMasks(
        int templateCount,
        int observedCount,
        float[,] templateDistances,
        float[,] observedDistances,
        float distanceTolerance)
    {
        ulong[,,] masks = new ulong[templateCount, templateCount, observedCount];
        for (int templateA = 0; templateA < templateCount; ++templateA)
        {
            for (int templateB = 0; templateB < templateCount; ++templateB)
            {
                float targetDistance = templateDistances[templateA, templateB];
                for (int observedA = 0; observedA < observedCount; ++observedA)
                {
                    ulong mask = 0UL;
                    for (int observedB = 0; observedB < observedCount; ++observedB)
                    {
                        if (observedA == observedB) continue;
                        if (Mathf.Abs(observedDistances[observedA, observedB] - targetDistance) <= distanceTolerance)
                        {
                            mask |= 1UL << observedB;
                        }
                    }

                    masks[templateA, templateB, observedA] = mask;
                }
            }
        }

        return masks;
    }

    private static int[] BuildTemplateSearchOrder(IList<Vector3> points, float[,] distances)
    {
        int count = points.Count;
        int[] order = new int[count];
        if (count <= 2)
        {
            for (int i = 0; i < count; ++i) order[i] = i;
            return order;
        }

        int first = 0;
        int second = 1;
        float largestDistance = -1f;
        for (int i = 0; i < count; ++i)
        {
            for (int j = i + 1; j < count; ++j)
            {
                if (distances[i, j] > largestDistance)
                {
                    largestDistance = distances[i, j];
                    first = i;
                    second = j;
                }
            }
        }

        bool[] selected = new bool[count];
        order[0] = first;
        order[1] = second;
        selected[first] = true;
        selected[second] = true;
        int orderCount = 2;

        Vector3 baseVector = points[second] - points[first];
        float baseLength = baseVector.magnitude;
        if (baseLength > 1e-8f && orderCount < count)
        {
            int third = -1;
            float bestAreaHeight = -1f;
            for (int i = 0; i < count; ++i)
            {
                if (selected[i]) continue;
                float areaHeight = Vector3.Cross(baseVector, points[i] - points[first]).magnitude / baseLength;
                if (areaHeight > bestAreaHeight)
                {
                    bestAreaHeight = areaHeight;
                    third = i;
                }
            }

            if (third >= 0)
            {
                order[orderCount++] = third;
                selected[third] = true;
            }
        }

        while (orderCount < count)
        {
            int next = -1;
            float bestNearestDistance = -1f;
            for (int candidate = 0; candidate < count; ++candidate)
            {
                if (selected[candidate]) continue;

                float nearestDistance = float.PositiveInfinity;
                for (int i = 0; i < orderCount; ++i)
                {
                    nearestDistance = Mathf.Min(nearestDistance, distances[candidate, order[i]]);
                }

                if (nearestDistance > bestNearestDistance)
                {
                    bestNearestDistance = nearestDistance;
                    next = candidate;
                }
            }

            order[orderCount++] = next;
            selected[next] = true;
        }

        return order;
    }

    private static int[] CreateFilledArray(int count, int value)
    {
        int[] result = new int[count];
        for (int i = 0; i < count; ++i) result[i] = value;
        return result;
    }

    private static int BitIndex(ulong bit)
    {
        int index = 0;
        while ((bit >>= 1) != 0UL) ++index;
        return index;
    }

    private static ulong IndicesToMask(int[] indices)
    {
        ulong mask = 0UL;
        for (int i = 0; i < indices.Length; ++i)
        {
            if (indices[i] >= 0 && indices[i] < 64)
            {
                mask |= 1UL << indices[i];
            }
        }

        return mask;
    }

    private struct SearchState
    {
        public IList<Vector3> ObservedPoints;
        public IList<Vector3> TemplatePoints;
        public int[] TemplateOrder;
        public ulong[,,] CompatibleMasks;
        public int[] AssignmentByOrder;
        public int[] AssignmentByTemplate;
        public float MaxError;
        public int MaxSearchNodes;
        public int SearchNodes;
        public ulong AllObservedMask;
        public bool HasBestMatch;
        public float BestScore;
        public AimToolRigidMatch BestMatch;
    }
}
