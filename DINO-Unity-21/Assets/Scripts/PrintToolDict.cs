using UnityEngine;
using System.Linq;
using System.Collections.Generic;
using System.Text;

/** @file           PrintToolDict.cs
 *  @brief          Helper Unity utils script to update a TextMesh with the contents
 *                  of a ToolDictionary grabbed from UnityToolManager
 *
 *  @author         Hisham Iqbal
 *  @copyright      &copy; 2023 Hisham Iqbal
 */
public class PrintToolDict : MonoBehaviour
{
    public UnityToolManager toolMgr;
    public TMPro.TextMeshProUGUI meshText;
    public float refreshIntervalSeconds = 0.25f;

    IReadOnlyDictionary<int, ToolTrackingUtils.TrackedTool> ToolDictToPrint = new Dictionary<int, ToolTrackingUtils.TrackedTool>();
    private readonly StringBuilder stringBuilder = new StringBuilder(512);
    private float nextRefreshTime;

    // Update is called once per frame
    void Update()
    {
        if (Time.unscaledTime < nextRefreshTime) return;
        nextRefreshTime = Time.unscaledTime + Mathf.Max(0f, refreshIntervalSeconds);
        PrintToolDictionary();
    }

    /// <summary>
    /// Grab the tool dictionary exposed by UnityToolManager, and print its contents
    /// </summary>
    private void PrintToolDictionary()
    {
        if (toolMgr == null || meshText == null) return;
        ToolDictToPrint = toolMgr.GetToolDictionary();

        stringBuilder.Length = 0;

        // cast to ToArray to avoid race-condition issues?
        foreach (var pair in ToolDictToPrint.ToArray())
        {
            stringBuilder.Append(pair.Key).Append('\n');
            stringBuilder.Append(pair.Value.Tool_HoloFrame_LH.ToString("F3"));
        }

        meshText.text = stringBuilder.ToString();
    }
}
