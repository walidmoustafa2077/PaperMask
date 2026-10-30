using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

/// <summary>
/// PaperMask console app: finds the four corners of a photographed page and warps it flat.
///
/// The model emits a 4-channel heatmap (one peak per corner) plus an 8-channel sub-pixel
/// offset map (dx, dy per corner). Corners are decoded as (argmax + offset) * stride, which
/// is what the Python trainer measures; the previous argmax-only decode was up to one
/// heatmap cell (4 px at 256, ~63 px on a 4032 px photo) off.
///
/// Divergence from Python's decode_corners: the C# side validates every quad (in frame,
/// positive area, clockwise, above the confidence threshold). Python reports invalid quads
/// instead of rejecting them, on purpose, so a bad model still shows up in the metrics.
/// </summary>
class Program
{
    static readonly string[] ImageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };

    // Defaults; overridden by model.json next to the ONNX file.
    sealed class ModelMeta
    {
        public int ArchVersion = 2;
        public int InputSize = 256;
        public int HeatmapSize = 64;
        public int Stride = 4;
        public string InputName = "image";
        public string HeatmapName = "heatmaps";
        public string OffsetName = "offsets";
        public double ConfidenceThreshold = 0.3;
        public float[] Mean = { 0.485f, 0.456f, 0.406f };
        public float[] Std = { 0.229f, 0.224f, 0.225f };

        public static ModelMeta LoadOrDefault(string? path)
        {
            var meta = new ModelMeta();
            if (path == null || !File.Exists(path)) return meta;

            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;

            if (root.TryGetProperty("arch_version", out var av) && av.TryGetInt32(out var v))
                meta.ArchVersion = v;
            if (root.TryGetProperty("confidence_threshold", out var ct))
                meta.ConfidenceThreshold = ct.GetDouble();

            if (root.TryGetProperty("input", out var input))
            {
                if (input.TryGetProperty("name", out var n) && n.GetString() is { Length: > 0 } s) meta.InputName = s;
                if (input.TryGetProperty("height", out var hh)) meta.InputSize = hh.GetInt32();
                if (input.TryGetProperty("mean", out var mean)) meta.Mean = mean.EnumerateArray().Select(e => e.GetSingle()).ToArray();
                if (input.TryGetProperty("std", out var std)) meta.Std = std.EnumerateArray().Select(e => e.GetSingle()).ToArray();
            }

            if (root.TryGetProperty("output", out var output))
            {
                if (output.TryGetProperty("heatmaps", out var hm) && hm.GetString() is { Length: > 0 } hs) meta.HeatmapName = hs;
                if (output.TryGetProperty("offsets", out var off) && off.GetString() is { Length: > 0 } os) meta.OffsetName = os;
                if (output.TryGetProperty("stride", out var st)) meta.Stride = st.GetInt32();
                if (output.TryGetProperty("heatmap_size", out var hsz)) meta.HeatmapSize = hsz.GetInt32();
            }

            if (meta.Mean.Length != 3 || meta.Std.Length != 3)
                throw new InvalidOperationException($"model.json input mean/std must have 3 values (got {meta.Mean.Length}/{meta.Std.Length})");

            return meta;
        }
    }

    internal sealed record Corner(double X, double Y, double Score);

    static ModelMeta Meta = new();
    static InferenceSession? Session;
    static string? RepoRoot;
    static string OutputRelativeDir = "../outputs/dotnet_cropped";

    static int Main(string[] args)
    {
        Console.WriteLine("PaperMask - document corner detection\n");

        RepoRoot = FindRepoRoot();

        string? modelArg = args.FirstOrDefault(a => !a.StartsWith("-"));
        string modelPath = ResolveModelPath(modelArg);
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"ERROR: model not found at {modelPath}");
            Console.WriteLine("Pass a path as the first argument, or place paper_v2.onnx in exports/.");
            return 1;
        }

        try
        {
            string metaPath = Path.Combine(Path.GetDirectoryName(modelPath)!, "model.json");
            Meta = ModelMeta.LoadOrDefault(metaPath);
            Session = new InferenceSession(modelPath);
            Console.WriteLine($"Model: {Path.GetFileName(modelPath)}");
            Console.WriteLine($"Meta:  {(File.Exists(metaPath) ? "model.json" : "defaults (no model.json found)")}");
            Console.WriteLine($"Input: {Meta.InputName} {Meta.InputSize}x{Meta.InputSize}, heatmap {Meta.HeatmapSize}, stride {Meta.Stride}, threshold {Meta.ConfidenceThreshold:0.00}");
            if (!Session.OutputMetadata.ContainsKey(Meta.OffsetName))
                Console.WriteLine($"WARNING: model has no '{Meta.OffsetName}' output - falling back to integer-cell corners.");
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR loading model:\n{ex}");
            return 1;
        }

        if (args.Contains("--quit")) return 0;

        while (true)
        {
            Console.WriteLine("Options:");
            Console.WriteLine("  File path   - process one image");
            Console.WriteLine("  Folder path - process every image below it (recursive)");
            Console.WriteLine("  q           - quit");
            Console.Write("\nPath or command: ");

            string? input = Console.ReadLine()?.Trim().Trim('"');
            if (string.IsNullOrEmpty(input)) continue;
            if (input.Equals("q", StringComparison.OrdinalIgnoreCase)) break;

            if (Directory.Exists(input)) ProcessFolder(input);
            else if (File.Exists(input)) ProcessSingleFile(input);
            else Console.WriteLine("Invalid path.\n");
        }

        Session?.Dispose();
        Console.WriteLine("\nBye.");
        return 0;
    }

    static string FindRepoRoot()
    {
        // Walk up looking for the .csproj (source layout) and fall back to the binary
        // directory (published layout, where the csproj is gone).
        for (string? dir = AppContext.BaseDirectory; dir != null; dir = Path.GetDirectoryName(dir))
            if (Directory.EnumerateFiles(dir, "*.csproj").Any()) return dir;
        return AppContext.BaseDirectory;
    }

    static string ResolveModelPath(string? explicitPath)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath)) return Path.GetFullPath(explicitPath);

        var env = Environment.GetEnvironmentVariable("PAPERMASK_MODEL");
        if (!string.IsNullOrWhiteSpace(env)) return Path.GetFullPath(env);

        string root = RepoRoot ?? AppContext.BaseDirectory;
        foreach (var candidate in new[]
                 {
                     Path.Combine(root, "..", "exports", "paper_v2.onnx"),
                     Path.Combine(root, "exports", "paper_v2.onnx"),
                     Path.Combine(root, "..", "exports", "heatmap_phase3.onnx"),   // legacy, no offsets
                 })
        {
            string full = Path.GetFullPath(candidate);
            if (File.Exists(full)) return full;
        }

        return Path.GetFullPath(Path.Combine(root, "..", "exports", "paper_v2.onnx"));
    }

    // ---------------------------------------------------------------- inference

    static Corner[]? Predict(Mat image, out string? rejectReason)
    {
        rejectReason = null;

        // The model was trained on square views of the page. Rather than squashing a
        // non-square photo (which distorts the page and biases every corner), take the
        // largest centred square and remember where it came from so corners map back.
        int side = Math.Min(image.Width, image.Height);
        int offsetX = (image.Width - side) / 2;
        int offsetY = (image.Height - side) / 2;

        using var square = new Mat(image, new Rect(offsetX, offsetY, side, side));
        using var resized = new Mat();
        Cv2.Resize(square, resized, new Size(Meta.InputSize, Meta.InputSize), interpolation: InterpolationFlags.Area);

        var pixelData = ToNormalizedChw(resized, Meta.Mean, Meta.Std, Meta.InputSize);
        var inputTensor = new DenseTensor<float>(pixelData, new[] { 1, 3, Meta.InputSize, Meta.InputSize });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(Meta.InputName, inputTensor) };

        using var results = Session!.Run(inputs);

        var heatmapValue = results.FirstOrDefault(r => r.Name == Meta.HeatmapName)
                           ?? throw new InvalidOperationException($"model produced no '{Meta.HeatmapName}' output");
        var heatmaps = heatmapValue.AsTensor<float>().ToArray();

        float[] offsets;
        var offsetValue = results.FirstOrDefault(r => r.Name == Meta.OffsetName);
        if (offsetValue != null)
        {
            offsets = offsetValue.AsTensor<float>().ToArray();
        }
        else
        {
            offsets = new float[8 * Meta.HeatmapSize * Meta.HeatmapSize];
        }

        int hm = Meta.HeatmapSize;
        var corners = Decode(heatmaps, offsets, 4, hm, hm, Meta.Stride);

        // Back to source pixels: 256-space -> square crop -> original image.
        double scale = (double)side / Meta.InputSize;
        var mapped = corners
            .Select(c => new Corner(offsetX + c.X * scale, offsetY + c.Y * scale, c.Score))
            .ToArray();

        if (mapped.Any(c => c.Score < Meta.ConfidenceThreshold))
        {
            rejectReason = $"low confidence (scores {string.Join(", ", mapped.Select(c => c.Score.ToString("0.000")))})";
            return null;
        }

        if (!ValidateQuad(mapped, image.Width, image.Height, out string why))
        {
            rejectReason = why;
            return null;
        }

        return mapped;
    }

    static float[] ToNormalizedChw(Mat bgr, float[] mean, float[] std, int size)
    {
        using var floatMat = new Mat();
        bgr.ConvertTo(floatMat, MatType.CV_32FC3, 1.0 / 255.0);

        var chw = new float[3 * size * size];
        int plane = size * size;
        var indexer = floatMat.GetGenericIndexer<Vec3f>();

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                var pixel = indexer[y, x];
                int i = y * size + x;
                chw[0 * plane + i] = (pixel.Item2 - mean[0]) / std[0]; // R
                chw[1 * plane + i] = (pixel.Item1 - mean[1]) / std[1]; // G
                chw[2 * plane + i] = (pixel.Item0 - mean[2]) / std[2]; // B
            }
        }

        return chw;
    }

    /// <summary>Mirrors <c>decode_corners</c> in <c>src/utils/corners.py</c>.</summary>
    internal static Corner[] Decode(float[] heatmaps, float[] offsets, int corners, int h, int w, int stride)
    {
        var result = new Corner[corners];
        int plane = h * w;

        for (int c = 0; c < corners; c++)
        {
            int best = 0;
            float bestValue = float.MinValue;
            int channelBase = c * plane;

            for (int i = 0; i < plane; i++)
            {
                float value = heatmaps[channelBase + i];
                if (value > bestValue)
                {
                    bestValue = value;
                    best = i;
                }
            }

            int iy = best / w;
            int ix = best % w;
            float ox = offsets[(2 * c) * plane + best];
            float oy = offsets[(2 * c + 1) * plane + best];

            result[c] = new Corner((ix + ox) * stride, (iy + oy) * stride, bestValue);
        }

        return result;
    }

    /// <summary>In frame, positive area, and clockwise in image coordinates (TL, TR, BR, BL).</summary>
    internal static bool ValidateQuad(Corner[] c, int width, int height, out string reason)
    {
        reason = string.Empty;
        if (c.Length != 4) { reason = $"expected 4 corners, got {c.Length}"; return false; }

        foreach (var p in c)
        {
            if (double.IsNaN(p.X) || double.IsNaN(p.Y) || double.IsInfinity(p.X) || double.IsInfinity(p.Y))
            {
                reason = "non-finite corner";
                return false;
            }

            if (p.X < 0 || p.Y < 0 || p.X > width || p.Y > height)
            {
                reason = $"corner ({p.X:0.0}, {p.Y:0.0}) outside {width}x{height}";
                return false;
            }
        }

        double twiceArea = 0;
        for (int i = 0; i < 4; i++)
        {
            var a = c[i];
            var b = c[(i + 1) % 4];
            twiceArea += a.X * b.Y - b.X * a.Y;
        }

        double area = Math.Abs(twiceArea) / 2;
        if (area < 64)
        {
            reason = $"quad area {area:0.0} too small";
            return false;
        }

        for (int i = 0; i < 4; i++)
        {
            var a = c[i];
            var b = c[(i + 1) % 4];
            var d = c[(i + 2) % 4];
            double cross = (b.X - a.X) * (d.Y - b.Y) - (b.Y - a.Y) * (d.X - b.X);
            if (cross <= 0)
            {
                reason = $"quad not clockwise at corner {i} (cross {cross:0.0})";
                return false;
            }
        }

        return true;
    }

    // ---------------------------------------------------------------- file handling

    static void ProcessSingleFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            Console.WriteLine($"ERROR: file not found {filePath}\n");
            return;
        }

        string outputDir = Path.GetFullPath(Path.Combine(RepoRoot ?? AppContext.BaseDirectory, OutputRelativeDir));
        string? result = ProcessOne(filePath, outputDir);
        if (result != null)
        {
            Console.WriteLine($"-> {result}");
            Console.Write("Show preview? (y/n): ");
            if (Console.ReadLine()?.Trim().ToLowerInvariant() == "y") ShowPreview(filePath, result);
        }

        Console.WriteLine();
    }

    static void ProcessFolder(string folderPath)
    {
        string root = Path.GetFullPath(folderPath);
        string outputDir = Path.GetFullPath(Path.Combine(RepoRoot ?? AppContext.BaseDirectory, OutputRelativeDir));

        string[] files;
        try
        {
            files = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
                .Where(f => ImageExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR scanning {root}:\n{ex}\n");
            return;
        }

        Console.WriteLine($"Found {files.Length} images under {root}\n");
        int ok = 0, failed = 0;

        for (int i = 0; i < files.Length; i++)
        {
            // Mirror the input tree: <out>/<relative dir>/<name>_cropped.jpg
            string relativeDir = Path.GetDirectoryName(Path.GetRelativePath(root, files[i])) ?? string.Empty;
            string targetDir = Path.Combine(outputDir, relativeDir);
            string? result = ProcessOne(files[i], targetDir, quiet: true);
            if (result != null) { ok++; Console.Write('.'); } else { failed++; Console.Write('X'); }

            if ((i + 1) % 50 == 0) Console.WriteLine($"  [{i + 1}/{files.Length}] ok={ok} failed={failed}");
        }

        Console.WriteLine($"\n\nDone: {ok} cropped, {failed} rejected in {outputDir}\n");
    }

    static string? ProcessOne(string filePath, string outputDir, bool quiet = false)
    {
        try
        {
            using var image = Cv2.ImRead(filePath, ImreadModes.Color);
            if (image.Empty())
            {
                if (!quiet) Console.WriteLine($"ERROR: cannot read {Path.GetFileName(filePath)}");
                return null;
            }

            Corner[]? corners = Predict(image, out string? reject);
            if (corners == null)
            {
                if (!quiet) Console.WriteLine($"REJECTED {Path.GetFileName(filePath)}: {reject}");
                else if (reject != null) Console.WriteLine($"\n  rejected {Path.GetFileName(filePath)}: {reject}");
                return null;
            }

            using var cropped = Warp(image, corners);
            Directory.CreateDirectory(outputDir);

            string outputFile = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(filePath) + "_cropped.jpg");
            if (!Cv2.ImWrite(outputFile, cropped))
            {
                if (!quiet) Console.WriteLine($"ERROR: could not write {outputFile}");
                return null;
            }

            if (!quiet) Console.WriteLine($"Cropped {Path.GetFileName(filePath)}");
            return outputFile;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR processing {Path.GetFileName(filePath)}:\n{ex}");
            return null;
        }
    }

    static Mat Warp(Mat image, Corner[] corners)
    {
        var points = corners.Select(c => new Point2f((float)c.X, (float)c.Y)).ToArray();

        float width = Math.Max(Distance(points[0], points[1]), Distance(points[3], points[2]));
        float height = Math.Max(Distance(points[0], points[3]), Distance(points[1], points[2]));
        width = Math.Max(width, 1f);
        height = Math.Max(height, 1f);

        Point2f[] destination =
        {
            new(0, 0),
            new(width - 1, 0),
            new(width - 1, height - 1),
            new(0, height - 1),
        };

        using var matrix = Cv2.GetPerspectiveTransform(points, destination);
        var warped = new Mat();
        Cv2.WarpPerspective(image, warped, matrix, new Size((int)width, (int)height), InterpolationFlags.Linear, BorderTypes.Replicate);
        return warped;
    }

    static float Distance(Point2f a, Point2f b)
    {
        float dx = a.X - b.X;
        float dy = a.Y - b.Y;
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    static void ShowPreview(string originalPath, string croppedPath)
    {
        try
        {
            using var original = Cv2.ImRead(originalPath, ImreadModes.Color);
            using var cropped = Cv2.ImRead(croppedPath, ImreadModes.Color);
            if (original.Empty() || cropped.Empty()) return;

            Corner[]? corners = Predict(original, out _);
            if (corners != null)
            {
                var intCorners = corners.Select(c => new Point((int)Math.Round(c.X), (int)Math.Round(c.Y))).ToArray();
                Cv2.Polylines(original, new[] { intCorners }, true, new Scalar(0, 255, 0), 3);
                for (int i = 0; i < intCorners.Length; i++)
                {
                    Cv2.Circle(original, intCorners[i], 8, new Scalar(0, 0, 255), -1);
                    Cv2.PutText(original, i.ToString(), intCorners[i] + new Point(15, -15),
                        HersheyFonts.HersheySimplex, 1.0, new Scalar(255, 255, 0), 2);
                }
            }

            const string left = "Detected corners";
            const string right = "Warped";
            Cv2.NamedWindow(left, WindowFlags.Normal);
            Cv2.NamedWindow(right, WindowFlags.Normal);
            Cv2.ImShow(left, original);
            Cv2.ImShow(right, cropped);
            Cv2.ResizeWindow(left, 640, 480);
            Cv2.ResizeWindow(right, 640, 480);
            Console.WriteLine("Press any key in the preview window to continue...");
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Preview error:\n{ex}");
        }
    }
}
