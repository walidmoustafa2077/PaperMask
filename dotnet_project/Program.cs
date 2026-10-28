using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

class Program {
    static string ModelPath = "../exports/heatmap_phase3.onnx";
    static string OutputPath = "../outputs/dotnet_cropped";
    static InferenceSession? session;
    static string? BaseDirectory;

    static void Main() {
        Console.WriteLine("Document Corner Detection\n");
        
        // Determine base directory (project root) by finding .csproj file
        BaseDirectory = AppContext.BaseDirectory;
        string? currentDir = BaseDirectory;
        while (currentDir != null && !Directory.GetFiles(currentDir, "*.csproj").Any()) {
            currentDir = Path.GetDirectoryName(currentDir);
        }
        BaseDirectory = currentDir ?? BaseDirectory;
        
        try {
            string fullModelPath = Path.GetFullPath(Path.Combine(BaseDirectory, ModelPath));
            if (!File.Exists(fullModelPath)) {
                Console.WriteLine($"ERROR: Model not found at {fullModelPath}");
                Console.WriteLine("Press any key to exit...");
                try { Console.Read(); } catch { /* Handle redirected input */ }
                return;
            }
            session = new InferenceSession(fullModelPath);
            Console.WriteLine($"Loaded model: {Path.GetFileName(fullModelPath)}\n");
            
            // Print model input/output information
            Console.WriteLine("Model Input Names:");
            foreach (var input in session.InputMetadata) {
                Console.WriteLine($"  - {input.Key}");
            }
            Console.WriteLine("Model Output Names:");
            foreach (var output in session.OutputMetadata) {
                Console.WriteLine($"  - {output.Key}");
            }
            Console.WriteLine();
        } catch (Exception ex) {
            Console.WriteLine($"ERROR loading model: {ex.Message}");
            Console.WriteLine("Press any key to exit...");
            try { Console.Read(); } catch { /* Handle redirected input */ }
            return;
        }

        while (true) {
            Console.WriteLine("Options:");
            Console.WriteLine("  File path: Process a single image (e.g., D:\\path\\to\\image.jpg)");
            Console.WriteLine("  Folder path: Batch process all images in folder (e.g., D:\\path\\to\\folder)");
            Console.WriteLine("  q: Quit");
            Console.Write("\nEnter path or command: ");
            
            string? input = Console.ReadLine()?.Trim().Trim('"');
            if (string.IsNullOrEmpty(input)) continue;

            if (input.ToLower() == "q") break;
            
            if (Directory.Exists(input)) {
                Console.WriteLine($"\nBatch processing all images in {input}...");
                ProcessFolder(input);
            } else if (File.Exists(input)) {
                ProcessSingleFile(input, true);
            } else {
                Console.WriteLine("Invalid path. Try again.\n");
            }
        }
        Console.WriteLine("\nExiting...");
    }

    static void ProcessSingleFile(string filePath, bool showPreviewPrompt = false) {
        if (BaseDirectory == null) return;
        if (!File.Exists(filePath)) {
            Console.WriteLine($"ERROR: File not found at {filePath}\n");
            return;
        }

        try {
            Mat img = Cv2.ImRead(filePath);
            if (img.Empty()) {
                Console.WriteLine($"ERROR: Could not read image {Path.GetFileName(filePath)}\n");
                return;
            }

            Point2f[] corners = PredictCorners(img);
            if (corners == null) {
                Console.WriteLine($"ERROR: Prediction failed\n");
                return;
            }

            Mat cropped = PerspectiveTransform(img, corners);
            string outputDir = Path.GetFullPath(Path.Combine(BaseDirectory, OutputPath));
            Directory.CreateDirectory(outputDir);
            
            string outputFile = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(filePath) + "_cropped.jpg");
            Cv2.ImWrite(outputFile, cropped);
            Console.WriteLine($"Processed {Path.GetFileName(filePath)} -> {Path.GetFileName(outputFile)}");

            if (showPreviewPrompt) {
                Console.Write("Show preview? (y/n): ");
                if (Console.ReadLine()?.ToLower() == "y") {
                    ShowPreview(img, corners, cropped, 0);
                }
            }
            Console.WriteLine();
        } catch (Exception ex) {
            Console.WriteLine($"ERROR: {ex.Message}\n");
        }
    }

    static void ProcessFolder(string folderPath) {
        if (BaseDirectory == null) return;
        if (!Directory.Exists(folderPath)) {
            Console.WriteLine($"ERROR: Folder not found at {folderPath}\n");
            return;
        }

        try {
            string outputDir = Path.GetFullPath(Path.Combine(BaseDirectory, OutputPath));
            Directory.CreateDirectory(outputDir);

            // Get all image files from the folder
            string[] imageFiles = Directory.GetFiles(folderPath, "*.jpg")
                .Concat(Directory.GetFiles(folderPath, "*.jpeg"))
                .Concat(Directory.GetFiles(folderPath, "*.png"))
                .ToArray();

            int success = 0, failed = 0;
            Console.WriteLine($"Found {imageFiles.Length} images\n");

            for (int i = 0; i < imageFiles.Length; i++) {
                try {
                    string filePath = imageFiles[i];
                    Mat img = Cv2.ImRead(filePath);
                    if (img.Empty()) {
                        failed++;
                        Console.Write("X");
                        continue;
                    }

                    Point2f[] corners = PredictCorners(img);
                    if (corners == null) {
                        failed++;
                        Console.Write("X");
                        continue;
                    }

                    Mat cropped = PerspectiveTransform(img, corners);
                    string outputFile = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(filePath) + "_cropped.jpg");
                    Cv2.ImWrite(outputFile, cropped);
                    success++;
                    Console.Write(".");

                    img.Dispose();
                    cropped.Dispose();
                } catch {
                    failed++;
                    Console.Write("X");
                }

                if ((i + 1) % 50 == 0) Console.WriteLine($" [{i + 1}/{imageFiles.Length}]");
            }

            Console.WriteLine($"\n\nBatch complete: {success} succeeded, {failed} failed\n");
        } catch (Exception ex) {
            Console.WriteLine($"ERROR: {ex.Message}\n");
        }
    }

    static Point2f[] PredictCorners(Mat img) {
        try {
            Mat resized = new Mat();
            Cv2.Resize(img, resized, new Size(256, 256));

            Mat normalized = new Mat();
            resized.ConvertTo(normalized, MatType.CV_32FC3, 1.0 / 255.0);
            
            float[] pixelData = new float[3 * 256 * 256];
            float[] mean = { 0.485f, 0.456f, 0.406f };
            float[] std = { 0.229f, 0.224f, 0.225f };

            for (int h = 0; h < 256; h++) {
                for (int w = 0; w < 256; w++) {
                    Vec3f pixel = normalized.At<Vec3f>(h, w);
                    int idx = h * 256 + w;
                    pixelData[0 * 256 * 256 + idx] = (pixel[2] - mean[0]) / std[0];
                    pixelData[1 * 256 * 256 + idx] = (pixel[1] - mean[1]) / std[1];
                    pixelData[2 * 256 * 256 + idx] = (pixel[0] - mean[2]) / std[2];
                }
            }

            var inputTensor = new DenseTensor<float>(pixelData, new[] { 1, 3, 256, 256 });
            var inputs = new List<NamedOnnxValue> { 
                NamedOnnxValue.CreateFromTensor("image", inputTensor) 
            };

            using var results = session!.Run(inputs);
            var outputTensor = results.First(o => o.Name == "heatmaps").AsTensor<float>();
            
            Point2f[] corners = new Point2f[4];
            float scaleX = (float)img.Width / 256f;
            float scaleY = (float)img.Height / 256f;

            for (int c = 0; c < 4; c++) {
                float maxVal = float.MinValue;
                int maxY = 0, maxX = 0;
                
                for (int h = 0; h < 64; h++) {
                    for (int w = 0; w < 64; w++) {
                        float val = outputTensor[0, c, h, w];
                        if (val > maxVal) {
                            maxVal = val;
                            maxY = h;
                            maxX = w;
                        }
                    }
                }
                
                float x = maxX * 4f * scaleX;
                float y = maxY * 4f * scaleY;
                corners[c] = new Point2f(x, y);
            }

            resized.Dispose();
            normalized.Dispose();
            return corners;
        } catch (Exception ex) {
            Console.WriteLine($"Prediction error: {ex.Message}");
            return null!;
        }
    }

    static Mat PerspectiveTransform(Mat img, Point2f[] corners) {
        float width = Math.Max(
            Distance(corners[0], corners[1]),
            Distance(corners[3], corners[2])
        );
        float height = Math.Max(
            Distance(corners[0], corners[3]),
            Distance(corners[1], corners[2])
        );

        Point2f[] dst = {
            new Point2f(0, 0),
            new Point2f(width - 1, 0),
            new Point2f(width - 1, height - 1),
            new Point2f(0, height - 1)
        };

        Mat matrix = Cv2.GetPerspectiveTransform(corners, dst);
        Mat warped = new Mat();
        Cv2.WarpPerspective(img, warped, matrix, new Size((int)width, (int)height));
        
        matrix.Dispose();
        return warped;
    }

    static float Distance(Point2f p1, Point2f p2) {
        float dx = p1.X - p2.X;
        float dy = p1.Y - p2.Y;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }

    static void ShowPreview(Mat original, Point2f[] corners, Mat cropped, int imageNum) {
        try {
            Mat display = original.Clone();
            
            Point[] intCorners = corners.Select(p => new Point((int)p.X, (int)p.Y)).ToArray();
            Cv2.Polylines(display, new Point[][] { intCorners }, true, new Scalar(0, 255, 0), 3);
            
            for (int i = 0; i < 4; i++) {
                Cv2.Circle(display, intCorners[i], 8, new Scalar(0, 0, 255), -1);
                Cv2.PutText(display, $"{i}", intCorners[i] + new Point(15, -15), 
                    HersheyFonts.HersheySimplex, 1.0, new Scalar(255, 255, 0), 2);
            }
            
            string originalWindowName = "Original - Corner Detection";
            string croppedWindowName = "Cropped - Result";
            
            // Create resizable windows
            Cv2.NamedWindow(originalWindowName, WindowFlags.Normal);
            Cv2.NamedWindow(croppedWindowName, WindowFlags.Normal);
            
            // Show images
            Cv2.ImShow(originalWindowName, display);
            Cv2.ImShow(croppedWindowName, cropped);
            
            // Resize windows to fit screen (approximately 1/3 of screen each for side-by-side view)
            Cv2.ResizeWindow(originalWindowName, 640, 480);
            Cv2.ResizeWindow(croppedWindowName, 640, 480);
            
            Console.WriteLine("Press any key in preview window to close...");
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
            
            display.Dispose();
        } catch (Exception ex) {
            Console.WriteLine($"Preview error: {ex.Message}");
        }
    }
}
