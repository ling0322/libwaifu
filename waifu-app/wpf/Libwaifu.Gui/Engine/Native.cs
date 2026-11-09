// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Libwaifu.Gui.Engine
{
    /// <summary>
    /// The C API, declared as it is in <c>waifu.h</c>: ABI 5, pipelines made for one task.
    /// </summary>
    /// <remarks>
    /// Nothing here does anything of its own: the layouts are the header's, field for field, and
    /// the rules that come with them are in docs/ffi-abi5.md. Only what this program calls is
    /// declared. The two rules that matter most, and that the callers above have to keep:
    ///
    /// * **A callback runs on a thread of the library's.** It has to return quickly, must not
    ///   call back in (bar `waifu_pipeline_cancel`), and everything it is handed is valid only
    ///   until it returns -- so anything kept is copied inside the call.
    /// * **A delegate handed over has to outlive the call.** The library keeps the pointer, and a
    ///   collected delegate is a crash with no stack worth reading; every one of them is held in
    ///   a field for as long as the library can reach it.
    ///
    /// Every enum of the header is an <c>int32_t</c> and a list of numbers that only grows: a
    /// newer library can hand over one this program has no name for, so whatever switches on one
    /// keeps a default arm.
    /// </remarks>
    internal static class Native
    {
        /// <summary>
        /// Cargo names it after the crate and CMake renames the copy; this is the name beside
        /// the program.
        /// </summary>
        public const string Library = "waifu.dll";

        /// <summary>
        /// The model manager: the catalogue, fetching, where models are kept. A library of its
        /// own, which links no engine, beside the engine's.
        /// </summary>
        public const string ModelManagerLibrary = "waifu_modelmanager.dll";

        /// <summary>What a caller compiled against this header expects, from waifu.h.</summary>
        public const uint AbiVersion = 5;

        /// <summary>The same, from waifu_modelmanager.h.</summary>
        public const uint ModelManagerAbiVersion = 1;

        public enum LogLevel
        {
            Debug = 0,
            Info = 1,
            Warning = 2,
            Error = 3,
            Fatal = 4,
        }

        public enum Status
        {
            Ok = 0,
            Cancelled = 1,
            InvalidArgument = 2,
            UnknownModel = 3,
            // 4 was FETCH, and is not reused.
            Model = 5,
            NotLoaded = 6,
            Internal = 7,
            DeviceUnavailable = 8,
        }

        public enum Device
        {
            Auto = 0,
            Cpu = 1,
            Metal = 2,
            Cuda = 3,
            // 4 was CUDA_CPU_OFFLOAD, and is not reused.
            Vulkan = 5,
        }

        /// <summary>What a pipeline is made for.</summary>
        public enum PipelineType
        {
            ImageGeneration = 1,
            TextToSpeech = 2,
            VoiceConversion = 3,
            LayerDecomposition = 4,
            SpeechToText = 5,
            StreamingTextToSpeech = 6,
        }

        /// <summary>
        /// What can be asked of a model. A SUPPORTS_ property is 1 or 0, and where it is 0 the
        /// string property of the same number says why, in words to show; SUPPORTS_&lt;task&gt;
        /// has the number of the pipeline type for that task.
        /// </summary>
        public enum ModelProperty
        {
            SupportsImageGeneration = 1,
            SupportsTextToSpeech = 2,
            SupportsVoiceConversion = 3,
            SupportsLayerDecomposition = 4,
            SupportsSpeechToText = 5,
            SupportsStreamingTextToSpeech = 6,

            SupportsImageGenerationFromStartImage = 100,
            SupportsImageGenerationWithControlImage = 101,
            SupportsImageGenerationWithPromptImages = 102,

            SupportsTextToSpeechLikeReference = 200,
            SupportsTextToSpeechWithoutReference = 201,
            SupportsTextToSpeechWithStyle = 202,

            SupportsVoiceConversionOfStyle = 300,

            /// <summary>Integer, bytes; absent until every file is here.</summary>
            SizeOnDisk = 1000,

            /// <summary>Integer, of what it says or converts.</summary>
            SampleRate = 1001,

            /// <summary>Integer: 16, or 32 for SDXL and Qwen-Image.</summary>
            ImageGenerationWidthHeightMultipleOf = 1100,

            /// <summary>String: "Euler".</summary>
            ImageGenerationSampler = 1101,

            ImageGenerationMaxPromptImages = 1102,
            LayerDecompositionResolutionMultipleOf = 1400,
            SpeechToTextLanguages = 1500,
        }

        /// <summary>
        /// What a job is doing, by task: <c>WaifuStage</c>, of the tasks this program runs.
        /// Done and total count what the comment says, and are 0, 0 otherwise.
        /// </summary>
        public static class ProgressStage
        {
            public const int LoadingWeights = 1;

            public const int ImageGenerationEncodingPrompt = 100;
            public const int ImageGenerationDenoising = 101;          // steps
            public const int ImageGenerationDecodingImage = 102;

            public const int TextToSpeechReadingText = 200;
            public const int TextToSpeechGeneratingTokens = 201;      // tokens; total is an estimate
            public const int TextToSpeechSynthesizingAudio = 202;

            public const int VoiceConversionReadingRecordings = 300;
            public const int VoiceConversionGeneratingTokens = 301;   // tokens; total is an estimate
            public const int VoiceConversionDenoising = 302;          // steps
            public const int VoiceConversionSynthesizingAudio = 303;  // vocoder windows
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Log
        {
            public LogLevel Level;
            public IntPtr Source;
            public IntPtr Message;
            public IntPtr Time;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct ProgressInfo
        {
            public int Stage;
            public double Fraction;
            public ulong Done;
            public ulong Total;
            public double Seconds;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct StatusInfo
        {
            public Status Code;
            public IntPtr Message;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Result
        {
            public StatusInfo Status;
        }

        // -- waifu_modelmanager.dll's own types, which share nothing with the above ----------

        public enum ManagerStatus
        {
            Ok = 0,
            Cancelled = 1,
            InvalidArgument = 2,
            UnknownModel = 3,
            Fetch = 4,
            Internal = 5,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct ManagerLogEvent
        {
            public LogLevel Level;
            public IntPtr Source;
            public IntPtr Message;
        }

        /// <summary>How far a download is: the file, its bytes, and which hub it comes from.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ManagerProgressEvent
        {
            public double Fraction;
            public ulong Done;
            public ulong Total;
            public uint Part;
            public uint Parts;
            public IntPtr File;
            public IntPtr Words;
            public double Seconds;
            public IntPtr Hub;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct ManagerStatusInfo
        {
            public ManagerStatus Code;
            public IntPtr Message;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct ManagerCompletionEvent
        {
            public ManagerStatusInfo Status;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ManagerLogCallback(IntPtr user, ref ManagerLogEvent e);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ManagerProgressCallback(IntPtr user, ref ManagerProgressEvent e);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ManagerCompleteCallback(IntPtr user, ref ManagerCompletionEvent e);

        [StructLayout(LayoutKind.Sequential)]
        public struct Image
        {
            public uint Width;
            public uint Height;
            public IntPtr Rgb;
            public UIntPtr Length;
        }

        /// <summary>A WaifuPromptImage: an image the prompt names as &lt;|key|&gt;.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct PromptImage
        {
            public IntPtr Key;
            public IntPtr Image;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct ImageGenerationRequest
        {
            public uint StructSize;
            public IntPtr Prompt;
            public IntPtr Negative;
            public uint Width;
            public uint Height;
            public uint Steps;
            public float Guidance;
            public ulong Seed;

            /// <summary>A WaifuImage* to start from (img2img), or zero for none.</summary>
            public IntPtr StartImage;

            public float StartImageStrength;

            /// <summary>A ControlNet's condition, a WaifuImage*, or zero for none.</summary>
            public IntPtr ControlImage;

            public float ControlImageScale;

            /// <summary>A WaifuPromptImage array, or zero.</summary>
            public IntPtr PromptImages;

            public UIntPtr PromptImageCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct ImageGenerationResult
        {
            public StatusInfo Status;
            public IntPtr Image;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Audio
        {
            public IntPtr Samples;
            public UIntPtr Count;
            public uint SampleRate;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct TextToSpeechRequest
        {
            public uint StructSize;
            public IntPtr Text;

            /// <summary>A WaifuAudio* to sound like, or zero.</summary>
            public IntPtr Reference;

            /// <summary>What Reference says, as a UTF-8 char*, or zero where it is not known.</summary>
            public IntPtr ReferenceTranscript;

            public float Speed;
            public float Temperature;
            public IntPtr Style;
            public ulong Seed;
        }

        /// <summary>What both speech tasks end with.</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct AudioResult
        {
            public StatusInfo Status;
            public IntPtr Audio;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct VoiceConversionRequest
        {
            public uint StructSize;
            public IntPtr Source;
            public IntPtr Reference;
            public uint Steps;

            // One byte in C, where .NET would otherwise send four.
            [MarshalAs(UnmanagedType.U1)]
            public bool ConvertStyle;

            public ulong Seed;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LogCallback(IntPtr user, ref Log e);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ProgressCallback(IntPtr user, ref ProgressInfo e);

        /// <summary>A WaifuCallback: the end of a load or an unload.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ResultCallback(IntPtr user, ref Result e);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ImageGenerationCallback(IntPtr user, ref ImageGenerationResult e);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void AudioCallback(IntPtr user, ref AudioResult e);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint waifu_abi_version();

        /// <summary>This thread's last refused call; valid until its next one.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_last_error();

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern Status waifu_last_error_code();

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_set_log_callback(IntPtr user, LogCallback callback);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_set_log_level(LogLevel level);

        /// <summary>
        /// Reads the manifest at a path -- or "tones": see waifu_modelmanager_manifest_path --
        /// and none of the weights. Zero where it is refused, with the reason in
        /// waifu_last_error. Freed with waifu_model_free.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_model_open(byte[] manifestPath);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_model_free(IntPtr model);

        /// <summary>False where the model has no such property; the value is then untouched.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_model_get_int_property(
            IntPtr model, ModelProperty property, out long value);

        /// <summary>Zero where there is none; valid until waifu_model_free.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_model_get_string_property(IntPtr model, ModelProperty property);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint waifu_modelmanager_abi_version();

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_modelmanager_last_error();

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_modelmanager_string_free(IntPtr text);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_modelmanager_set_log_callback(IntPtr user, ManagerLogCallback callback);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_modelmanager_set_log_level(LogLevel level);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_modelmanager_catalog_json();

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_modelmanager_directory();

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern ManagerStatus waifu_modelmanager_set_directory(byte[] path);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern ManagerStatus waifu_modelmanager_set_bundled_manifests(byte[] directory);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern ManagerStatus waifu_modelmanager_remove(byte[] name);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_modelmanager_manifest_path(byte[] name);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_modelmanager_defaults_json(byte[] model);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_modelmanager_fetch_async(
            byte[] model, IntPtr user, ManagerProgressCallback onProgress, ManagerCompleteCallback onComplete);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_modelmanager_fetch_manifests_async(
            IntPtr user, ManagerProgressCallback onProgress, ManagerCompleteCallback onComplete);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_modelmanager_fetch_cancel(IntPtr fetch);

        [DllImport(ModelManagerLibrary, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_modelmanager_fetch_free(IntPtr fetch);

        /// <summary>
        /// Starts a pipeline's thread, with no model. Zero where the device is not available
        /// (WAIFU_ERR_DEVICE_UNAVAILABLE), with the reason in waifu_last_error.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_pipeline_new(PipelineType type, Device device);

        /// <summary>
        /// Stops what runs, cancels what waits, drops the model and joins the thread; returns once
        /// every job has had its on_complete. Never from one of its own callbacks.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_pipeline_free(IntPtr pipeline);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_pipeline_cancel(IntPtr pipeline, ulong job);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_pipeline_cancel_all(IntPtr pipeline);

        /// <summary>
        /// Refused -- job 0 -- for a model the pipeline's task is not supported by. The model may
        /// be freed as soon as this returns.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong waifu_pipeline_load_model_async(
            IntPtr pipeline,
            IntPtr model,
            IntPtr user,
            ProgressCallback onProgress,
            ResultCallback onComplete);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong waifu_pipeline_run_image_generation_async(
            IntPtr pipeline,
            ref ImageGenerationRequest request,
            IntPtr user,
            ProgressCallback onProgress,
            ImageGenerationCallback onComplete);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong waifu_pipeline_run_text_to_speech_async(
            IntPtr pipeline,
            ref TextToSpeechRequest request,
            IntPtr user,
            ProgressCallback onProgress,
            AudioCallback onComplete);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong waifu_pipeline_run_voice_conversion_async(
            IntPtr pipeline,
            ref VoiceConversionRequest request,
            IntPtr user,
            ProgressCallback onProgress,
            AudioCallback onComplete);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetDllDirectory(string directory);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr LoadLibrary(string path);
    }

    /// <summary>
    /// Strings across the boundary: UTF-8 both ways, NUL-terminated, and borrowed only for the
    /// length of the call. .NET Framework has no reader for them, so here is one.
    /// </summary>
    internal static class Utf8
    {
        /// <summary>What to hand the library for a string it reads during the call.</summary>
        public static byte[] Bytes(string text)
        {
            if (text == null)
            {
                return null;
            }

            var raw = Encoding.UTF8.GetBytes(text);
            var zeroed = new byte[raw.Length + 1];
            Buffer.BlockCopy(raw, 0, zeroed, 0, raw.Length);
            return zeroed;
        }

        /// <summary>A string the library handed over, copied before the callback returns.</summary>
        public static string Read(IntPtr pointer)
        {
            if (pointer == IntPtr.Zero)
            {
                return null;
            }

            var length = 0;
            while (Marshal.ReadByte(pointer, length) != 0)
            {
                length++;
            }

            if (length == 0)
            {
                return string.Empty;
            }

            var raw = new byte[length];
            Marshal.Copy(pointer, raw, 0, length);
            return Encoding.UTF8.GetString(raw);
        }

        /// <summary>
        /// A string the model manager returned, read and then freed as it asks. waifu.dll hands
        /// out none to free: its strings are its own, and are only read.
        /// </summary>
        public static string TakeFromModelManager(IntPtr pointer)
        {
            if (pointer == IntPtr.Zero)
            {
                return null;
            }

            try
            {
                return Read(pointer);
            }
            finally
            {
                Native.waifu_modelmanager_string_free(pointer);
            }
        }
    }
}
