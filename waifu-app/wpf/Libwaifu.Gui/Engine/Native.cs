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
    /// The C API, declared as it is in <c>waifu.h</c>: ABI 6, requests set an attribute at a
    /// time and tasks read rather than called back.
    /// </summary>
    /// <remarks>
    /// Nothing here does anything of its own: the layouts are the header's, field for field, and
    /// the rules that come with them are in docs/ffi-abi6.md. Only what this program calls is
    /// declared. The rules that matter most, and that the callers above have to keep:
    ///
    /// * **A request is copied.** Every setter copies what it is handed, and
    ///   <c>waifu_pipeline_run</c> copies the request, which may be freed as soon as it returns.
    /// * **A task is read, and freed once.** <c>waifu_task_next_event</c> is asked by one thread
    ///   at a time; everything a task hands out is valid until <c>waifu_task_free</c>, which must
    ///   not race with any other call on the same task.
    /// * **A delegate handed over has to outlive the call.** The library keeps the log
    ///   callback's pointer, and a collected delegate is a crash with no stack worth reading; it
    ///   is held in a field for as long as the library can reach it.
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

        /// <summary>
        /// What a caller compiled against this header expects, from waifu.h: a library of the
        /// same major version whose minor is at least this one's. waifu_abi_version says
        /// (major &lt;&lt; 16) | minor.
        /// </summary>
        public const uint AbiVersionMajor = 6;

        public const uint AbiVersionMinor = 0;

        /// <summary>The same, from waifu_modelmanager.h.</summary>
        public const uint ModelManagerAbiVersion = 1;

        /// <summary>waifu_task_next_event and waifu_task_wait: no timeout.</summary>
        public const uint WaitForever = uint.MaxValue;

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
            UnknownAttribute = 9,
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
        /// What can be asked of a model. A SUPPORTS_ attribute is 1 or 0, and where it is 0 the
        /// string attribute of the same number says why, in words to show; SUPPORTS_&lt;task&gt;
        /// has the number of the pipeline type for that task.
        /// </summary>
        public enum ModelAttribute
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

        /// <summary>
        /// A request's attributes, of the tasks this program runs: <c>WaifuRequestAttribute</c>.
        /// The setter each takes is after it; one left unset takes the library's default.
        /// </summary>
        public enum RequestAttribute
        {
            ImageGenerationPrompt = 100,              // string, required
            ImageGenerationNegative = 101,            // string
            ImageGenerationWidth = 102,               // int, 64..2048
            ImageGenerationHeight = 103,              // int, as width
            ImageGenerationSteps = 104,               // int, 1..
            ImageGenerationGuidance = 105,            // float, 0..
            ImageGenerationSeed = 106,                // int, read as uint64
            ImageGenerationStartImage = 107,          // image (img2img)
            ImageGenerationStartImageStrength = 108,  // float, 0..1

            TextToSpeechText = 200,                   // string, required
            TextToSpeechReference = 201,              // audio
            TextToSpeechReferenceTranscript = 202,    // string
            TextToSpeechSpeed = 203,                  // float, above 0
            TextToSpeechTemperature = 204,            // float, 0..
            TextToSpeechStyle = 205,                  // string
            TextToSpeechSeed = 206,                   // int, read as uint64

            VoiceConversionSource = 300,              // audio, required
            VoiceConversionReference = 301,           // audio, required
            VoiceConversionSteps = 302,               // int, 1..
            VoiceConversionConvertStyle = 303,        // int, 0 or 1
            VoiceConversionSeed = 304,                // int, read as uint64
        }

        /// <summary>What waifu_task_next_event says happened: <c>WaifuEventType</c>.</summary>
        public enum EventType
        {
            /// <summary>Nothing before the timeout.</summary>
            None = 0,

            /// <summary>The PROGRESS_ attributes say where the task is now.</summary>
            Progress = 1,

            /// <summary>Streaming text to speech only, which this program does not run.</summary>
            AudioPart = 2,

            /// <summary>STATUS is there, and the result for WAIFU_OK. END for every call after.</summary>
            End = 3,
        }

        /// <summary>
        /// What a task can be asked, of the tasks this program runs: <c>WaifuTaskAttribute</c>.
        /// </summary>
        public enum TaskAttribute
        {
            /// <summary>Int, a <see cref="Native.Status"/>; absent until the task has ended.</summary>
            Status = 1,

            /// <summary>String, why; absent for WAIFU_OK and WAIFU_CANCELLED.</summary>
            StatusMessage = 2,

            /// <summary>Int, a stage: the last PROGRESS event's; absent before one.</summary>
            ProgressStage = 10,

            ProgressDone = 11,                        // int
            ProgressTotal = 12,                       // int
            ProgressFraction = 13,                    // float, 0..1; absent where nothing can say
            ProgressSeconds = 14,                     // float, since the task started

            ImageGenerationImage = 100,               // image
            TextToSpeechAudio = 200,                  // audio
            VoiceConversionAudio = 300,               // audio
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Log
        {
            public LogLevel Level;
            public IntPtr Source;
            public IntPtr Message;
            public IntPtr Time;
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

        [StructLayout(LayoutKind.Sequential)]
        public struct Audio
        {
            public IntPtr Samples;
            public UIntPtr Count;
            public uint SampleRate;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LogCallback(IntPtr user, ref Log e);

        /// <summary>(major &lt;&lt; 16) | minor.</summary>
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

        /// <summary>False where the model has no such attribute; the value is then untouched.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_model_get_int_attribute(
            IntPtr model, ModelAttribute attribute, out long value);

        /// <summary>Zero where there is none; valid until waifu_model_free.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_model_get_string_attribute(IntPtr model, ModelAttribute attribute);

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
        /// Stops what runs, cancels what waits, drops the model and joins the thread. Returns once
        /// every task of the pipeline has ended -- WAIFU_CANCELLED for what had not -- and the
        /// tasks are still the caller's to free.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_pipeline_free(IntPtr pipeline);

        /// <summary>Every task of the pipeline, waiting or running, is cancelled.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_pipeline_cancel_all(IntPtr pipeline);

        /// <summary>
        /// A WaifuTask*, or zero where it is refused -- for a model the pipeline's task is not
        /// supported by -- with the reason in waifu_last_error. The model may be freed as soon as
        /// this returns.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_pipeline_load_model(IntPtr pipeline, IntPtr model);

        /// <summary>
        /// A WaifuTask*, or zero where it is refused -- a request of another task, a required
        /// attribute unset, one that contradicts another -- with the reason in waifu_last_error.
        /// The request is copied, and may be freed as soon as this returns.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_pipeline_run(IntPtr pipeline, IntPtr request);

        // -- requests ----------------------------------------------------------------------------

        /// <summary>
        /// A WaifuRequest* with every attribute unset; zero for a type the library does not know.
        /// Freed with waifu_request_free.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_request_new(PipelineType type);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_request_free(IntPtr request);

        /// <summary>
        /// Each setter copies what it is handed. False, with the reason in waifu_last_error, for a
        /// attribute that is not this task's, a value of another type or one out of range; the
        /// attribute is then as it was.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_request_set_int_attribute(
            IntPtr request, RequestAttribute attribute, long value);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_request_set_float_attribute(
            IntPtr request, RequestAttribute attribute, double value);

        /// <summary>A NUL-terminated UTF-8 string, from <see cref="Utf8.Bytes"/>.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_request_set_string_attribute(
            IntPtr request, RequestAttribute attribute, byte[] value);

        /// <summary>The pixels <c>value.Rgb</c> points at are pinned for the call, and copied.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_request_set_image_attribute(
            IntPtr request, RequestAttribute attribute, ref Image value);

        /// <summary>The samples <c>value.Samples</c> points at are pinned for the call, and copied.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_request_set_audio_attribute(
            IntPtr request, RequestAttribute attribute, ref Audio value);

        // -- tasks -------------------------------------------------------------------------------

        /// <summary>
        /// The next thing that happened to the task, waiting up to <paramref name="timeoutMs"/>
        /// for it. END once it has ended, and END for every call after. One thread at a time.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern EventType waifu_task_next_event(IntPtr task, uint timeoutMs);

        /// <summary>After the step it is on; it then ends with WAIFU_CANCELLED. Nothing once it has ended.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_task_cancel(IntPtr task);

        /// <summary>
        /// Cancels a task that has not ended; every pointer it handed out is then gone. Must not
        /// race with any other call on the same task.
        /// </summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_task_free(IntPtr task);

        /// <summary>False where the attribute is not there; the value is then untouched.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_task_get_int_attribute(
            IntPtr task, TaskAttribute attribute, out long value);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_task_get_float_attribute(
            IntPtr task, TaskAttribute attribute, out double value);

        /// <summary>Zero where it is not there; valid until waifu_task_free.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_task_get_string_attribute(IntPtr task, TaskAttribute attribute);

        /// <summary>The pixels are the task's, valid until waifu_task_free.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_task_get_image_attribute(
            IntPtr task, TaskAttribute attribute, out Image value);

        /// <summary>The samples are the task's, valid until waifu_task_free.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        public static extern bool waifu_task_get_audio_attribute(
            IntPtr task, TaskAttribute attribute, out Audio value);

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

        /// <summary>
        /// A string the library handed over, copied while it is valid: before a callback returns,
        /// or before the task or model it came from is freed.
        /// </summary>
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
