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
    /// The C API, declared as it is in <c>waifu.h</c>.
    /// </summary>
    /// <remarks>
    /// Nothing here does anything of its own: the layouts are the header's, field for field, and
    /// the rules that come with them are in docs/ffi.md. The two that matter most, and that the
    /// callers above have to keep:
    ///
    /// * **A callback runs on a thread of the library's.** It has to return quickly, must not
    ///   call back in (bar `waifu_engine_cancel` and the `_free` functions), and everything it is
    ///   handed is valid only until it returns -- so anything kept is copied inside the call.
    /// * **A delegate handed over has to outlive the call.** The library keeps the pointer, and a
    ///   collected delegate is a crash with no stack worth reading; every one of them is held in
    ///   a field for as long as the library can reach it.
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
        public const uint AbiVersion = 2;

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
            Fetch = 4,
            Model = 5,
            NotLoaded = 6,
            Internal = 7,
        }

        public enum Device
        {
            Auto = 0,
            Cpu = 1,
            Metal = 2,
            Cuda = 3,
            CudaCpuOffload = 4,
            Vulkan = 5,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LogEvent
        {
            public LogLevel Level;
            public IntPtr Source;
            public IntPtr Message;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct ProgressEvent
        {
            public int Stage;
            public double Fraction;
            public ulong Done;
            public ulong Total;
            public uint Part;
            public uint Parts;
            public IntPtr File;
            public IntPtr Words;
            public double Seconds;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct StatusInfo
        {
            public Status Code;
            public IntPtr Message;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct CompletionEvent
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

        [StructLayout(LayoutKind.Sequential)]
        public struct ImageCompletionEvent
        {
            public StatusInfo Status;
            public IntPtr Image;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Audio
        {
            public IntPtr Samples;
            public UIntPtr Count;
            public uint Rate;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct AudioCompletionEvent
        {
            public StatusInfo Status;
            public IntPtr Audio;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct ImageInput
        {
            public IntPtr Key;
            public IntPtr Image;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct DrawRequest
        {
            public uint StructSize;
            public IntPtr Prompt;
            public IntPtr Negative;
            public IntPtr Images;
            public UIntPtr ImageCount;
            public uint Width;
            public uint Height;
            public uint Steps;
            public float Guidance;
            public ulong Seed;
            public float Strength;
            public float ControlScale;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct SpeakRequest
        {
            public uint StructSize;
            public IntPtr Text;
            public IntPtr Like;
            public float Speed;
            public float Temperature;
            public IntPtr Style;
            public ulong Seed;
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
        public delegate void LogCallback(IntPtr user, ref LogEvent e);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ProgressCallback(IntPtr user, ref ProgressEvent e);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void CompleteCallback(IntPtr user, ref CompletionEvent e);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ImageCompleteCallback(IntPtr user, ref ImageCompletionEvent e);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void AudioCompleteCallback(IntPtr user, ref AudioCompletionEvent e);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint waifu_abi_version();

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_last_error();

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_string_free(IntPtr text);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_set_log_callback(IntPtr user, LogCallback callback);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_set_log_level(LogLevel level);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_machine_json();

        /// <summary>What the manifest at a path says, or "tones": see waifu_modelmanager_manifest_path.</summary>
        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_describe_json(byte[] model);

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

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr waifu_engine_new(Device device);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_engine_free(IntPtr engine);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_engine_cancel(IntPtr engine, ulong job);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern void waifu_engine_cancel_all(IntPtr engine);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong waifu_engine_load_async(
            IntPtr engine,
            byte[] model,
            IntPtr user,
            ProgressCallback onProgress,
            CompleteCallback onComplete);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong waifu_engine_unload_async(
            IntPtr engine, IntPtr user, CompleteCallback onComplete);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong waifu_engine_draw_async(
            IntPtr engine,
            ref DrawRequest request,
            IntPtr user,
            ProgressCallback onProgress,
            ImageCompleteCallback onComplete);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong waifu_engine_speak_async(
            IntPtr engine,
            ref SpeakRequest request,
            IntPtr user,
            ProgressCallback onProgress,
            AudioCompleteCallback onComplete);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong waifu_engine_voice_conversion_async(
            IntPtr engine,
            ref VoiceConversionRequest request,
            IntPtr user,
            ProgressCallback onProgress,
            AudioCompleteCallback onComplete);

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

        /// <summary>A string a synchronous call returned, read and then freed as it asks.</summary>
        public static string Take(IntPtr pointer)
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
                Native.waifu_string_free(pointer);
            }
        }

        /// <summary>The same for a string the model manager returned, which it frees itself.</summary>
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
