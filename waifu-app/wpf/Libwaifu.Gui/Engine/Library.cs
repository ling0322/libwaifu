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
using System.IO;
using System.Reflection;

namespace Libwaifu.Gui.Engine
{
    /// <summary>
    /// Finding <c>waifu.dll</c>, opening it, and the two things that have to happen before any
    /// other call: the log callback, and the manifests the program was built with.
    /// </summary>
    /// <remarks>
    /// The library is opened rather than linked, so that a build of this program without it
    /// beside it still starts -- with the screen on the stand-in engine and a line in the log
    /// saying why. Which is also how this program was written: the window first, the library
    /// behind it.
    /// </remarks>
    public static class Library
    {
        // The library keeps this pointer. A delegate that is collected is a crash in somebody
        // else's stack frame, so it is held here for as long as the process lives.
        private static Native.LogCallback logSink;
        private static Native.ManagerLogCallback managerLogSink;

        private static bool looked;
        private static bool here;

        /// <summary>Whether the real engine is behind the screen.</summary>
        public static bool Available
        {
            get
            {
                if (!looked)
                {
                    looked = true;
                    here = Open();
                }

                return here;
            }
        }

        /// <summary>Why it is not, for the Model card's footnote and the log window.</summary>
        public static string Missing { get; private set; }

        /// <summary>Where it was found.</summary>
        public static string Path { get; private set; }

        private static bool Open()
        {
            try
            {
                var found = Find();
                if (found == null)
                {
                    Missing = Native.Library + " is not beside the program: the screen is running " +
                              "on the stand-in engine, which draws nothing.";
                    Log.Write("warn", "library", Missing);
                    return false;
                }

                // The folder is handed to the loader rather than the file, so that the libraries
                // it needs of its own -- CUDA's, OpenMP's -- are looked for beside it.
                Native.SetDllDirectory(System.IO.Path.GetDirectoryName(found));
                if (Native.LoadLibrary(found) == IntPtr.Zero)
                {
                    Missing = "could not open " + found + ": " +
                              new System.ComponentModel.Win32Exception(
                                  System.Runtime.InteropServices.Marshal.GetLastWin32Error()).Message;
                    Log.Write("error", "library", Missing);
                    return false;
                }

                // The model manager is a library of its own, and the screen needs both: it is
                // looked for beside the engine's, where the build writes the two of them.
                var manager = System.IO.Path.Combine(
                    System.IO.Path.GetDirectoryName(found), Native.ModelManagerLibrary);
                if (!File.Exists(manager))
                {
                    Missing = Native.ModelManagerLibrary + " is not beside " + found + ".";
                    Log.Write("error", "library", Missing);
                    return false;
                }

                if (Native.LoadLibrary(manager) == IntPtr.Zero)
                {
                    Missing = "could not open " + manager + ": " +
                              new System.ComponentModel.Win32Exception(
                                  System.Runtime.InteropServices.Marshal.GetLastWin32Error()).Message;
                    Log.Write("error", "library", Missing);
                    return false;
                }

                Path = found;

                var abi = Native.waifu_abi_version();
                if (abi != Native.AbiVersion)
                {
                    Missing = found + " is ABI " + abi + " and this program was built for " +
                              Native.AbiVersion + ".";
                    Log.Write("error", "library", Missing);
                    return false;
                }

                var managerAbi = Native.waifu_modelmanager_abi_version();
                if (managerAbi != Native.ModelManagerAbiVersion)
                {
                    Missing = manager + " is ABI " + managerAbi + " and this program was built for " +
                              Native.ModelManagerAbiVersion + ".";
                    Log.Write("error", "library", Missing);
                    return false;
                }

                Listen();
                Bundled();
                Log.Write("info", "library", "opened " + found + ", ABI " + abi);
                return true;
            }
            catch (Exception error)
            {
                Missing = error.Message;
                Log.Write("error", "library", error.ToString());
                return false;
            }
        }

        /// <summary>
        /// Beside the program first, then where an environment variable points, then where this
        /// repository's own build writes it -- which is how it is run from a build tree.
        /// </summary>
        private static string Find()
        {
            foreach (var folder in Folders())
            {
                if (string.IsNullOrEmpty(folder))
                {
                    continue;
                }

                var file = System.IO.Path.Combine(folder, Native.Library);
                if (File.Exists(file))
                {
                    return file;
                }
            }

            return null;
        }

        private static System.Collections.Generic.IEnumerable<string> Folders()
        {
            var beside = System.IO.Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            yield return beside;
            yield return Environment.GetEnvironmentVariable("LIBWAIFU_LIB_DIR");

            // ...\waifu-app\wpf\Libwaifu.Gui\bin\Debug -> the repository's build directory.
            if (beside != null)
            {
                var repository = System.IO.Path.GetFullPath(
                    System.IO.Path.Combine(beside, "..", "..", "..", "..", ".."));
                yield return System.IO.Path.Combine(repository, "build");
            }
        }

        /// <summary>
        /// Every line either library would print, from whichever thread wrote it. Set before
        /// anything else: flint writes what hardware it found the first time a device is asked
        /// about, and that line is worth having.
        /// </summary>
        private static void Listen()
        {
            logSink = OnLine;
            Native.waifu_set_log_callback(IntPtr.Zero, logSink);
            Native.waifu_set_log_level(Native.LogLevel.Info);
            managerLogSink = OnManagerLine;
            Native.waifu_modelmanager_set_log_callback(IntPtr.Zero, managerLogSink);
            Native.waifu_modelmanager_set_log_level(Native.LogLevel.Info);
        }

        private static void OnLine(IntPtr user, ref Native.LogEvent e)
        {
            // Called on any thread, including this one, and must not call back into the library.
            // Everything it is given is valid only until it returns, so both strings are copied.
            try
            {
                Log.Write(Level(e.Level), Utf8.Read(e.Source) ?? "waifu", Utf8.Read(e.Message));
            }
            catch
            {
                // A log line is not worth taking the process down for, and this frame belongs to
                // the library: an exception thrown here has nowhere to go.
            }
        }

        /// <summary>The same for waifu_modelmanager.dll, whose event is its own type.</summary>
        private static void OnManagerLine(IntPtr user, ref Native.ManagerLogEvent e)
        {
            try
            {
                Log.Write(Level(e.Level), Utf8.Read(e.Source) ?? "models", Utf8.Read(e.Message));
            }
            catch
            {
                // As above: nowhere for an exception to go.
            }
        }

        private static string Level(Native.LogLevel level)
        {
            switch (level)
            {
                case Native.LogLevel.Debug: return "debug";
                case Native.LogLevel.Warning: return "warn";
                case Native.LogLevel.Error: return "error";
                case Native.LogLevel.Fatal: return "fatal";
                default: return "info";
            }
        }

        /// <summary>
        /// The manifests shipped beside the program, where there are any: with them a model can
        /// say what it suggests before any of it has been downloaded.
        /// </summary>
        private static void Bundled()
        {
            var beside = System.IO.Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            var manifests = System.IO.Path.Combine(beside ?? ".", "manifests");
            if (Directory.Exists(manifests))
            {
                Native.waifu_modelmanager_set_bundled_manifests(Utf8.Bytes(manifests));
                Log.Write("info", "library", "bundled manifests: " + manifests);
            }
        }

        /// <summary>This thread's last message, which is where a failed call says why.</summary>
        public static string LastError()
        {
            try
            {
                return Utf8.Read(Native.waifu_last_error()) ?? "no reason given";
            }
            catch (Exception error)
            {
                return error.Message;
            }
        }

        /// <summary>The same for a call into the model manager, which keeps its own.</summary>
        public static string ModelManagerLastError()
        {
            try
            {
                return Utf8.Read(Native.waifu_modelmanager_last_error()) ?? "no reason given";
            }
            catch (Exception error)
            {
                return error.Message;
            }
        }

        internal static Native.Device DeviceNamed(string name)
        {
            switch ((name ?? "auto").Trim().ToLowerInvariant())
            {
                case "cpu": return Native.Device.Cpu;
                // The offload device's names too, which a saved setting may still hold: cuda is
                // what it is now.
                case "cuda":
                case "cuda (offload)":
                case "cuda_cpu_offload": return Native.Device.Cuda;
                case "vulkan": return Native.Device.Vulkan;
                case "metal": return Native.Device.Metal;
                default: return Native.Device.Auto;
            }
        }
    }
}
