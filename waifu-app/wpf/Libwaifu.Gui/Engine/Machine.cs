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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Win32;

namespace Libwaifu.Gui.Engine
{
    /// <summary>What the foot of the sidebar says about this machine.</summary>
    public class MachineSnapshot
    {
        public string CpuName { get; set; }
        public int Cores { get; set; }
        public int Threads { get; set; }
        public ulong MemoryUsed { get; set; }
        public ulong MemoryTotal { get; set; }
        public string GpuName { get; set; }
        public ulong Vram { get; set; }

        /// <summary>How much of the card is in use, where anything says.</summary>
        public ulong VramUsed { get; set; }

        public bool VramKnown { get; set; }

        /// <summary>The card and the processor share one pool, as on Apple silicon.</summary>
        public bool Unified { get; set; }

        /// <summary>
        /// Why there is no bar for the video memory. The mac mostly answers "memory is shared
        /// with the CPU"; here it is that nothing reports how much of a card is in use without
        /// asking the driver, which this screen does not do.
        /// </summary>
        public string WhyNoVram { get; set; }

        public List<string> Accelerators { get; set; }

        public MachineSnapshot()
        {
            Accelerators = new List<string>();
        }
    }

    /// <summary>
    /// Reads the machine, for the lines under "This PC".
    /// </summary>
    /// <remarks>
    /// The real screen asks the library (<c>waifu_machine_json</c>), which answers the same
    /// questions on every platform. This reads them from Windows directly so that the sidebar is
    /// true before the two are joined up. Everything that does not change -- the processor's
    /// name, how many cores it has, which card is in the machine -- is read once; only the memory
    /// is read again every two seconds.
    /// </remarks>
    public static class Machine
    {
        private static MachineSnapshot fixedParts;

        public static MachineSnapshot Read()
        {
            var snapshot = Windows();

            // The library knows two things this side cannot: how much of the card is in use, and
            // which accelerators this build can actually reach. It answers null for whatever it
            // has nothing to say about -- on Windows, today, that is most of the processor and
            // the memory -- so what comes back is laid over what Windows said rather than
            // replacing it.
            if (Library.Available)
            {
                try
                {
                    Lay(NativeModels.Machine(), snapshot);
                }
                catch (Exception error)
                {
                    Log.Write("warn", "machine", error.Message);
                }
            }

            return snapshot;
        }

        private static void Lay(MachineSnapshot said, MachineSnapshot onto)
        {
            if (said == null)
            {
                return;
            }

            if (!string.IsNullOrEmpty(said.CpuName))
            {
                onto.CpuName = said.CpuName;
            }

            if (said.Cores > 0)
            {
                onto.Cores = said.Cores;
            }

            if (said.Threads > 0)
            {
                onto.Threads = said.Threads;
            }

            if (said.MemoryTotal > 0)
            {
                onto.MemoryTotal = said.MemoryTotal;
                onto.MemoryUsed = said.MemoryUsed;
            }

            if (!string.IsNullOrEmpty(said.GpuName))
            {
                onto.GpuName = said.GpuName;
            }

            if (said.Vram > 0)
            {
                onto.Vram = said.Vram;
                onto.VramUsed = said.VramUsed;
                onto.VramKnown = said.VramKnown;
            }

            if (said.Unified)
            {
                onto.Unified = true;
            }

            if (!string.IsNullOrEmpty(said.WhyNoVram))
            {
                onto.WhyNoVram = said.WhyNoVram;
            }

            if (said.Accelerators != null && said.Accelerators.Count > 0)
            {
                onto.Accelerators = said.Accelerators;
            }
        }

        private static MachineSnapshot Windows()
        {
            if (fixedParts == null)
            {
                fixedParts = ReadTheFixedParts();
            }

            var snapshot = new MachineSnapshot
            {
                CpuName = fixedParts.CpuName,
                Cores = fixedParts.Cores,
                Threads = fixedParts.Threads,
                GpuName = fixedParts.GpuName,
                Vram = fixedParts.Vram,
                WhyNoVram = fixedParts.WhyNoVram,
                Accelerators = fixedParts.Accelerators,
            };

            var status = new MEMORYSTATUSEX();
            status.dwLength = (uint)Marshal.SizeOf(typeof(MEMORYSTATUSEX));
            if (GlobalMemoryStatusEx(ref status))
            {
                snapshot.MemoryTotal = status.ullTotalPhys;
                snapshot.MemoryUsed = status.ullTotalPhys - status.ullAvailPhys;
            }

            return snapshot;
        }

        private static MachineSnapshot ReadTheFixedParts()
        {
            var snapshot = new MachineSnapshot
            {
                CpuName = "Unknown processor",
                Threads = Environment.ProcessorCount,
                Cores = Environment.ProcessorCount,
                WhyNoVram = "how much of it is in use is not reported here",
            };

            try
            {
                using (var key = Registry.LocalMachine.OpenSubKey(
                    @"HARDWARE\DESCRIPTION\System\CentralProcessor\0"))
                {
                    if (key != null)
                    {
                        var name = key.GetValue("ProcessorNameString") as string;
                        if (!string.IsNullOrEmpty(name))
                        {
                            snapshot.CpuName = name.Trim();
                        }
                    }
                }
            }
            catch (Exception error)
            {
                Log.Write("warn", "machine", error.Message);
            }

            snapshot.Cores = PhysicalCores(snapshot.Threads);
            ReadTheCard(snapshot);
            snapshot.Accelerators = Accelerators();
            return snapshot;
        }

        /// <summary>
        /// How many cores there are rather than how many threads. Asked of the kernel rather than
        /// of WMI, which would be a second service to start for one number.
        /// </summary>
        private static int PhysicalCores(int threads)
        {
            try
            {
                uint bytes = 0;
                GetLogicalProcessorInformationEx(RelationProcessorCore, IntPtr.Zero, ref bytes);
                if (bytes == 0)
                {
                    return threads;
                }

                var buffer = Marshal.AllocHGlobal((int)bytes);
                try
                {
                    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, ref bytes))
                    {
                        return threads;
                    }

                    // A walk over variable-length records: each one begins with its relationship
                    // and its own size, and one record is one core.
                    var cores = 0;
                    var at = 0;
                    while (at + 8 <= (int)bytes)
                    {
                        var size = Marshal.ReadInt32(buffer, at + 4);
                        if (size <= 0)
                        {
                            break;
                        }

                        cores++;
                        at += size;
                    }

                    return cores > 0 ? cores : threads;
                }
                finally
                {
                    Marshal.FreeHGlobal(buffer);
                }
            }
            catch (Exception error)
            {
                Log.Write("warn", "machine", error.Message);
                return threads;
            }
        }

        /// <summary>
        /// The display adapter and how much memory is on it. The size comes from the driver's own
        /// key because the usual WMI answer is a 32-bit field that stops at four gigabytes, which
        /// every card this program cares about is past.
        /// </summary>
        private static void ReadTheCard(MachineSnapshot snapshot)
        {
            try
            {
                using (var adapters = Registry.LocalMachine.OpenSubKey(
                    @"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"))
                {
                    if (adapters == null)
                    {
                        return;
                    }

                    // A laptop has two adapters and a run goes to the bigger one, so that is the
                    // one the sidebar names: the card with the most memory on it.
                    foreach (var name in adapters.GetSubKeyNames().Where(IsNumbered).OrderBy(each => each))
                    {
                        using (var adapter = adapters.OpenSubKey(name))
                        {
                            if (adapter == null)
                            {
                                continue;
                            }

                            var described = adapter.GetValue("DriverDesc") as string;
                            if (string.IsNullOrEmpty(described))
                            {
                                continue;
                            }

                            var memory = Memory(adapter);
                            if (snapshot.GpuName == null || memory > snapshot.Vram)
                            {
                                snapshot.GpuName = described;
                                snapshot.Vram = memory;
                            }
                        }
                    }
                }
            }
            catch (Exception error)
            {
                Log.Write("warn", "machine", error.Message);
            }
        }

        private static ulong Memory(RegistryKey adapter)
        {
            var size = adapter.GetValue("HardwareInformation.qwMemorySize");
            if (size is long)
            {
                return (ulong)(long)size;
            }

            var older = adapter.GetValue("HardwareInformation.MemorySize");
            if (older is int)
            {
                return (ulong)(uint)(int)older;
            }

            var bytes = older as byte[];
            if (bytes != null && bytes.Length >= 4)
            {
                return BitConverter.ToUInt32(bytes, 0);
            }

            return 0;
        }

        private static bool IsNumbered(string name)
        {
            int ignored;
            return name.Length == 4 && int.TryParse(name, out ignored);
        }

        /// <summary>
        /// What a run could go to besides the processor. Each is named by the one library that
        /// has to be there for it to work at all, which is also what the engine looks for.
        /// </summary>
        private static List<string> Accelerators()
        {
            var found = new List<string>();
            var system = Environment.GetFolderPath(Environment.SpecialFolder.System);

            if (File.Exists(Path.Combine(system, "nvcuda.dll")))
            {
                found.Add("cuda");
            }

            if (File.Exists(Path.Combine(system, "vulkan-1.dll")))
            {
                found.Add("vulkan");
            }

            return found;
        }

        private const uint RelationProcessorCore = 0;

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX status);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetLogicalProcessorInformationEx(
            uint relationship, IntPtr buffer, ref uint length);

        [StructLayout(LayoutKind.Sequential)]
        private struct MEMORYSTATUSEX
        {
            public uint dwLength;
            public uint dwMemoryLoad;
            public ulong ullTotalPhys;
            public ulong ullAvailPhys;
            public ulong ullTotalPageFile;
            public ulong ullAvailPageFile;
            public ulong ullTotalVirtual;
            public ulong ullAvailVirtual;
            public ulong ullAvailExtendedVirtual;
        }
    }
}
