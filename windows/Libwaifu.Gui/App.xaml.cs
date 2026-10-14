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
using System.Windows;
using System.Windows.Media;
using Libwaifu.Gui.Engine;
using Microsoft.Win32;

namespace Libwaifu.Gui
{
    public partial class App : Application
    {
        /// <summary>
        /// What was asked for on the command line: a model to start on, a device, and an output
        /// folder for this run only.
        /// </summary>
        public static string AskedForModel { get; private set; }

        public static string AskedForDevice { get; private set; }

        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            // The log callback is registered before anything else the library would write to it.
            Log.Write("info", "app", "libwaifu-gui starting");

            ReadTheCommandLine(e.Args);
            TakeTheAccentColour();
        }

        private static void ReadTheCommandLine(string[] args)
        {
            for (var i = 0; i < args.Length; i++)
            {
                var flag = args[i].TrimStart('-').ToLowerInvariant();
                var value = i + 1 < args.Length ? args[i + 1] : null;
                if (value == null)
                {
                    continue;
                }

                if (flag == "m" || flag == "model")
                {
                    AskedForModel = value;
                    i++;
                }
                else if (flag == "device")
                {
                    AskedForDevice = value;
                    i++;
                }
                else if (flag == "output")
                {
                    // This run only, and not written to the settings: that is what the flag means.
                    Places.Overridden = value;
                    i++;
                }
            }
        }

        /// <summary>
        /// The one colour on the screen that is not fixed. The mac takes the system accent; this
        /// takes the Windows one, and falls back to the blue the mac defaults to.
        /// </summary>
        private void TakeTheAccentColour()
        {
            try
            {
                using (var key = Registry.CurrentUser.OpenSubKey(@"Software\Microsoft\Windows\DWM"))
                {
                    if (key == null)
                    {
                        return;
                    }

                    var stored = key.GetValue("AccentColor");
                    if (!(stored is int))
                    {
                        return;
                    }

                    // DWM keeps it as ABGR, which is the usual order with the ends swapped.
                    var value = unchecked((uint)(int)stored);
                    var colour = Color.FromRgb(
                        (byte)(value & 0xFF),
                        (byte)((value >> 8) & 0xFF),
                        (byte)((value >> 16) & 0xFF));

                    Resources["AccentColour"] = colour;
                    Resources["AccentDimColour"] =
                        Color.FromArgb(0x33, colour.R, colour.G, colour.B);
                }
            }
            catch (Exception error)
            {
                Log.Write("warn", "app", "accent colour: " + error.Message);
            }
        }
    }
}
