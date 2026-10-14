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
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Threading;

namespace Libwaifu.Gui.Engine
{
    /// <summary>
    /// The screen on the library: one engine, one job at a time, and the model read onto the
    /// device by the job that needs it.
    /// </summary>
    /// <remarks>
    /// Three rules shape all of this, and they are the library's rather than this program's.
    ///
    /// * **Every callback arrives on a thread of the library's.** Nothing here touches the screen
    ///   from one: a callback copies what it was handed -- the strings, the pixels, the samples,
    ///   all of which stop being valid when it returns -- and hands the copy to the dispatcher.
    /// * **A callback must not call back into the library** (cancel and the free functions
    ///   aside). So the next step of a job is always taken on the UI thread, after the hand-off,
    ///   and never inside the completion that asked for it.
    /// * **A delegate the library holds has to outlive the call**, which is why the four of them
    ///   are fields and not lambdas written at the call site.
    ///
    /// One engine at a time, for one device: two engines are two models on one card. Changing
    /// the device frees the engine and makes another.
    /// </remarks>
    public sealed class NativeEngine : IEngine
    {
        private readonly Dispatcher ui;
        private readonly Native.ProgressCallback onProgress;
        private readonly Native.CompleteCallback onLoaded;
        private readonly Native.ImageCompleteCallback onDrawn;
        private readonly Native.AudioCompleteCallback onSounded;
        private readonly Stopwatch clock = new Stopwatch();

        private IntPtr engine;
        private string device;
        private ulong job;
        private JobRequest current;
        private long seed;
        private double started;

        public NativeEngine()
        {
            ui = Dispatcher.CurrentDispatcher;
            onProgress = Advanced;
            onLoaded = Loaded;
            onDrawn = Drawn;
            onSounded = Sounded;
        }

        public string LoadedModel { get; private set; }

        public string LoadedDevice { get; private set; }

        public event Action<EngineEvent> Progress;

        public event Action<JobOutcome> Finished;

        public void Begin(JobRequest request)
        {
            current = request;
            seed = request.Settle();
            clock.Restart();
            started = ResultRecord.Now();

            if (!Ready(request.Device))
            {
                End(new JobOutcome { Ending = Ending.Failed, Error = Library.LastError() });
                return;
            }

            if (LoadedModel == request.Model && LoadedDevice == request.Device)
            {
                Work();
                return;
            }

            // Loading fetches what is not here and drops whatever was loaded before it, so there
            // is nothing to unload first.
            Say(Stage.Reading, null, "reading " + request.Model);
            job = Native.waifu_engine_load_async(
                engine, Utf8.Bytes(request.Model), IntPtr.Zero, onProgress, onLoaded);

            if (job == 0)
            {
                End(new JobOutcome { Ending = Ending.Failed, Error = Library.LastError() });
            }
        }

        public void Cancel()
        {
            if (engine != IntPtr.Zero && job != 0)
            {
                Native.waifu_engine_cancel(engine, job);
            }
        }

        public void Dispose()
        {
            Drop();
        }

        /// <summary>
        /// The engine for a device, made the first time one is asked for and again whenever the
        /// device changes.
        /// </summary>
        private bool Ready(string wanted)
        {
            if (engine != IntPtr.Zero && device == wanted)
            {
                return true;
            }

            Drop();
            engine = Native.waifu_engine_new(Library.DeviceNamed(wanted));
            if (engine == IntPtr.Zero)
            {
                Log.Write("error", "engine", wanted + ": " + Library.LastError());
                return false;
            }

            device = wanted;
            Log.Write("info", "engine", "engine on " + wanted);
            return true;
        }

        private void Drop()
        {
            if (engine == IntPtr.Zero)
            {
                return;
            }

            // Freeing stops what is running and joins the thread, which can take until the end of
            // the step it is on.
            var going = engine;
            engine = IntPtr.Zero;
            device = null;
            LoadedModel = null;
            LoadedDevice = null;
            Native.waifu_engine_cancel_all(going);
            Native.waifu_engine_free(going);
        }

        // -- the job itself ----------------------------------------------------------------------

        private void Work()
        {
            switch (Tasks.CategoryOf(current.Kind))
            {
                case Category.Image:
                    Draw();
                    break;
                case Category.Speech:
                    Speak();
                    break;
                default:
                    Convert();
                    break;
            }

            if (job == 0)
            {
                End(new JobOutcome { Ending = Ending.Failed, Error = Library.LastError() });
            }
        }

        private void Draw()
        {
            var prompt = Utf8.Bytes(current.Prompt ?? string.Empty);
            var negative = Utf8.Bytes(current.Negative ?? string.Empty);
            var key = Utf8.Bytes("start_from");

            var pinnedPrompt = GCHandle.Alloc(prompt, GCHandleType.Pinned);
            var pinnedNegative = GCHandle.Alloc(negative, GCHandleType.Pinned);
            var pinnedKey = GCHandle.Alloc(key, GCHandleType.Pinned);
            var pinnedPixels = default(GCHandle);
            var images = IntPtr.Zero;
            var picture = IntPtr.Zero;

            try
            {
                var request = new Native.DrawRequest
                {
                    StructSize = (uint)Marshal.SizeOf(typeof(Native.DrawRequest)),
                    Prompt = pinnedPrompt.AddrOfPinnedObject(),
                    Negative = pinnedNegative.AddrOfPinnedObject(),
                    Width = (uint)current.Width,
                    Height = (uint)current.Height,
                    Steps = (uint)current.Steps,
                    Guidance = (float)(current.Guidance ?? 1.0),
                    Seed = (ulong)seed,
                    Strength = (float)(current.Strength ?? 0.8),
                    ControlScale = 1.0f,
                };

                // The picture img2img starts from, under the key the engine keeps for it.
                if (current.SourceRgb != null)
                {
                    pinnedPixels = GCHandle.Alloc(current.SourceRgb, GCHandleType.Pinned);
                    var image = new Native.Image
                    {
                        Width = (uint)current.SourceWidth,
                        Height = (uint)current.SourceHeight,
                        Rgb = pinnedPixels.AddrOfPinnedObject(),
                        Length = (UIntPtr)(uint)current.SourceRgb.Length,
                    };

                    picture = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(Native.Image)));
                    Marshal.StructureToPtr(image, picture, false);

                    var input = new Native.ImageInput
                    {
                        Key = pinnedKey.AddrOfPinnedObject(),
                        Image = picture,
                    };

                    images = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(Native.ImageInput)));
                    Marshal.StructureToPtr(input, images, false);

                    request.Images = images;
                    request.ImageCount = (UIntPtr)1;
                }

                job = Native.waifu_engine_draw_async(
                    engine, ref request, IntPtr.Zero, onProgress, onDrawn);
            }
            finally
            {
                // Everything above is borrowed only for the length of the call: by the time it
                // has returned the library has copied what it keeps.
                if (images != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(images);
                }

                if (picture != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(picture);
                }

                if (pinnedPixels.IsAllocated)
                {
                    pinnedPixels.Free();
                }

                pinnedKey.Free();
                pinnedNegative.Free();
                pinnedPrompt.Free();
            }
        }

        private void Speak()
        {
            var text = Utf8.Bytes(current.Text ?? string.Empty);
            var style = Utf8.Bytes(current.Style ?? string.Empty);
            var pinnedText = GCHandle.Alloc(text, GCHandleType.Pinned);
            var pinnedStyle = GCHandle.Alloc(style, GCHandleType.Pinned);
            var like = Hold(current.Reference);

            try
            {
                var request = new Native.SpeakRequest
                {
                    StructSize = (uint)Marshal.SizeOf(typeof(Native.SpeakRequest)),
                    Text = pinnedText.AddrOfPinnedObject(),
                    Like = like.Pointer,
                    Speed = (float)(current.Speed ?? 1.0),
                    Temperature = (float)(current.Temperature ?? 1.0),
                    Style = pinnedStyle.AddrOfPinnedObject(),
                    Seed = (ulong)seed,
                };

                job = Native.waifu_engine_speak_async(
                    engine, ref request, IntPtr.Zero, onProgress, onSounded);
            }
            finally
            {
                like.Free();
                pinnedStyle.Free();
                pinnedText.Free();
            }
        }

        private void Convert()
        {
            var source = Hold(current.Source);
            var reference = Hold(current.Reference);

            try
            {
                var request = new Native.VoiceConversionRequest
                {
                    StructSize = (uint)Marshal.SizeOf(typeof(Native.VoiceConversionRequest)),
                    Source = source.Pointer,
                    Reference = reference.Pointer,
                    Steps = (uint)current.Steps,
                    ConvertStyle = current.ConvertStyle ?? false,
                    Seed = (ulong)seed,
                };

                job = Native.waifu_engine_voice_conversion_async(
                    engine, ref request, IntPtr.Zero, onProgress, onSounded);
            }
            finally
            {
                reference.Free();
                source.Free();
            }
        }

        /// <summary>A recording, pinned and wrapped, for the length of one call.</summary>
        private struct Held
        {
            public GCHandle Pinned;
            public IntPtr Pointer;

            public void Free()
            {
                if (Pointer != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(Pointer);
                }

                if (Pinned.IsAllocated)
                {
                    Pinned.Free();
                }
            }
        }

        private static Held Hold(Recording recording)
        {
            if (recording == null || recording.Samples == null || recording.Samples.Length == 0)
            {
                return new Held();
            }

            var held = new Held
            {
                Pinned = GCHandle.Alloc(recording.Samples, GCHandleType.Pinned),
            };

            var audio = new Native.Audio
            {
                Samples = held.Pinned.AddrOfPinnedObject(),
                Count = (UIntPtr)(uint)recording.Samples.Length,
                Rate = (uint)recording.Rate,
            };

            held.Pointer = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(Native.Audio)));
            Marshal.StructureToPtr(audio, held.Pointer, false);
            return held;
        }

        // -- what comes back ----------------------------------------------------------------------

        private void Advanced(IntPtr user, ref Native.ProgressEvent e)
        {
            // On a thread of the library's. Everything is copied before it returns.
            var stage = (Stage)e.Stage;
            var fraction = e.Fraction >= 0 ? (double?)e.Fraction : null;
            var seconds = e.Seconds;
            var words = Utf8.Read(e.Words);
            var file = Utf8.Read(e.File);
            var line = words ?? file ?? Words(stage);

            ui.BeginInvoke((Action)delegate { Say(stage, fraction, line, seconds); });
        }

        private void Loaded(IntPtr user, ref Native.CompletionEvent e)
        {
            var code = e.Status.Code;
            var message = Utf8.Read(e.Status.Message);
            var model = current == null ? null : current.Model;
            var on = current == null ? null : current.Device;

            ui.BeginInvoke((Action)delegate
            {
                if (code != Native.Status.Ok)
                {
                    End(Ended(code, message));
                    return;
                }

                LoadedModel = model;
                LoadedDevice = on;

                // The next call is made here rather than in the callback: nothing may call back
                // into the library from one of its own threads.
                Work();
            });
        }

        private void Drawn(IntPtr user, ref Native.ImageCompletionEvent e)
        {
            var code = e.Status.Code;
            var message = Utf8.Read(e.Status.Message);
            byte[] rgb = null;
            var width = 0;
            var height = 0;

            if (code == Native.Status.Ok && e.Image != IntPtr.Zero)
            {
                // Valid only until this returns, so it is copied now rather than remembered.
                var image = (Native.Image)Marshal.PtrToStructure(e.Image, typeof(Native.Image));
                width = (int)image.Width;
                height = (int)image.Height;
                rgb = new byte[(int)image.Length];
                Marshal.Copy(image.Rgb, rgb, 0, rgb.Length);
            }

            var seconds = clock.Elapsed.TotalSeconds;
            ui.BeginInvoke((Action)delegate
            {
                if (code != Native.Status.Ok || rgb == null)
                {
                    End(Ended(code, message));
                    return;
                }

                try
                {
                    var record = Output.Picture(current, seed, started, seconds, width, height, rgb);
                    End(new JobOutcome { Ending = Ending.Done, Result = record });
                }
                catch (Exception error)
                {
                    Log.Write("error", "engine", error.Message);
                    End(new JobOutcome { Ending = Ending.Failed, Error = error.Message });
                }
            });
        }

        private void Sounded(IntPtr user, ref Native.AudioCompletionEvent e)
        {
            var code = e.Status.Code;
            var message = Utf8.Read(e.Status.Message);
            float[] samples = null;
            var rate = 0;

            if (code == Native.Status.Ok && e.Audio != IntPtr.Zero)
            {
                var audio = (Native.Audio)Marshal.PtrToStructure(e.Audio, typeof(Native.Audio));
                rate = (int)audio.Rate;
                samples = new float[(int)audio.Count];
                Marshal.Copy(audio.Samples, samples, 0, samples.Length);
            }

            var seconds = clock.Elapsed.TotalSeconds;
            var category = Tasks.CategoryOf(current.Kind);

            ui.BeginInvoke((Action)delegate
            {
                if (code != Native.Status.Ok || samples == null)
                {
                    End(Ended(code, message));
                    return;
                }

                try
                {
                    var record = Output.Sound(
                        current, category, seed, started, seconds, samples, rate);
                    End(new JobOutcome { Ending = Ending.Done, Result = record });
                }
                catch (Exception error)
                {
                    Log.Write("error", "engine", error.Message);
                    End(new JobOutcome { Ending = Ending.Failed, Error = error.Message });
                }
            });
        }

        private JobOutcome Ended(Native.Status code, string message)
        {
            if (code == Native.Status.Cancelled)
            {
                return new JobOutcome { Ending = Ending.Cancelled };
            }

            return new JobOutcome
            {
                Ending = Ending.Failed,
                Error = message ?? "the job failed with no reason given",
            };
        }

        private void End(JobOutcome outcome)
        {
            job = 0;
            clock.Stop();

            var handler = Finished;
            if (handler != null)
            {
                handler(outcome);
            }
        }

        private void Say(Stage stage, double? fraction, string line, double? seconds = null)
        {
            var handler = Progress;
            if (handler != null)
            {
                handler(new EngineEvent(
                    stage, fraction, seconds ?? clock.Elapsed.TotalSeconds, line));
            }
        }

        /// <summary>What to say for a stage that sent no words of its own.</summary>
        private static string Words(Stage stage)
        {
            switch (stage)
            {
                case Stage.Fetching: return "fetching";
                case Stage.Reading: return "reading";
                case Stage.Encoding: return "encoding";
                case Stage.Drawing: return "drawing";
                case Stage.Decoding: return "decoding";
                case Stage.Listening: return "listening";
                case Stage.Saying: return "saying";
                default: return "sounding";
            }
        }
    }
}
