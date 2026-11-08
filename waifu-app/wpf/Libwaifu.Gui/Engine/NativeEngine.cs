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
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows.Threading;

namespace Libwaifu.Gui.Engine
{
    /// <summary>
    /// The screen on the library: a pipeline for each task it has run, one job at a time, and the
    /// model read onto the device by the job that needs it.
    /// </summary>
    /// <remarks>
    /// Three rules shape all of this, and they are the library's rather than this program's.
    ///
    /// * **Every callback arrives on a thread of the library's.** Nothing here touches the screen
    ///   from one: a callback copies what it was handed -- the strings, the pixels, the samples,
    ///   all of which stop being valid when it returns -- and hands the copy to the dispatcher.
    /// * **A callback must not call back into the library** (bar <c>waifu_pipeline_cancel</c>).
    ///   So the next step of a job is always taken on the UI thread, after the hand-off, and
    ///   never inside the completion that asked for it.
    /// * **A delegate the library holds has to outlive the call**, which is why the four of them
    ///   are fields and not lambdas written at the call site.
    ///
    /// A pipeline is made for one task, so a picture, a reading and a conversion each run on a
    /// pipeline of their own, all on one device. One model is on the device at a time, as it was
    /// when one engine ran every task: reading a model frees every pipeline that holds another.
    /// A model that does two tasks -- CosyVoice3 reads and converts -- is read into a pipeline
    /// for each, and is on the device twice while both are kept. Changing the device frees every
    /// pipeline, and the next job makes the one it needs on the new one.
    /// </remarks>
    public sealed class NativeEngine : IEngine
    {
        private readonly Dispatcher ui;
        private readonly Native.ProgressCallback onProgress;
        private readonly Native.ResultCallback onLoaded;
        private readonly Native.ImageGenerationCallback onDrawn;
        private readonly Native.AudioCallback onSounded;
        private readonly Stopwatch clock = new Stopwatch();

        /// <summary>The pipelines made so far, one per task, every one of them on <see cref="device"/>.</summary>
        private readonly Dictionary<Native.PipelineType, Pipeline> pipelines =
            new Dictionary<Native.PipelineType, Pipeline>();

        private string device;
        private Pipeline running;
        private ulong job;
        private Fetch fetching;
        private JobRequest current;
        private string reading;
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

        /// <summary>One task's pipeline, and the model read into it, if any.</summary>
        private sealed class Pipeline
        {
            public Native.PipelineType Type;
            public IntPtr Handle;
            public string Model;
        }

        public void Begin(JobRequest request)
        {
            current = request;
            seed = request.Settle();
            clock.Restart();
            started = ResultRecord.Now();

            running = Ready(request.Device, PipelineFor(request.Kind));
            if (running == null)
            {
                End(new JobOutcome { Ending = Ending.Failed, Error = Library.LastError() });
                return;
            }

            if (running.Model == request.Model)
            {
                Work();
                return;
            }

            // waifu.dll reads a model by the path of its manifest. A published one not all here
            // is fetched by the model manager first, and its manifest found there.
            if (NativeModels.NeedsFetching(request.Model))
            {
                FetchThenLoad(request);
                return;
            }

            Load(request);
        }

        public void Cancel()
        {
            if (fetching != null)
            {
                fetching.Cancel();
            }

            if (running != null && running.Handle != IntPtr.Zero && job != 0)
            {
                Native.waifu_pipeline_cancel(running.Handle, job);
            }
        }

        /// <summary>Fetches the model a job needs, and loads it once it is here.</summary>
        private void FetchThenLoad(JobRequest request)
        {
            fetching = Fetch.Start(request.Model);
            if (fetching == null)
            {
                End(new JobOutcome { Ending = Ending.Failed, Error = Library.ModelManagerLastError() });
                return;
            }

            fetching.Progress += (fraction, line) =>
                Say(Stage.Fetching, fraction >= 0 ? (double?)fraction : null, line);
            fetching.Finished += error =>
            {
                fetching = null;
                if (error == null)
                {
                    Load(request);
                }
                else if (error.Length == 0)
                {
                    End(new JobOutcome { Ending = Ending.Cancelled });
                }
                else
                {
                    End(new JobOutcome { Ending = Ending.Failed, Error = error });
                }
            };
        }

        /// <summary>
        /// Reads the model into the pipeline for the job's task, which drops whatever that
        /// pipeline held before it, so there is nothing to unload first.
        /// </summary>
        private void Load(JobRequest request)
        {
            // Freed while a download ran, which only closing the engine does.
            if (running == null || running.Handle == IntPtr.Zero)
            {
                End(new JobOutcome { Ending = Ending.Failed, Error = "the engine was closed while the model was fetched" });
                return;
            }

            var manifest = NativeModels.ManifestPath(request.Model);
            if (manifest == null)
            {
                End(new JobOutcome { Ending = Ending.Failed, Error = Library.ModelManagerLastError() });
                return;
            }

            var model = Native.waifu_model_open(Utf8.Bytes(manifest));
            if (model == IntPtr.Zero)
            {
                End(new JobOutcome { Ending = Ending.Failed, Error = Library.LastError() });
                return;
            }

            try
            {
                // Asked before the load, which would refuse it at the call: the model says why
                // in words to show. SUPPORTS_<task> has the number of the task's pipeline type.
                var supports = (Native.ModelProperty)(int)running.Type;
                if (!NativeModels.Supports(model, supports))
                {
                    End(new JobOutcome
                    {
                        Ending = Ending.Failed,
                        Error = NativeModels.WhyNot(model, supports) ??
                                request.Model + " cannot be used for " + Tasks.Name(request.Kind),
                    });
                    return;
                }

                // One model on the device at a time: pipelines that hold another let it go first.
                Settle(running, request.Model);

                // The load drops what the pipeline held before it reads anything.
                running.Model = null;
                reading = request.Model;
                Say(Stage.Reading, null, "reading " + request.Model);
                job = Native.waifu_pipeline_load_model_async(
                    running.Handle, model, IntPtr.Zero, onProgress, onLoaded);

                if (job == 0)
                {
                    End(new JobOutcome { Ending = Ending.Failed, Error = Library.LastError() });
                }
            }
            finally
            {
                // The load copied the manifest; the weights are read from the files.
                Native.waifu_model_free(model);
            }
        }

        public void Dispose()
        {
            Drop();
        }

        /// <summary>
        /// The pipeline for a task on a device, made the first time one is asked for. A change of
        /// device frees every pipeline first. Null where the library refused, with the reason in
        /// <see cref="Library.LastError"/>.
        /// </summary>
        private Pipeline Ready(string wanted, Native.PipelineType type)
        {
            if (device != wanted)
            {
                Drop();
                device = wanted;
            }

            Pipeline pipeline;
            if (pipelines.TryGetValue(type, out pipeline))
            {
                return pipeline;
            }

            var handle = Native.waifu_pipeline_new(type, Library.DeviceNamed(wanted));
            if (handle == IntPtr.Zero)
            {
                Log.Write("error", "engine", wanted + ": " + Library.LastError());
                return null;
            }

            pipeline = new Pipeline { Type = type, Handle = handle };
            pipelines[type] = pipeline;
            Log.Write("info", "engine", Named(type) + " on " + wanted);
            return pipeline;
        }

        /// <summary>Frees every pipeline but <paramref name="keep"/> that holds a model other than <paramref name="model"/>.</summary>
        private void Settle(Pipeline keep, string model)
        {
            foreach (var other in pipelines.Values.ToList())
            {
                if (other != keep && other.Model != null && other.Model != model)
                {
                    pipelines.Remove(other.Type);
                    Free(other);
                }
            }
        }

        private void Drop()
        {
            var going = pipelines.Values.ToList();
            pipelines.Clear();
            device = null;
            running = null;
            LoadedModel = null;
            LoadedDevice = null;

            foreach (var pipeline in going)
            {
                Free(pipeline);
            }
        }

        private static void Free(Pipeline pipeline)
        {
            if (pipeline.Handle == IntPtr.Zero)
            {
                return;
            }

            // Freeing stops what is running and joins the thread, which can take until the end of
            // the step it is on.
            var handle = pipeline.Handle;
            pipeline.Handle = IntPtr.Zero;
            pipeline.Model = null;
            Native.waifu_pipeline_cancel_all(handle);
            Native.waifu_pipeline_free(handle);
        }

        private static Native.PipelineType PipelineFor(TaskKind kind)
        {
            switch (Tasks.CategoryOf(kind))
            {
                case Category.Image:
                    return Native.PipelineType.ImageGeneration;
                case Category.Speech:
                    return Native.PipelineType.TextToSpeech;
                default:
                    return Native.PipelineType.VoiceConversion;
            }
        }

        private static string Named(Native.PipelineType type)
        {
            switch (type)
            {
                case Native.PipelineType.ImageGeneration: return "image generation";
                case Native.PipelineType.TextToSpeech: return "text to speech";
                case Native.PipelineType.VoiceConversion: return "voice conversion";
                default: return "pipeline " + ((int)type).ToString(CultureInfo.InvariantCulture);
            }
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

            var pinnedPrompt = GCHandle.Alloc(prompt, GCHandleType.Pinned);
            var pinnedNegative = GCHandle.Alloc(negative, GCHandleType.Pinned);
            var pinnedPixels = default(GCHandle);
            var picture = IntPtr.Zero;

            try
            {
                var request = new Native.ImageGenerationRequest
                {
                    StructSize = (uint)Marshal.SizeOf(typeof(Native.ImageGenerationRequest)),
                    Prompt = pinnedPrompt.AddrOfPinnedObject(),
                    Negative = pinnedNegative.AddrOfPinnedObject(),
                    Width = (uint)current.Width,
                    Height = (uint)current.Height,
                    Steps = (uint)current.Steps,
                    Guidance = (float)(current.Guidance ?? 1.0),
                    Seed = (ulong)seed,
                    StartImage = IntPtr.Zero,
                    StartImageStrength = (float)(current.Strength ?? 0.8),
                    ControlImage = IntPtr.Zero,
                    ControlImageScale = 1.0f,
                    PromptImages = IntPtr.Zero,
                    PromptImageCount = UIntPtr.Zero,
                };

                // The picture img2img starts from.
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
                    request.StartImage = picture;
                }

                job = Native.waifu_pipeline_run_image_generation_async(
                    running.Handle, ref request, IntPtr.Zero, onProgress, onDrawn);
            }
            finally
            {
                // Everything above is borrowed only for the length of the call: by the time it
                // has returned the library has copied what it keeps.
                if (picture != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(picture);
                }

                if (pinnedPixels.IsAllocated)
                {
                    pinnedPixels.Free();
                }

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
            var reference = Hold(current.Reference);

            try
            {
                var request = new Native.TextToSpeechRequest
                {
                    StructSize = (uint)Marshal.SizeOf(typeof(Native.TextToSpeechRequest)),
                    Text = pinnedText.AddrOfPinnedObject(),
                    Reference = reference.Pointer,
                    Speed = (float)(current.Speed ?? 1.0),
                    Temperature = (float)(current.Temperature ?? 1.0),
                    Style = pinnedStyle.AddrOfPinnedObject(),
                    Seed = (ulong)seed,
                };

                job = Native.waifu_pipeline_run_text_to_speech_async(
                    running.Handle, ref request, IntPtr.Zero, onProgress, onSounded);
            }
            finally
            {
                reference.Free();
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

                job = Native.waifu_pipeline_run_voice_conversion_async(
                    running.Handle, ref request, IntPtr.Zero, onProgress, onSounded);
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
                SampleRate = (uint)recording.Rate,
            };

            held.Pointer = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(Native.Audio)));
            Marshal.StructureToPtr(audio, held.Pointer, false);
            return held;
        }

        // -- what comes back ----------------------------------------------------------------------

        private void Advanced(IntPtr user, ref Native.ProgressInfo e)
        {
            // On a thread of the library's. Everything is copied before it returns.
            var stage = e.Stage;
            var fraction = e.Fraction >= 0 ? (double?)e.Fraction : null;
            var done = e.Done;
            var total = e.Total;
            var seconds = e.Seconds;

            ui.BeginInvoke((Action)delegate
            {
                Say(Gathered(stage), fraction, Words(stage, done, total, reading), seconds);
            });
        }

        private void Loaded(IntPtr user, ref Native.Result e)
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

                if (running != null)
                {
                    running.Model = model;
                }

                LoadedModel = model;
                LoadedDevice = on;

                // The next call is made here rather than in the callback: nothing may call back
                // into the library from one of its own threads.
                Work();
            });
        }

        private void Drawn(IntPtr user, ref Native.ImageGenerationResult e)
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

        private void Sounded(IntPtr user, ref Native.AudioResult e)
        {
            var code = e.Status.Code;
            var message = Utf8.Read(e.Status.Message);
            float[] samples = null;
            var rate = 0;

            if (code == Native.Status.Ok && e.Audio != IntPtr.Zero)
            {
                var audio = (Native.Audio)Marshal.PtrToStructure(e.Audio, typeof(Native.Audio));
                rate = (int)audio.SampleRate;
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

        /// <summary>
        /// Which of this program's few phases a stage of the library's is. A stage it has no name
        /// for -- a newer library's -- is Working.
        /// </summary>
        private static Stage Gathered(int stage)
        {
            switch (stage)
            {
                case Native.ProgressStage.LoadingWeights:
                    return Stage.Reading;
                case Native.ProgressStage.ImageGenerationEncodingPrompt:
                case Native.ProgressStage.TextToSpeechReadingText:
                    return Stage.Encoding;
                case Native.ProgressStage.ImageGenerationDenoising:
                case Native.ProgressStage.VoiceConversionDenoising:
                    return Stage.Drawing;
                case Native.ProgressStage.ImageGenerationDecodingImage:
                    return Stage.Decoding;
                case Native.ProgressStage.VoiceConversionReadingRecordings:
                    return Stage.Listening;
                case Native.ProgressStage.TextToSpeechGeneratingTokens:
                case Native.ProgressStage.VoiceConversionGeneratingTokens:
                    return Stage.Saying;
                case Native.ProgressStage.TextToSpeechSynthesizingAudio:
                case Native.ProgressStage.VoiceConversionSynthesizingAudio:
                    return Stage.Sounding;
                default:
                    return Stage.Working;
            }
        }

        /// <summary>
        /// The line under the bar for a stage: the library hands over a stage and its counts, and
        /// the words are this program's. One for each stage the three pipelines here can send,
        /// and a plain one for any other.
        /// </summary>
        private static string Words(int stage, ulong done, ulong total, string model)
        {
            switch (stage)
            {
                case Native.ProgressStage.LoadingWeights:
                    return string.IsNullOrEmpty(model) ? "reading" : "reading " + model;

                case Native.ProgressStage.ImageGenerationEncodingPrompt:
                    return "reading the prompt";
                case Native.ProgressStage.ImageGenerationDenoising:
                    return total > 0 ? "step " + Of(done, total) : "drawing";
                case Native.ProgressStage.ImageGenerationDecodingImage:
                    return "making the picture";

                case Native.ProgressStage.TextToSpeechReadingText:
                    return "reading the text";
                // "of about": the total is worked out from the length of the text, and can be
                // passed. A bar that said "of" and then went past it would have been lying.
                case Native.ProgressStage.TextToSpeechGeneratingTokens:
                    return total > 0 ? "token " + OfAbout(done, total) : "saying";
                case Native.ProgressStage.TextToSpeechSynthesizingAudio:
                    return "making the sound";

                case Native.ProgressStage.VoiceConversionReadingRecordings:
                    return "listening to the recordings";
                case Native.ProgressStage.VoiceConversionGeneratingTokens:
                    return total > 0 ? "token " + OfAbout(done, total) : "saying";
                case Native.ProgressStage.VoiceConversionDenoising:
                    return total > 0 ? "step " + Of(done, total) : "drawing";
                // The window being sounded is the one after those already done.
                case Native.ProgressStage.VoiceConversionSynthesizingAudio:
                    return total > 1
                        ? "making the sound, " + Of(Math.Min(done + 1, total), total)
                        : "making the sound";

                default:
                    return "working";
            }
        }

        private static string Of(ulong done, ulong total)
        {
            return done.ToString(CultureInfo.InvariantCulture) + " of " +
                   total.ToString(CultureInfo.InvariantCulture);
        }

        private static string OfAbout(ulong done, ulong total)
        {
            return done.ToString(CultureInfo.InvariantCulture) + " of about " +
                   total.ToString(CultureInfo.InvariantCulture);
        }
    }
}
