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
using System.Threading;
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
    /// * **A task is read on a thread of its own.** Every load and run hands back a
    ///   <c>WaifuTask</c>, and a background thread blocks in <c>waifu_task_next_event</c> until
    ///   it ends. Nothing there touches the screen: it copies what the task says -- the progress,
    ///   the pixels, the samples -- and hands the copy to the dispatcher.
    /// * **A task is freed once, and never while another call is on it.** The reader frees it
    ///   after END, having copied the result out, under the same lock a cancel takes.
    /// * **The next step of a job is always taken on the UI thread**, after the hand-off, and
    ///   never on the reader's thread: so the library is only ever asked for a new task from
    ///   here, one at a time.
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
        private readonly Stopwatch clock = new Stopwatch();

        /// <summary>The pipelines made so far, one per task, every one of them on <see cref="device"/>.</summary>
        private readonly Dictionary<Native.PipelineType, Pipeline> pipelines =
            new Dictionary<Native.PipelineType, Pipeline>();

        private string device;
        private Pipeline running;

        /// <summary>The task of the load or run under way, or null between them.</summary>
        private TaskReader reader;

        private Fetch fetching;
        private JobRequest current;
        private string reading;
        private long seed;
        private double started;

        public NativeEngine()
        {
            ui = Dispatcher.CurrentDispatcher;
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

            if (reader != null)
            {
                reader.Cancel();
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
                var supports = (Native.ModelAttribute)(int)running.Type;
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
                var loading = Native.waifu_pipeline_load_model(running.Handle, model);
                if (loading == IntPtr.Zero)
                {
                    End(new JobOutcome { Ending = Ending.Failed, Error = Library.LastError() });
                    return;
                }

                var name = request.Model;
                var on = request.Device;
                reader = Follow(loading, null, end => Loaded(end, name, on));
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
            IntPtr task;
            Func<IntPtr, object> output;
            Action<TaskEnd> ended;

            switch (Tasks.CategoryOf(current.Kind))
            {
                case Category.Image:
                    task = Run(Native.PipelineType.ImageGeneration, Draw);
                    output = ReadPicture;
                    ended = Drawn;
                    break;
                case Category.Speech:
                    task = Run(Native.PipelineType.TextToSpeech, Speak);
                    output = ReadSound(Native.TaskAttribute.TextToSpeechAudio);
                    ended = Sounded;
                    break;
                default:
                    task = Run(Native.PipelineType.VoiceConversion, Convert);
                    output = ReadSound(Native.TaskAttribute.VoiceConversionAudio);
                    ended = Sounded;
                    break;
            }

            if (task == IntPtr.Zero)
            {
                End(new JobOutcome { Ending = Ending.Failed, Error = Library.LastError() });
                return;
            }

            reader = Follow(task, output, ended);
        }

        /// <summary>
        /// A request of the pipeline's type, set by <paramref name="set"/> and run. Zero where a
        /// setter or the run refused, with the reason in <see cref="Library.LastError"/>: the
        /// request is freed either way, which leaves the last error as it was.
        /// </summary>
        private IntPtr Run(Native.PipelineType type, Func<IntPtr, bool> set)
        {
            var request = Native.waifu_request_new(type);
            if (request == IntPtr.Zero)
            {
                return IntPtr.Zero;
            }

            try
            {
                // The setters copy what they are handed and the run copies the request, so
                // nothing here has to outlive the call.
                return set(request) ? Native.waifu_pipeline_run(running.Handle, request) : IntPtr.Zero;
            }
            finally
            {
                Native.waifu_request_free(request);
            }
        }

        private bool Draw(IntPtr request)
        {
            if (!SetText(request, Native.RequestAttribute.ImageGenerationPrompt, current.Prompt ?? string.Empty) ||
                !SetText(request, Native.RequestAttribute.ImageGenerationNegative, current.Negative ?? string.Empty) ||
                !SetInt(request, Native.RequestAttribute.ImageGenerationWidth, current.Width) ||
                !SetInt(request, Native.RequestAttribute.ImageGenerationHeight, current.Height) ||
                !SetInt(request, Native.RequestAttribute.ImageGenerationSteps, current.Steps) ||
                !SetFloat(request, Native.RequestAttribute.ImageGenerationGuidance, current.Guidance ?? 1.0) ||
                // Read as a uint64: the same 64 bits the seed always was.
                !SetInt(request, Native.RequestAttribute.ImageGenerationSeed, seed))
            {
                return false;
            }

            // The picture img2img starts from, and how far it walks away from it: the strength
            // means nothing without one, and is left to the library's default.
            if (current.SourceRgb != null)
            {
                return SetImage(request, Native.RequestAttribute.ImageGenerationStartImage,
                           current.SourceRgb, current.SourceWidth, current.SourceHeight) &&
                       SetFloat(request, Native.RequestAttribute.ImageGenerationStartImageStrength,
                           current.Strength ?? 0.8);
            }

            return true;
        }

        private bool Speak(IntPtr request)
        {
            return SetText(request, Native.RequestAttribute.TextToSpeechText, current.Text ?? string.Empty) &&
                   SetAudio(request, Native.RequestAttribute.TextToSpeechReference, current.Reference) &&
                   SetFloat(request, Native.RequestAttribute.TextToSpeechSpeed, current.Speed ?? 1.0) &&
                   SetFloat(request, Native.RequestAttribute.TextToSpeechTemperature, current.Temperature ?? 1.0) &&
                   SetText(request, Native.RequestAttribute.TextToSpeechStyle, current.Style ?? string.Empty) &&
                   SetInt(request, Native.RequestAttribute.TextToSpeechSeed, seed);
        }

        private bool Convert(IntPtr request)
        {
            return SetAudio(request, Native.RequestAttribute.VoiceConversionSource, current.Source) &&
                   SetAudio(request, Native.RequestAttribute.VoiceConversionReference, current.Reference) &&
                   SetInt(request, Native.RequestAttribute.VoiceConversionSteps, current.Steps) &&
                   SetInt(request, Native.RequestAttribute.VoiceConversionConvertStyle,
                       (current.ConvertStyle ?? false) ? 1 : 0) &&
                   SetInt(request, Native.RequestAttribute.VoiceConversionSeed, seed);
        }

        // -- setting a request ---------------------------------------------------------------------

        private static bool SetInt(IntPtr request, Native.RequestAttribute attribute, long value)
        {
            return Native.waifu_request_set_int_attribute(request, attribute, value);
        }

        private static bool SetFloat(IntPtr request, Native.RequestAttribute attribute, double value)
        {
            return Native.waifu_request_set_float_attribute(request, attribute, value);
        }

        private static bool SetText(IntPtr request, Native.RequestAttribute attribute, string value)
        {
            return Native.waifu_request_set_string_attribute(request, attribute, Utf8.Bytes(value));
        }

        /// <summary>An image's pixels, pinned for the length of the call, which copies them.</summary>
        private static bool SetImage(
            IntPtr request, Native.RequestAttribute attribute, byte[] rgb, int width, int height)
        {
            var pinned = GCHandle.Alloc(rgb, GCHandleType.Pinned);
            try
            {
                var image = new Native.Image
                {
                    Width = (uint)width,
                    Height = (uint)height,
                    Rgb = pinned.AddrOfPinnedObject(),
                    Length = (UIntPtr)(uint)rgb.Length,
                };

                return Native.waifu_request_set_image_attribute(request, attribute, ref image);
            }
            finally
            {
                pinned.Free();
            }
        }

        /// <summary>
        /// A recording, pinned for the length of the call, which copies it. One that is missing or
        /// empty leaves the attribute unset, as a null pointer always did.
        /// </summary>
        private static bool SetAudio(IntPtr request, Native.RequestAttribute attribute, Recording recording)
        {
            if (recording == null || recording.Samples == null || recording.Samples.Length == 0)
            {
                return true;
            }

            var pinned = GCHandle.Alloc(recording.Samples, GCHandleType.Pinned);
            try
            {
                var audio = new Native.Audio
                {
                    Samples = pinned.AddrOfPinnedObject(),
                    Count = (UIntPtr)(uint)recording.Samples.Length,
                    SampleRate = (uint)recording.Rate,
                };

                return Native.waifu_request_set_audio_attribute(request, attribute, ref audio);
            }
            finally
            {
                pinned.Free();
            }
        }

        // -- reading a task -----------------------------------------------------------------------

        /// <summary>
        /// A task of the library's, until its reader has read it to the end and freed it. A
        /// cancel and the free both take the lock: a task must not be freed while another call
        /// is on it.
        /// </summary>
        private sealed class TaskReader
        {
            private readonly object gate = new object();
            private IntPtr task;

            public TaskReader(IntPtr task)
            {
                this.task = task;
            }

            /// <summary>Takes effect after the step it is on; nothing once the task has been freed.</summary>
            public void Cancel()
            {
                lock (gate)
                {
                    if (task != IntPtr.Zero)
                    {
                        Native.waifu_task_cancel(task);
                    }
                }
            }

            /// <summary>Called by the reader alone, once it has copied out what it needs.</summary>
            public void Free()
            {
                lock (gate)
                {
                    if (task != IntPtr.Zero)
                    {
                        Native.waifu_task_free(task);
                        task = IntPtr.Zero;
                    }
                }
            }
        }

        /// <summary>How a task ended, copied out of it before it was freed.</summary>
        private sealed class TaskEnd
        {
            public Native.Status Code;
            public string Message;

            /// <summary>What the task's output reader copied, for WAIFU_OK; null otherwise.</summary>
            public object Output;

            /// <summary>The clock when the end was read, as each completion used to take it.</summary>
            public double Seconds;
        }

        private sealed class CopiedPicture
        {
            public int Width;
            public int Height;
            public byte[] Rgb;
        }

        private sealed class CopiedSound
        {
            public float[] Samples;
            public int Rate;
        }

        /// <summary>
        /// Starts a background thread that reads <paramref name="task"/> to its end, hands its
        /// progress to the dispatcher as it comes, copies its output with
        /// <paramref name="output"/> (on that thread, for WAIFU_OK), frees it, and hands
        /// <paramref name="ended"/> what it came to on the UI thread.
        /// </summary>
        private TaskReader Follow(IntPtr task, Func<IntPtr, object> output, Action<TaskEnd> ended)
        {
            var following = new TaskReader(task);
            var thread = new Thread(() => Read(following, task, output, ended))
            {
                IsBackground = true,
                Name = "waifu task",
            };

            try
            {
                thread.Start();
            }
            catch (Exception error)
            {
                // Without a reader the job would never be told its end: it is told now, and the
                // task let go of.
                following.Cancel();
                following.Free();
                var end = new TaskEnd
                {
                    Code = Native.Status.Internal,
                    Message = "could not start a thread to wait on the job: " + error.Message,
                    Seconds = clock.Elapsed.TotalSeconds,
                };
                ui.BeginInvoke((Action)delegate { ended(end); });
            }

            return following;
        }

        /// <summary>The reader's thread: nothing here touches the screen.</summary>
        private void Read(TaskReader following, IntPtr task, Func<IntPtr, object> output, Action<TaskEnd> ended)
        {
            var end = new TaskEnd();
            try
            {
                while (true)
                {
                    var happened = Native.waifu_task_next_event(task, Native.WaitForever);
                    if (happened == Native.EventType.End)
                    {
                        break;
                    }

                    if (happened == Native.EventType.Progress)
                    {
                        Advanced(task);
                    }

                    // NONE never comes without a timeout; anything else is an event this program
                    // did not ask for, and is let go of.
                }

                long code;
                if (Native.waifu_task_get_int_attribute(task, Native.TaskAttribute.Status, out code))
                {
                    end.Code = (Native.Status)(int)code;
                    end.Message = Utf8.Read(
                        Native.waifu_task_get_string_attribute(task, Native.TaskAttribute.StatusMessage));
                }
                else
                {
                    end.Code = Native.Status.Internal;
                    end.Message = "the library said the job had ended, and had no status for it";
                }

                // Everything the task hands out is valid until it is freed, so it is copied now.
                if (end.Code == Native.Status.Ok && output != null)
                {
                    end.Output = output(task);
                }
            }
            catch (Exception error)
            {
                // This thread is the program's own, and an exception that left it would take the
                // process down with it.
                end.Code = Native.Status.Internal;
                end.Message = error.Message;
                end.Output = null;
            }
            finally
            {
                following.Free();
            }

            end.Seconds = clock.Elapsed.TotalSeconds;
            ui.BeginInvoke((Action)delegate { ended(end); });
        }

        /// <summary>What the last PROGRESS event said, copied and handed to the dispatcher.</summary>
        private void Advanced(IntPtr task)
        {
            long stage;
            if (!Native.waifu_task_get_int_attribute(task, Native.TaskAttribute.ProgressStage, out stage))
            {
                return;
            }

            long done, total;
            double fraction, seconds;
            var at = (int)stage;
            var doneSoFar = Native.waifu_task_get_int_attribute(task, Native.TaskAttribute.ProgressDone, out done)
                ? (ulong)done
                : 0UL;
            var ofTotal = Native.waifu_task_get_int_attribute(task, Native.TaskAttribute.ProgressTotal, out total)
                ? (ulong)total
                : 0UL;
            var part = Native.waifu_task_get_float_attribute(task, Native.TaskAttribute.ProgressFraction, out fraction) &&
                       fraction >= 0
                ? (double?)fraction
                : null;
            var since = Native.waifu_task_get_float_attribute(task, Native.TaskAttribute.ProgressSeconds, out seconds)
                ? (double?)seconds
                : null;

            ui.BeginInvoke((Action)delegate
            {
                Say(Gathered(at), part, Words(at, doneSoFar, ofTotal, reading), since);
            });
        }

        /// <summary>The picture a run drew, copied out of the task; null where it handed over none.</summary>
        private static object ReadPicture(IntPtr task)
        {
            Native.Image image;
            if (!Native.waifu_task_get_image_attribute(task, Native.TaskAttribute.ImageGenerationImage, out image))
            {
                return null;
            }

            var rgb = new byte[(int)image.Length.ToUInt64()];
            if (rgb.Length > 0)
            {
                Marshal.Copy(image.Rgb, rgb, 0, rgb.Length);
            }

            return new CopiedPicture { Width = (int)image.Width, Height = (int)image.Height, Rgb = rgb };
        }

        /// <summary>The recording a run made, copied out of the task; null where it handed over none.</summary>
        private static Func<IntPtr, object> ReadSound(Native.TaskAttribute attribute)
        {
            return task =>
            {
                Native.Audio audio;
                if (!Native.waifu_task_get_audio_attribute(task, attribute, out audio))
                {
                    return null;
                }

                var samples = new float[(int)audio.Count.ToUInt64()];
                if (samples.Length > 0)
                {
                    Marshal.Copy(audio.Samples, samples, 0, samples.Length);
                }

                return new CopiedSound { Samples = samples, Rate = (int)audio.SampleRate };
            };
        }

        // -- what comes back ----------------------------------------------------------------------

        private void Loaded(TaskEnd end, string model, string on)
        {
            if (end.Code != Native.Status.Ok)
            {
                End(Ended(end.Code, end.Message));
                return;
            }

            if (running != null)
            {
                running.Model = model;
            }

            LoadedModel = model;
            LoadedDevice = on;

            // The next step is taken here, on the UI thread, and never on the reader's.
            Work();
        }

        private void Drawn(TaskEnd end)
        {
            var picture = end.Output as CopiedPicture;
            if (end.Code != Native.Status.Ok || picture == null)
            {
                End(Ended(end.Code, end.Message));
                return;
            }

            try
            {
                var record = Output.Picture(
                    current, seed, started, end.Seconds, picture.Width, picture.Height, picture.Rgb);
                End(new JobOutcome { Ending = Ending.Done, Result = record });
            }
            catch (Exception error)
            {
                Log.Write("error", "engine", error.Message);
                End(new JobOutcome { Ending = Ending.Failed, Error = error.Message });
            }
        }

        private void Sounded(TaskEnd end)
        {
            var sound = end.Output as CopiedSound;
            if (end.Code != Native.Status.Ok || sound == null)
            {
                End(Ended(end.Code, end.Message));
                return;
            }

            try
            {
                var record = Output.Sound(
                    current, Tasks.CategoryOf(current.Kind), seed, started, end.Seconds,
                    sound.Samples, sound.Rate);
                End(new JobOutcome { Ending = Ending.Done, Result = record });
            }
            catch (Exception error)
            {
                Log.Write("error", "engine", error.Message);
                End(new JobOutcome { Ending = Ending.Failed, Error = error.Message });
            }
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
            reader = null;
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
