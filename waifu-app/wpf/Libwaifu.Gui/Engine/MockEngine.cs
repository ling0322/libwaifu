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
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace Libwaifu.Gui.Engine
{
    /// <summary>
    /// An engine that goes through the motions without a model: it fetches, reads, draws its
    /// steps and writes a picture, taking about as long over each as the real one does.
    /// </summary>
    /// <remarks>
    /// This exists so that every state the screen has -- a bar with no fraction while a model is
    /// fetched, a step count while one is drawn, a result in the strip, a run stopped halfway --
    /// can be seen and worked on before <c>waifu.dll</c> is wired up behind the same interface.
    /// The one thing it does not copy is where the callbacks come from: the real engine calls
    /// back on its own threads, and that implementation has to hand each event to the dispatcher
    /// rather than, as here, already being on it.
    /// </remarks>
    public class MockEngine : IEngine
    {
        private readonly DispatcherTimer tick;
        private readonly Stopwatch clock = new Stopwatch();

        private JobRequest job;
        private long seed;
        private Phase phase;
        private int step;
        private int steps;
        private double phaseStarted;
        private bool stopping;

        private enum Phase
        {
            Idle,
            Unloading,
            Fetching,
            Reading,
            Working,
        }

        public MockEngine()
        {
            tick = new DispatcherTimer(DispatcherPriority.Background)
            {
                Interval = TimeSpan.FromMilliseconds(80),
            };
            tick.Tick += OnTick;
        }

        public string LoadedModel { get; private set; }

        public string LoadedDevice { get; private set; }

        public event Action<EngineEvent> Progress;

        public event Action<JobOutcome> Finished;

        public void Begin(JobRequest request)
        {
            job = request;
            seed = request.Settle();
            stopping = false;
            step = 0;
            steps = Math.Max(1, request.Steps);
            clock.Restart();
            phaseStarted = 0;

            var entry = Models.Find(request.Model);
            var loaded = LoadedModel == request.Model && LoadedDevice == request.Device;

            if (loaded)
            {
                Enter(Phase.Working);
            }
            else if (LoadedModel != null)
            {
                Enter(Phase.Unloading);
            }
            else if (entry != null && !entry.Cached)
            {
                Enter(Phase.Fetching);
            }
            else
            {
                Enter(Phase.Reading);
            }

            Log.Write("info", "engine", "job " + Tasks.Name(request.Kind) + " on " + request.Model);
            tick.Start();
        }

        public void Cancel()
        {
            if (phase == Phase.Idle)
            {
                return;
            }

            stopping = true;
        }

        public void Dispose()
        {
            tick.Stop();
            tick.Tick -= OnTick;
        }

        private void Enter(Phase next)
        {
            phase = next;
            phaseStarted = clock.Elapsed.TotalSeconds;
        }

        private double Since
        {
            get { return clock.Elapsed.TotalSeconds - phaseStarted; }
        }

        private void OnTick(object sender, EventArgs e)
        {
            // Fetching stops between files and the steps stop after the one they are on; reading
            // a model onto a device cannot be stopped partway, so that phase runs to its end.
            if (stopping && phase != Phase.Reading)
            {
                Stop(new JobOutcome { Ending = Ending.Cancelled });
                return;
            }

            switch (phase)
            {
                case Phase.Unloading:
                    Say(Stage.Reading, null, "unloading " + LoadedModel);
                    if (Since > 0.6)
                    {
                        var old = Models.Find(job.Model);
                        LoadedModel = null;
                        LoadedDevice = null;
                        Enter(old != null && !old.Cached ? Phase.Fetching : Phase.Reading);
                    }

                    break;

                case Phase.Fetching:
                    Fetching();
                    break;

                case Phase.Reading:
                    Say(Stage.Reading, null, "reading " + job.Model);
                    if (Since > 1.8)
                    {
                        LoadedModel = job.Model;
                        LoadedDevice = job.Device;
                        var entry = Models.Find(job.Model);
                        if (entry != null)
                        {
                            entry.Cached = true;
                            entry.BytesOnDisk = entry.BytesTotal;
                        }

                        Enter(Phase.Working);
                    }

                    break;

                case Phase.Working:
                    Working();
                    break;
            }
        }

        private void Fetching()
        {
            var entry = Models.Find(job.Model);
            var packages = 5;
            var at = Math.Min(packages, 1 + (int)(Since / 0.7));
            Say(Stage.Fetching, null,
                "fetching " + job.Model + " (" + at + " of " + packages + ")");

            if (entry != null)
            {
                entry.BytesOnDisk = (long)(entry.BytesTotal * Math.Min(1.0, Since / (packages * 0.7)));
            }

            if (Since > packages * 0.7)
            {
                Enter(Phase.Reading);
            }
        }

        private void Working()
        {
            var category = Tasks.CategoryOf(job.Kind);
            var each = category == Category.Image ? 0.14 : 0.09;
            var next = Math.Min(steps, (int)(Since / each) + 1);

            if (next != step)
            {
                step = next;
            }

            var stage = category == Category.Image
                ? Stage.Drawing
                : (category == Category.Speech ? Stage.Saying : Stage.Sounding);

            Say(stage, (double)step / steps, "step " + step + " of " + steps);

            if (Since > steps * each)
            {
                Deliver();
            }
        }

        private void Deliver()
        {
            try
            {
                var category = Tasks.CategoryOf(job.Kind);
                var seconds = clock.Elapsed.TotalSeconds;
                var started = ResultRecord.Now() - clock.Elapsed.TotalMilliseconds;

                // The same two calls the real engine's completions make: what a job produces is
                // written in one place whichever engine produced it.
                ResultRecord record;
                if (category == Category.Image)
                {
                    Say(Stage.Decoding, 1.0, "decoding");
                    int width, height;
                    var rgb = Picture(out width, out height);
                    record = Output.Picture(job, seed, started, seconds, width, height, rgb);
                }
                else
                {
                    record = Output.Sound(
                        job, category, seed, started, seconds, Samples(category), Rate);
                }

                Stop(new JobOutcome { Ending = Ending.Done, Result = record });
            }
            catch (Exception error)
            {
                Log.Write("error", "engine", error.Message);
                Stop(new JobOutcome { Ending = Ending.Failed, Error = error.Message });
            }
        }

        private void Stop(JobOutcome outcome)
        {
            tick.Stop();
            clock.Stop();
            phase = Phase.Idle;
            stopping = false;

            var handler = Finished;
            if (handler != null)
            {
                handler(outcome);
            }
        }

        private void Say(Stage stage, double? fraction, string line)
        {
            var handler = Progress;
            if (handler != null)
            {
                handler(new EngineEvent(stage, fraction, clock.Elapsed.TotalSeconds, line));
            }
        }

        // -- what comes out ---------------------------------------------------------------------

        private const int Rate = 24000;

        /// <summary>
        /// A picture, of a sort: bands of colour settled by the seed, so that the same seed gives
        /// the same picture back and two runs are told apart at a glance in the strip. Handed
        /// over as rows of RGB, which is the shape the real engine hands its pixels over in.
        /// </summary>
        private byte[] Picture(out int wide, out int high)
        {
            var width = Math.Max(64, job.Width);
            var height = Math.Max(64, job.Height);
            var random = new Random((int)(seed & 0x7FFFFFFF));

            var hue = random.NextDouble() * 360;
            var visual = new DrawingVisual();
            using (var draw = visual.RenderOpen())
            {
                var sky = new LinearGradientBrush(
                    FromHue(hue, 0.55, 0.30), FromHue(hue + 40, 0.65, 0.62), 90.0);
                draw.DrawRectangle(sky, null, new Rect(0, 0, width, height));

                for (var i = 0; i < 7; i++)
                {
                    var radius = (0.1 + random.NextDouble() * 0.35) * Math.Min(width, height);
                    var centre = new Point(random.NextDouble() * width, random.NextDouble() * height);
                    var blob = new SolidColorBrush(FromHue(hue + 120 + random.NextDouble() * 120, 0.6, 0.55))
                    {
                        Opacity = 0.35,
                    };
                    draw.DrawEllipse(blob, null, centre, radius, radius);
                }

                var caption = new FormattedText(
                    "seed " + seed.ToString(CultureInfo.InvariantCulture),
                    CultureInfo.InvariantCulture,
                    FlowDirection.LeftToRight,
                    new Typeface("Segoe UI"),
                    Math.Max(11, width / 48.0),
                    Brushes.White,
                    1.0);
                draw.DrawText(caption, new Point(width * 0.04, height * 0.9));
            }

            var bitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
            bitmap.Render(visual);

            var rows = new FormatConvertedBitmap(bitmap, PixelFormats.Rgb24, null, 0);
            var rgb = new byte[width * height * 3];
            rows.CopyPixels(rgb, width * 3, 0);

            wide = width;
            high = height;
            return rgb;
        }

        private static Color FromHue(double hue, double saturation, double value)
        {
            hue = ((hue % 360) + 360) % 360;
            var c = value * saturation;
            var x = c * (1 - Math.Abs((hue / 60.0) % 2 - 1));
            var m = value - c;
            double r = 0, g = 0, b = 0;

            if (hue < 60) { r = c; g = x; }
            else if (hue < 120) { r = x; g = c; }
            else if (hue < 180) { g = c; b = x; }
            else if (hue < 240) { g = x; b = c; }
            else if (hue < 300) { r = x; b = c; }
            else { r = c; b = x; }

            return Color.FromRgb(
                (byte)((r + m) * 255), (byte)((g + m) * 255), (byte)((b + m) * 255));
        }

        /// <summary>
        /// A recording, of a sort: mono samples at the rate the library speaks at, long enough to
        /// have something to drag the player's slider along.
        /// </summary>
        private float[] Samples(Category category)
        {
            var words = string.IsNullOrEmpty(job.Text) ? 12 : job.Text.Split(' ').Length;
            var seconds = Math.Max(1.5, Math.Min(20.0, words * 0.4 / Math.Max(0.25, job.Speed ?? 1.0)));
            var count = (int)(seconds * Rate);
            var random = new Random((int)(seed & 0x7FFFFFFF));
            var pitch = category == Category.Speech ? 180.0 : 140.0;
            var samples = new float[count];

            var wobble = 0.0;
            for (var i = 0; i < count; i++)
            {
                var t = (double)i / Rate;
                if (i % (Rate / 6) == 0)
                {
                    wobble = (random.NextDouble() - 0.5) * 40;
                }

                var envelope = Math.Min(1.0, Math.Min(t * 8, (seconds - t) * 8));
                samples[i] = (float)(Math.Sin(2 * Math.PI * (pitch + wobble) * t) * 0.35 * envelope);
            }

            return samples;
        }
    }
}
