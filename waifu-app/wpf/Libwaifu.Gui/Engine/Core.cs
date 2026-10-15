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
using System.Globalization;

namespace Libwaifu.Gui.Engine
{
    /// <summary>The four things this screen can be asked for.</summary>
    public enum TaskKind
    {
        Txt2Img,
        Img2Img,
        Text2Speech,
        Speech2Speech,
    }

    /// <summary>
    /// What a model is for, which is also what a result is filed under: the strip under a picture
    /// shows every picture ever drawn, whichever of the two picture tasks drew it.
    /// </summary>
    public enum Category
    {
        Image,
        Speech,
        Conversion,
    }

    /// <summary>Where a run has got to. The names are the library's own progress phases.</summary>
    public enum Stage
    {
        Fetching,
        Reading,
        Encoding,
        Drawing,
        Decoding,
        Listening,
        Saying,
        Sounding,
    }

    public static class Tasks
    {
        /// <summary>
        /// The tasks the sidebar lists. All four pages exist; this is the one switch that decides
        /// which of them are offered, and it is the same switch on both platforms.
        /// </summary>
        public static readonly TaskKind[] Tabs = { TaskKind.Txt2Img };

        public static string Name(TaskKind kind)
        {
            switch (kind)
            {
                case TaskKind.Txt2Img: return "txt2img";
                case TaskKind.Img2Img: return "img2img";
                case TaskKind.Text2Speech: return "text2speech";
                default: return "speech2speech";
            }
        }

        public static Category CategoryOf(TaskKind kind)
        {
            switch (kind)
            {
                case TaskKind.Txt2Img:
                case TaskKind.Img2Img:
                    return Category.Image;
                case TaskKind.Text2Speech:
                    return Category.Speech;
                default:
                    return Category.Conversion;
            }
        }

        /// <summary>What the big button says, and what the empty output area says.</summary>
        public static string Verb(TaskKind kind)
        {
            switch (kind)
            {
                case TaskKind.Txt2Img:
                case TaskKind.Img2Img:
                    return "Generate";
                case TaskKind.Text2Speech:
                    return "Speak";
                default:
                    return "Convert";
            }
        }

        public static string NothingYet(Category category)
        {
            switch (category)
            {
                case Category.Image: return "Nothing drawn yet.";
                case Category.Speech: return "Nothing said yet.";
                default: return "Nothing converted yet.";
            }
        }
    }

    /// <summary>
    /// A device a model can be read onto. The mac offers auto, metal and cpu; these are what the
    /// library has on Windows.
    /// </summary>
    /// <remarks>
    /// "cuda (offload)" is WAIFU_DEVICE_CUDA_CPU_OFFLOAD, which keeps what will not fit in the
    /// card's memory in the host's and streams it in. It is the answer to a model that is bigger
    /// than the card, which on a 6 GB laptop card is most of them.
    /// </remarks>
    public static class Devices
    {
        public static readonly string[] All = { "cuda", "cuda (offload)", "vulkan", "cpu" };

        /// <summary>
        /// What to start on. The library has an "auto" of its own, but a list that says which
        /// device a run went to is worth more than one that says somebody else chose: this picks
        /// the same way auto would and then says so by name.
        /// </summary>
        public static string Best()
        {
            var system = Environment.GetFolderPath(Environment.SpecialFolder.System);

            if (System.IO.File.Exists(System.IO.Path.Combine(system, "nvcuda.dll")))
            {
                return "cuda";
            }

            if (System.IO.File.Exists(System.IO.Path.Combine(system, "vulkan-1.dll")))
            {
                return "vulkan";
            }

            return "cpu";
        }

        /// <summary>A remembered device this build no longer offers reads as the best one.</summary>
        public static string Known(string name)
        {
            return Array.IndexOf(All, name) >= 0 ? name : Best();
        }
    }

    /// <summary>One thing the screen asked for, with every setting it was asked with.</summary>
    public class JobRequest
    {
        public TaskKind Kind { get; set; }
        public string Model { get; set; }
        public string ModelFullName { get; set; }
        public string Device { get; set; }

        public string Prompt { get; set; }
        public string Negative { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public int Steps { get; set; }
        public double? Guidance { get; set; }
        public double? Strength { get; set; }

        /// <summary>
        /// The picture img2img starts from, as rows of RGB -- the shape the engine takes, so the
        /// conversion happens once, where the picture is chosen, rather than at every job.
        /// </summary>
        public byte[] SourceRgb { get; set; }

        public int SourceWidth { get; set; }

        public int SourceHeight { get; set; }

        /// <summary>The recording to convert, for speech2speech.</summary>
        public Recording Source { get; set; }

        /// <summary>The recording to sound like, or the voice to convert into.</summary>
        public Recording Reference { get; set; }

        public string Text { get; set; }
        public double? Speed { get; set; }
        public double? Temperature { get; set; }
        public string Style { get; set; }
        public bool FromARecording { get; set; }
        public bool? ConvertStyle { get; set; }

        /// <summary>
        /// The seed as it was typed. Anything that is not a whole number -- "-1", or nothing at
        /// all -- means a new one every time, which is settled when the job starts rather than
        /// when it is queued, so two queued jobs do not draw the same picture.
        /// </summary>
        public string SeedText { get; set; }

        public long Settle()
        {
            long typed;
            if (!string.IsNullOrWhiteSpace(SeedText) &&
                long.TryParse(SeedText.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out typed) &&
                typed >= 0)
            {
                return typed;
            }

            var bytes = new byte[8];
            using (var random = new System.Security.Cryptography.RNGCryptoServiceProvider())
            {
                random.GetBytes(bytes);
            }

            return (long)(BitConverter.ToUInt64(bytes, 0) & 0x7FFFFFFFFFFFFFFFUL);
        }
    }

    /// <summary>One progress event: how far, how long, and the line to put under the bar.</summary>
    public class EngineEvent
    {
        public EngineEvent(Stage stage, double? fraction, double seconds, string line)
        {
            Stage = stage;
            Fraction = fraction;
            Seconds = seconds;
            Line = line;
        }

        public Stage Stage { get; private set; }

        /// <summary>Null where nothing knows how much is left, which is most of a download.</summary>
        public double? Fraction { get; private set; }

        public double Seconds { get; private set; }
        public string Line { get; private set; }
    }

    public enum Ending
    {
        Done,
        Cancelled,
        Failed,
    }

    public class JobOutcome
    {
        public Ending Ending { get; set; }
        public string Error { get; set; }
        public ResultRecord Result { get; set; }
    }

    /// <summary>
    /// What the screen drives. One implementation simulates a run; the real one calls
    /// <c>waifu_engine_*</c> through the C API and marshals every callback onto the UI thread.
    /// </summary>
    public interface IEngine : IDisposable
    {
        /// <summary>The model that is on the device now, or null while nothing is loaded.</summary>
        string LoadedModel { get; }

        string LoadedDevice { get; }

        event Action<EngineEvent> Progress;

        event Action<JobOutcome> Finished;

        /// <summary>
        /// Start one job. Loading the model, if it is not the one already on the device, is part
        /// of the job rather than something the screen asks for separately -- which is why the
        /// first picture after a model is chosen spends its first seconds saying "reading".
        /// </summary>
        void Begin(JobRequest job);

        /// <summary>Stop after the step it is on. Between files, while fetching.</summary>
        void Cancel();
    }

    /// <summary>
    /// The last 500 lines the library wrote, for the log window. Registered before any other call
    /// on the real thing, so that a failure during the first load is in here to be read.
    /// </summary>
    public static class Log
    {
        private const int Keep = 500;

        private static readonly Queue<string> Lines = new Queue<string>();

        public static void Write(string level, string source, string message)
        {
            lock (Lines)
            {
                Lines.Enqueue(level.ToUpperInvariant() + " " + source + "] " + message);
                while (Lines.Count > Keep)
                {
                    Lines.Dequeue();
                }
            }
        }

        public static string All()
        {
            lock (Lines)
            {
                return string.Join(Environment.NewLine, Lines.ToArray());
            }
        }
    }
}
