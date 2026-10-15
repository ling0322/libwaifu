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
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Libwaifu.Gui.Engine
{
    /// <summary>
    /// Writing what a job made, and the record that goes beside it.
    /// </summary>
    /// <remarks>
    /// The engine hands over pixels and samples and keeps nothing; the file and its JSON are this
    /// program's to write. Both engines come through here, so a picture from the stand-in and one
    /// from the library are filed the same way and the mac can read either.
    /// </remarks>
    public static class Output
    {
        /// <summary>Rows of RGB, as the engine gives them, written as a PNG.</summary>
        public static ResultRecord Picture(
            JobRequest job, long seed, double started, double seconds, int width, int height, byte[] rgb)
        {
            var id = NewId();
            Places.Ensure(Places.Output);

            var frame = BitmapSource.Create(
                width, height, 96, 96, PixelFormats.Rgb24, null, rgb, width * 3);
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(frame));
            using (var stream = File.Create(Path.Combine(Places.Output, id + ".png")))
            {
                encoder.Save(stream);
            }

            var made = new Made
            {
                Seed = seed,
                Seconds = seconds,
                Model = job.Model,
                Prompt = job.Prompt,
                Negative = job.Negative,
                Width = width,
                Height = height,
                Steps = job.Steps,
                Guidance = job.Guidance,
                Strength = job.Strength,
                Parameters = Parameters.ForPicture(job, seed),
            };

            return Keep(id, Category.Image, started, made);
        }

        /// <summary>
        /// Mono samples between -1 and 1, written as the 16-bit PCM the mac writes.
        /// </summary>
        public static ResultRecord Sound(
            JobRequest job,
            Category category,
            long seed,
            double started,
            double seconds,
            float[] samples,
            int rate)
        {
            var id = NewId();
            Places.Ensure(Places.Output);
            Wav(Path.Combine(Places.Output, id + ".wav"), samples, rate);

            var length = rate > 0 ? (double)samples.Length / rate : 0;
            var made = new Made
            {
                Seed = seed,
                Seconds = seconds,
                Model = job.Model,
                Length = length,
                Rate = rate,
            };

            if (category == Category.Speech)
            {
                made.Text = job.Text;
                made.Speed = job.Speed;
                made.Temperature = job.Temperature;
                made.Style = job.Style;
                made.FromARecording = job.FromARecording;
                made.Parameters = Parameters.ForSpeech(job, seed, rate);
            }
            else
            {
                made.Steps = job.Steps;
                made.ConvertStyle = job.ConvertStyle;
                made.Parameters = Parameters.ForConversion(job, seed, rate);
            }

            return Keep(id, category, started, made);
        }

        public static void Wav(string path, float[] samples, int rate)
        {
            using (var stream = File.Create(path))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(new[] { 'R', 'I', 'F', 'F' });
                writer.Write(36 + samples.Length * 2);
                writer.Write(new[] { 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' });
                writer.Write(16);
                writer.Write((short)1);
                writer.Write((short)1);
                writer.Write(rate);
                writer.Write(rate * 2);
                writer.Write((short)2);
                writer.Write((short)16);
                writer.Write(new[] { 'd', 'a', 't', 'a' });
                writer.Write(samples.Length * 2);

                foreach (var sample in samples)
                {
                    var held = Math.Max(-1f, Math.Min(1f, sample));
                    writer.Write((short)Math.Round(held * short.MaxValue));
                }
            }
        }

        private static ResultRecord Keep(string id, Category category, double started, Made made)
        {
            var record = new ResultRecord
            {
                Id = id,
                Kind = ResultRecord.KindOf(category),
                Created = started,
                Finished = ResultRecord.Now(),
                Made = made,
            };

            Store.Write(record);
            return record;
        }

        private static string NewId()
        {
            return Guid.NewGuid().ToString("N").ToLowerInvariant();
        }
    }
}
