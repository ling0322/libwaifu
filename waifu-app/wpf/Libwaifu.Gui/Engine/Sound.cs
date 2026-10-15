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
using System.Text;

namespace Libwaifu.Gui.Engine
{
    /// <summary>A recording read off the disk: mono samples between -1 and 1, and their rate.</summary>
    public class Recording
    {
        public float[] Samples { get; set; }

        public int Rate { get; set; }

        public double Seconds
        {
            get { return Rate > 0 ? (double)Samples.Length / Rate : 0; }
        }

        /// <summary>The first <paramref name="seconds"/> of it, which is all the engine takes.</summary>
        public Recording Head(double seconds)
        {
            var keep = (int)Math.Min(Samples.Length, Math.Max(1, seconds * Rate));
            if (keep >= Samples.Length)
            {
                return this;
            }

            var cut = new float[keep];
            Array.Copy(Samples, cut, keep);
            return new Recording { Samples = cut, Rate = Rate };
        }
    }

    /// <summary>
    /// Reading a recording into the samples the engine takes.
    /// </summary>
    /// <remarks>
    /// WAV only, for now: .NET Framework has no decoder of its own, and the ones Windows has are
    /// COM interfaces rather than a library to call. What this program writes is WAV, so what it
    /// made can be handed back to it; anything else says what it is and stops there rather than
    /// failing somewhere further in.
    /// </remarks>
    public static class Sound
    {
        /// <summary>The samples, or null with the reason.</summary>
        public static Recording Read(string path, out string trouble)
        {
            trouble = null;

            try
            {
                var extension = (Path.GetExtension(path) ?? string.Empty).ToLowerInvariant();
                if (extension != ".wav")
                {
                    trouble = "only WAV recordings can be read here, and this is " +
                              (extension.Length > 1 ? extension.Substring(1).ToUpperInvariant() : "not one");
                    return null;
                }

                return ReadWav(path);
            }
            catch (Exception error)
            {
                trouble = error.Message;
                return null;
            }
        }

        /// <summary>
        /// RIFF, as every recorder writes it: the chunks in whatever order, PCM of 8, 16, 24 or
        /// 32 bits or floats, and however many channels averaged down to one.
        /// </summary>
        private static Recording ReadWav(string path)
        {
            using (var stream = File.OpenRead(path))
            using (var reader = new BinaryReader(stream))
            {
                if (Tag(reader) != "RIFF")
                {
                    throw new InvalidDataException("not a RIFF file");
                }

                reader.ReadInt32();
                if (Tag(reader) != "WAVE")
                {
                    throw new InvalidDataException("not a WAVE file");
                }

                int format = 1, channels = 1, rate = 44100, bits = 16;
                var seen = false;

                while (stream.Position + 8 <= stream.Length)
                {
                    var name = Tag(reader);
                    var size = reader.ReadInt32();
                    var next = stream.Position + size + (size % 2);

                    if (name == "fmt ")
                    {
                        format = reader.ReadInt16();
                        channels = Math.Max(1, (int)reader.ReadInt16());
                        rate = reader.ReadInt32();
                        reader.ReadInt32();
                        reader.ReadInt16();
                        bits = reader.ReadInt16();

                        // WAVE_FORMAT_EXTENSIBLE says what it really is at the end of the block.
                        if (format == 0xFFFE && size >= 40)
                        {
                            reader.ReadInt16();
                            reader.ReadInt16();
                            reader.ReadInt32();
                            format = reader.ReadInt16();
                        }

                        seen = true;
                    }
                    else if (name == "data")
                    {
                        if (!seen)
                        {
                            throw new InvalidDataException("the data comes before the format");
                        }

                        var raw = reader.ReadBytes(size);
                        return new Recording
                        {
                            Samples = Mono(raw, format, bits, channels),
                            Rate = rate,
                        };
                    }

                    stream.Position = next;
                }

                throw new InvalidDataException("there is no sound in it");
            }
        }

        private static float[] Mono(byte[] raw, int format, int bits, int channels)
        {
            var width = bits / 8;
            if (width < 1)
            {
                throw new InvalidDataException(bits + " bits a sample");
            }

            var frames = raw.Length / (width * channels);
            var samples = new float[frames];

            for (var frame = 0; frame < frames; frame++)
            {
                double sum = 0;
                for (var channel = 0; channel < channels; channel++)
                {
                    sum += One(raw, (frame * channels + channel) * width, format, bits);
                }

                samples[frame] = (float)(sum / channels);
            }

            return samples;
        }

        private static double One(byte[] raw, int at, int format, int bits)
        {
            // 3 is IEEE float; 1 is PCM, which is signed at every width but eight bits.
            if (format == 3)
            {
                return bits == 64 ? BitConverter.ToDouble(raw, at) : BitConverter.ToSingle(raw, at);
            }

            switch (bits)
            {
                case 8:
                    return (raw[at] - 128) / 128.0;
                case 16:
                    return BitConverter.ToInt16(raw, at) / 32768.0;
                case 24:
                    var whole = raw[at] | (raw[at + 1] << 8) | (raw[at + 2] << 16);
                    if ((whole & 0x800000) != 0)
                    {
                        whole |= unchecked((int)0xFF000000);
                    }

                    return whole / 8388608.0;
                case 32:
                    return BitConverter.ToInt32(raw, at) / 2147483648.0;
                default:
                    throw new InvalidDataException(bits + " bits a sample");
            }
        }

        private static string Tag(BinaryReader reader)
        {
            return Encoding.ASCII.GetString(reader.ReadBytes(4));
        }
    }
}
