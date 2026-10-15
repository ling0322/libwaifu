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
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;

namespace Libwaifu.Gui.Engine
{
    /// <summary>Where everything this app keeps lives.</summary>
    /// <remarks>
    /// Roaming rather than Local, and fixed: the mac writes under Application Support and this
    /// writes under AppData, and the two directories hold the same JSON so one machine's output
    /// folder opens on the other.
    /// </remarks>
    public static class Places
    {
        public static string Root
        {
            get
            {
                return Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "libwaifu");
            }
        }

        public static string Output
        {
            get { return Overridden ?? Path.Combine(Root, "output"); }
        }

        /// <summary>
        /// Where <c>-output</c> on the command line points. It changes this run and is not saved,
        /// which is the whole of what that flag means.
        /// </summary>
        public static string Overridden { get; set; }

        public static string SettingsFile
        {
            get { return Path.Combine(Root, "settings.json"); }
        }

        /// <summary>Where models are fetched into. The library owns this; the screen can move it.</summary>
        public static string Models { get; set; }

        public static string Ensure(string directory)
        {
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            return directory;
        }
    }

    /// <summary>What a run was asked for, written beside the picture it drew.</summary>
    /// <remarks>
    /// The names are the mac side's, letter for letter, because the point of writing them down is
    /// that either app can read the other's output folder.
    /// </remarks>
    [DataContract]
    public class Made
    {
        [DataMember(Name = "seed", EmitDefaultValue = false)]
        public long Seed { get; set; }

        [DataMember(Name = "seconds", EmitDefaultValue = false)]
        public double Seconds { get; set; }

        [DataMember(Name = "model", EmitDefaultValue = false)]
        public string Model { get; set; }

        [DataMember(Name = "parameters", EmitDefaultValue = false)]
        public string Parameters { get; set; }

        [DataMember(Name = "prompt", EmitDefaultValue = false)]
        public string Prompt { get; set; }

        [DataMember(Name = "negative", EmitDefaultValue = false)]
        public string Negative { get; set; }

        [DataMember(Name = "width", EmitDefaultValue = false)]
        public int? Width { get; set; }

        [DataMember(Name = "height", EmitDefaultValue = false)]
        public int? Height { get; set; }

        [DataMember(Name = "steps", EmitDefaultValue = false)]
        public int? Steps { get; set; }

        [DataMember(Name = "guidance", EmitDefaultValue = false)]
        public double? Guidance { get; set; }

        [DataMember(Name = "strength", EmitDefaultValue = false)]
        public double? Strength { get; set; }

        [DataMember(Name = "text", EmitDefaultValue = false)]
        public string Text { get; set; }

        [DataMember(Name = "speed", EmitDefaultValue = false)]
        public double? Speed { get; set; }

        [DataMember(Name = "temperature", EmitDefaultValue = false)]
        public double? Temperature { get; set; }

        [DataMember(Name = "style", EmitDefaultValue = false)]
        public string Style { get; set; }

        [DataMember(Name = "fromARecording", EmitDefaultValue = false)]
        public bool? FromARecording { get; set; }

        [DataMember(Name = "convertStyle", EmitDefaultValue = false)]
        public bool? ConvertStyle { get; set; }

        [DataMember(Name = "length", EmitDefaultValue = false)]
        public double? Length { get; set; }

        [DataMember(Name = "rate", EmitDefaultValue = false)]
        public int? Rate { get; set; }
    }

    /// <summary>One finished result: a file on the disk and the JSON beside it.</summary>
    [DataContract]
    public class ResultRecord
    {
        [DataMember(Name = "id")]
        public string Id { get; set; }

        /// <summary>"image", "speech" or "conversion" -- the category, spelled as the mac spells it.</summary>
        [DataMember(Name = "kind")]
        public string Kind { get; set; }

        /// <summary>Milliseconds since 1970, as a double, on both platforms.</summary>
        [DataMember(Name = "created")]
        public double Created { get; set; }

        [DataMember(Name = "finished")]
        public double Finished { get; set; }

        [DataMember(Name = "made")]
        public Made Made { get; set; }

        public Category Category
        {
            get
            {
                if (Kind == "speech")
                {
                    return Category.Speech;
                }

                return Kind == "conversion" ? Category.Conversion : Category.Image;
            }
        }

        public string FileName
        {
            get { return Id + (Category == Category.Image ? ".png" : ".wav"); }
        }

        public static string KindOf(Category category)
        {
            switch (category)
            {
                case Category.Image: return "image";
                case Category.Speech: return "speech";
                default: return "conversion";
            }
        }

        public static double Now()
        {
            return (DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalMilliseconds;
        }
    }

    /// <summary>The output folder, read back at startup and written to as runs finish.</summary>
    public static class Store
    {
        /// <summary>Everything in the output folder, newest first.</summary>
        public static List<ResultRecord> ReadAll()
        {
            var found = new List<ResultRecord>();
            var folder = Places.Output;
            if (!Directory.Exists(folder))
            {
                return found;
            }

            foreach (var file in Directory.GetFiles(folder, "*.json"))
            {
                try
                {
                    using (var stream = File.OpenRead(file))
                    {
                        var reader = new DataContractJsonSerializer(typeof(ResultRecord));
                        var record = (ResultRecord)reader.ReadObject(stream);
                        if (record != null && record.Id != null)
                        {
                            found.Add(record);
                        }
                    }
                }
                catch (Exception error)
                {
                    // One unreadable file is not a reason to start with no history at all.
                    Log.Write("warn", "store", Path.GetFileName(file) + ": " + error.Message);
                }
            }

            return found.OrderByDescending(record => record.Created).ToList();
        }

        public static void Write(ResultRecord record)
        {
            Places.Ensure(Places.Output);
            var file = Path.Combine(Places.Output, record.Id + ".json");
            using (var stream = File.Create(file))
            {
                new DataContractJsonSerializer(typeof(ResultRecord)).WriteObject(stream, record);
            }
        }

        /// <summary>The picture and its JSON both go, because nothing else keeps a copy.</summary>
        public static void Delete(ResultRecord record)
        {
            foreach (var file in new[]
            {
                Path.Combine(Places.Output, record.FileName),
                Path.Combine(Places.Output, record.Id + ".json"),
            })
            {
                if (File.Exists(file))
                {
                    File.Delete(file);
                }
            }
        }

        public static string PathOf(ResultRecord record)
        {
            return Path.Combine(Places.Output, record.FileName);
        }
    }

    /// <summary>
    /// The settings text, in the shape every tool that reads a picture's metadata expects.
    /// </summary>
    public static class Parameters
    {
        public static string ForPicture(JobRequest job, long seed)
        {
            var text = new StringBuilder();
            text.Append(job.Prompt);

            if (job.Guidance.HasValue && !string.IsNullOrEmpty(job.Negative))
            {
                text.Append("\nNegative prompt: ").Append(job.Negative);
            }

            text.Append("\nSteps: ").Append(job.Steps.ToString(CultureInfo.InvariantCulture));
            if (job.Guidance.HasValue)
            {
                text.Append(", CFG scale: ")
                    .Append(job.Guidance.Value.ToString("0.0", CultureInfo.InvariantCulture));
            }

            text.Append(", Seed: ").Append(seed.ToString(CultureInfo.InvariantCulture));
            text.Append(", Size: ").Append(job.Width).Append("x").Append(job.Height);
            text.Append(", Model: ").Append(job.Model);

            if (job.Strength.HasValue)
            {
                text.Append(", Denoising strength: ")
                    .Append(job.Strength.Value.ToString("0.00", CultureInfo.InvariantCulture));
            }

            return text.ToString();
        }

        public static string ForSpeech(JobRequest job, long seed, int rate)
        {
            var text = new StringBuilder();
            text.Append(job.Text).Append("\n");
            text.Append("Speed: ").Append(Round(job.Speed));
            text.Append(", Temperature: ").Append(Round(job.Temperature));
            text.Append(", Seed: ").Append(seed.ToString(CultureInfo.InvariantCulture));
            text.Append(", Rate: ").Append(rate).Append(" Hz");
            text.Append(", Voice: ").Append(job.Model);

            if (job.FromARecording)
            {
                text.Append(", From a recording: yes");
            }

            if (!string.IsNullOrEmpty(job.Style))
            {
                text.Append(", Style: ").Append(job.Style);
            }

            return text.ToString();
        }

        public static string ForConversion(JobRequest job, long seed, int rate)
        {
            var text = new StringBuilder();
            text.Append("Steps: ").Append(job.Steps.ToString(CultureInfo.InvariantCulture));
            text.Append(", Style: ").Append(job.ConvertStyle == true ? "yes" : "no");
            text.Append(", Seed: ").Append(seed.ToString(CultureInfo.InvariantCulture));
            text.Append(", Rate: ").Append(rate).Append(" Hz");
            text.Append(", Converter: ").Append(job.Model);
            return text.ToString();
        }

        private static string Round(double? value)
        {
            return (value ?? 1.0).ToString("0.00", CultureInfo.InvariantCulture);
        }
    }

    /// <summary>What is remembered between runs: the device, and one model per category.</summary>
    [DataContract]
    public class Settings
    {
        [DataMember(Name = "device")]
        public string Device { get; set; }

        [DataMember(Name = "model.image")]
        public string ImageModel { get; set; }

        [DataMember(Name = "model.speech")]
        public string SpeechModel { get; set; }

        [DataMember(Name = "model.conversion")]
        public string ConversionModel { get; set; }

        public static Settings Read()
        {
            try
            {
                if (File.Exists(Places.SettingsFile))
                {
                    using (var stream = File.OpenRead(Places.SettingsFile))
                    {
                        var settings = (Settings)new DataContractJsonSerializer(typeof(Settings))
                            .ReadObject(stream);
                        if (settings != null)
                        {
                            return settings;
                        }
                    }
                }
            }
            catch (Exception error)
            {
                Log.Write("warn", "settings", error.Message);
            }

            return new Settings { Device = "auto" };
        }

        public void Save()
        {
            try
            {
                Places.Ensure(Places.Root);
                using (var stream = File.Create(Places.SettingsFile))
                {
                    new DataContractJsonSerializer(typeof(Settings)).WriteObject(stream, this);
                }
            }
            catch (Exception error)
            {
                Log.Write("warn", "settings", error.Message);
            }
        }

        public string ModelFor(Category category)
        {
            switch (category)
            {
                case Category.Image: return ImageModel;
                case Category.Speech: return SpeechModel;
                default: return ConversionModel;
            }
        }

        public void SetModelFor(Category category, string name)
        {
            switch (category)
            {
                case Category.Image:
                    ImageModel = name;
                    break;
                case Category.Speech:
                    SpeechModel = name;
                    break;
                default:
                    ConversionModel = name;
                    break;
            }
        }
    }
}
