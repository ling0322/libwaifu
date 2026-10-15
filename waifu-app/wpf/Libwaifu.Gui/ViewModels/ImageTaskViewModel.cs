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
using System.Collections.ObjectModel;
using System.Globalization;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Libwaifu.Gui.Engine;

namespace Libwaifu.Gui.ViewModels
{
    /// <summary>txt2img and img2img, which differ by one group of settings and one condition.</summary>
    public class ImageTaskViewModel : TaskViewModel
    {
        private const int Smallest = 64;
        private const int Largest = 2048;

        private string prompt = string.Empty;
        private string avoid = string.Empty;
        private string suggestedPrompt;
        private string suggestedAvoid;
        private int width = 1024;
        private int height = 1024;
        private int steps = 10;
        private double cfg = 5.0;
        private double strength = 0.8;
        private string seed = "-1";
        private string preset;
        private int alignment = 16;
        private bool settling;
        private BitmapSource source;
        private string sourcePath;
        private byte[] rgb;
        private int rgbWidth;
        private int rgbHeight;

        public ImageTaskViewModel(MainViewModel main, TaskKind kind)
            : base(main, kind)
        {
            Dice = new Command(delegate
            {
                Seed = new Random().Next(1, 65536).ToString(CultureInfo.InvariantCulture);
            });

            ClearSource = new Command(delegate
            {
                Source = null;
                SourcePath = null;
            });

            RebuildPresets();
        }

        protected override string IconKey
        {
            get { return Kind == TaskKind.Txt2Img ? "IconPicture" : "IconPictureStack"; }
        }

        protected override string BrushKey
        {
            get { return Kind == TaskKind.Txt2Img ? "TaskImageBrush" : "TaskImage2Brush"; }
        }

        public Command Dice { get; private set; }

        public Command ClearSource { get; private set; }

        public bool IsImg2Img
        {
            get { return Kind == TaskKind.Img2Img; }
        }

        // -- what to draw ------------------------------------------------------------------------

        public string Prompt
        {
            get { return prompt; }
            set
            {
                if (Set(ref prompt, value))
                {
                    GoChanged();
                }
            }
        }

        public string Avoid
        {
            get { return avoid; }
            set { Set(ref avoid, value); }
        }

        /// <summary>
        /// A model distilled for no guidance has nothing to steer away from, so the box is not
        /// shown at all and the prompt takes the whole width.
        /// </summary>
        public bool TakesGuidance
        {
            get { return Description != null && Description.TakesGuidance; }
        }

        // -- how big -----------------------------------------------------------------------------

        public ObservableCollection<string> Presets { get; private set; }

        public string Preset
        {
            get { return preset; }
            set
            {
                if (!Set(ref preset, value) || settling || value == null || value == "Custom")
                {
                    return;
                }

                var sides = value.Split('x');
                if (sides.Length == 2)
                {
                    settling = true;
                    Width = int.Parse(sides[0].Trim(), CultureInfo.InvariantCulture);
                    Height = int.Parse(sides[1].Trim(), CultureInfo.InvariantCulture);
                    settling = false;
                }
            }
        }

        public int Width
        {
            get { return width; }
            set
            {
                if (Set(ref width, Fit(value)))
                {
                    MatchPreset();
                }
            }
        }

        public int Height
        {
            get { return height; }
            set
            {
                if (Set(ref height, Fit(value)))
                {
                    MatchPreset();
                }
            }
        }

        // -- how hard ------------------------------------------------------------------------------

        public int Steps
        {
            get { return steps; }
            set { Set(ref steps, Math.Max(1, Math.Min(150, value))); }
        }

        public double Cfg
        {
            get { return cfg; }
            set { Set(ref cfg, Math.Round(Math.Max(1, Math.Min(30, value)), 1)); }
        }

        public double Strength
        {
            get { return strength; }
            set { Set(ref strength, Math.Round(Math.Max(0, Math.Min(1, value)), 2)); }
        }

        public string Seed
        {
            get { return seed; }
            set { Set(ref seed, value); }
        }

        // -- the picture to start from ----------------------------------------------------------------

        public bool CanStartFromPicture
        {
            get { return Description != null && Description.CanStartFromPicture; }
        }

        /// <summary>The whole sentence, which is the only thing in that group when it cannot.</summary>
        public string WhyNotFromPicture
        {
            get
            {
                if (Description == null || Description.CanStartFromPicture)
                {
                    return null;
                }

                return "This model cannot start from a picture: " + Description.WhyNotFromPicture + ".";
            }
        }

        public BitmapSource Source
        {
            get { return source; }
            set
            {
                if (!Set(ref source, value))
                {
                    return;
                }

                // The engine takes rows of RGB at whatever size, and scales them itself. Turning
                // the picture into that shape here means it happens once, when the picture is
                // chosen, rather than at every job it is drawn from.
                rgb = null;
                if (value != null)
                {
                    try
                    {
                        var rows = new FormatConvertedBitmap(value, PixelFormats.Rgb24, null, 0);
                        var width = rows.PixelWidth;
                        var height = rows.PixelHeight;
                        var pixels = new byte[width * height * 3];
                        rows.CopyPixels(pixels, width * 3, 0);
                        rgb = pixels;
                        rgbWidth = width;
                        rgbHeight = height;
                    }
                    catch (Exception error)
                    {
                        Log.Write("warn", "img2img", error.Message);
                    }
                }

                Raise("HasSource");
                GoChanged();
            }
        }

        public string SourcePath
        {
            get { return sourcePath; }
            set { Set(ref sourcePath, value); }
        }

        public bool HasSource
        {
            get { return source != null; }
        }

        /// <summary>Read a dropped or chosen file, which is as far as a picture gets until a job runs.</summary>
        public void TakePicture(string path, BitmapSource pixels)
        {
            try
            {
                if (pixels != null)
                {
                    Source = pixels;
                    SourcePath = null;
                    return;
                }

                var read = new BitmapImage();
                read.BeginInit();
                read.CacheOption = BitmapCacheOption.OnLoad;
                read.UriSource = new Uri(path);
                read.EndInit();
                read.Freeze();
                Source = read;
                SourcePath = path;
            }
            catch (Exception error)
            {
                Log.Write("warn", "img2img", error.Message);
            }
        }

        // -- what the base class asks for ---------------------------------------------------------------

        public override bool CanGo
        {
            get
            {
                if (Main.IsQueued(this) || ChosenModel == null || string.IsNullOrWhiteSpace(Prompt))
                {
                    return false;
                }

                return !IsImg2Img || (CanStartFromPicture && HasSource);
            }
        }

        protected override JobRequest Build()
        {
            return new JobRequest
            {
                Prompt = Prompt,
                Negative = TakesGuidance ? Avoid : null,
                Width = Width,
                Height = Height,
                Steps = Steps,
                Guidance = TakesGuidance ? (double?)Cfg : null,
                Strength = IsImg2Img ? (double?)Strength : null,
                SeedText = Seed,
                SourceRgb = IsImg2Img ? rgb : null,
                SourceWidth = rgbWidth,
                SourceHeight = rgbHeight,
            };
        }

        protected override void AdoptDescription(ModelDescription model)
        {
            alignment = model.Alignment;
            settling = true;
            if (model.Sizes.Count > 0)
            {
                width = Fit(model.Sizes[0].Key);
                height = Fit(model.Sizes[0].Value);
            }

            steps = model.Steps;
            cfg = model.Guidance;
            settling = false;

            // The two boxes of words are the one thing a model does not get to overwrite: what is
            // in them was either typed by somebody or put there by the model before this one.
            if (string.IsNullOrEmpty(prompt) || prompt == suggestedPrompt)
            {
                prompt = model.Prompt ?? string.Empty;
            }

            if (string.IsNullOrEmpty(avoid) || avoid == suggestedAvoid)
            {
                avoid = model.Avoid ?? string.Empty;
            }

            suggestedPrompt = model.Prompt;
            suggestedAvoid = model.Avoid;

            RebuildPresets();
            foreach (var name in new[]
            {
                "Prompt", "Avoid", "Width", "Height", "Steps", "Cfg", "TakesGuidance",
                "CanStartFromPicture", "WhyNotFromPicture",
            })
            {
                Raise(name);
            }
        }

        protected override void ReuseFrom(ResultRecord record)
        {
            var made = record.Made;
            settling = true;
            Prompt = made.Prompt ?? string.Empty;
            Avoid = made.Negative ?? string.Empty;
            if (made.Width.HasValue)
            {
                Width = made.Width.Value;
            }

            if (made.Height.HasValue)
            {
                Height = made.Height.Value;
            }

            if (made.Steps.HasValue)
            {
                Steps = made.Steps.Value;
            }

            if (made.Guidance.HasValue)
            {
                Cfg = made.Guidance.Value;
            }

            if (made.Strength.HasValue)
            {
                Strength = made.Strength.Value;
            }

            Seed = made.Seed.ToString(CultureInfo.InvariantCulture);
            settling = false;
            MatchPreset();
        }

        // -- the two ways of saying the same size ---------------------------------------------------------

        private void RebuildPresets()
        {
            if (Presets == null)
            {
                Presets = new ObservableCollection<string>();
                Raise("Presets");
            }

            Presets.Clear();
            if (Description != null)
            {
                foreach (var size in Description.Sizes)
                {
                    Presets.Add(size.Key + " x " + size.Value);
                }
            }

            MatchPreset();
        }

        /// <summary>
        /// A size typed into the boxes that is not one the model lists shows as Custom rather
        /// than as whichever preset it is nearest to.
        /// </summary>
        private void MatchPreset()
        {
            var here = width + " x " + height;
            if (!Presets.Contains(here))
            {
                if (Presets.Count == 0 || Presets[0] != "Custom")
                {
                    Presets.Insert(0, "Custom");
                }

                preset = "Custom";
            }
            else
            {
                if (Presets.Count > 0 && Presets[0] == "Custom")
                {
                    Presets.RemoveAt(0);
                }

                preset = here;
            }

            Raise("Preset");
        }

        /// <summary>
        /// Down to the multiple of the model's alignment below it, and inside what the engine
        /// will take: a side the model cannot be asked for is not a side worth keeping.
        /// </summary>
        private int Fit(int side)
        {
            var step = Math.Max(1, alignment);
            var fitted = side / step * step;
            return Math.Max(Smallest, Math.Min(Largest, fitted));
        }
    }
}
