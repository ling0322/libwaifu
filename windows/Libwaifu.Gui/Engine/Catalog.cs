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

using System.Collections.Generic;
using System.Linq;

namespace Libwaifu.Gui.Engine
{
    /// <summary>One model the library knows by name, as the catalog describes it.</summary>
    public class ModelEntry
    {
        public string Name { get; set; }
        public string FullName { get; set; }
        public Category Category { get; set; }

        /// <summary>Every package of it is on the disk, so the first job reads rather than fetches.</summary>
        public bool Cached { get; set; }

        /// <summary>
        /// Its manifest is here, with or without its weights -- which is what makes a model able
        /// to say what it suggests before any of it has been downloaded.
        /// </summary>
        public bool ManifestHere { get; set; }

        /// <summary>
        /// How much of it is here. For a model fetched partway that is not all of it, which is
        /// what "N of it here" means in the Model Manager.
        /// </summary>
        public long BytesOnDisk { get; set; }

        /// <summary>
        /// How big it is in all. The library does not say for a model that is not here, so this
        /// is zero outside the stand-in catalog.
        /// </summary>
        public long BytesTotal { get; set; }
    }

    /// <summary>
    /// What a model says about itself: the numbers to start from and the things it cannot do.
    /// </summary>
    /// <remarks>
    /// This is what the real screen gets back from <c>waifu_modelmanager_describe_json</c>, and
    /// choosing a model asks for it and nothing else -- reading the description is not loading
    /// the model, and no job runs until the Generate button is pressed.
    /// </remarks>
    public class ModelDescription
    {
        public string Name { get; set; }
        public string FullName { get; set; }
        public Category Category { get; set; }

        /// <summary>Already fetched. One that is not is fetched by the first job that needs it.</summary>
        public bool OnDisk { get; set; }

        /// <summary>Already read onto the device this engine is on.</summary>
        public bool InMemory { get; set; }

        // Pictures.
        public bool TakesGuidance { get; set; }
        public bool CanStartFromPicture { get; set; }
        public string WhyNotFromPicture { get; set; }
        public List<KeyValuePair<int, int>> Sizes { get; set; }
        public int Alignment { get; set; }
        public int Steps { get; set; }
        public double Guidance { get; set; }
        public string Prompt { get; set; }
        public string Avoid { get; set; }

        // Voices.
        public bool TakesARecording { get; set; }
        public string WhyNotARecording { get; set; }
        public string NotAVoiceBecause { get; set; }
        public List<KeyValuePair<string, string>> Styles { get; set; }
        public double Speed { get; set; }
        public double Temperature { get; set; }

        /// <summary>How many samples a second what it makes has.</summary>
        public int Rate { get; set; }

        // Converting one voice into another.
        public bool ConvertsStyle { get; set; }

        public string WhyNotStyle { get; set; }

        public ModelDescription()
        {
            Sizes = new List<KeyValuePair<int, int>>();
            Styles = new List<KeyValuePair<string, string>>();
            Alignment = 16;
            Steps = 10;
            Speed = 1.0;
            Temperature = 1.0;
            Rate = 24000;
        }
    }

    /// <summary>
    /// The models this build knows, and what each of them says about itself.
    /// </summary>
    /// <remarks>
    /// Standing in for the library's catalog and describe calls. The names and full names are the
    /// published ones so that the screen reads as it will when the two are joined up; what is
    /// made up here is only how much of each is on this disk.
    /// </remarks>
    public static class Catalog
    {
        private static readonly ModelEntry[] Known =
        {
            Picture("anima:turbo", "Anima Turbo v1.1", 5_480_000_000),
            Picture("anima:miaomiao", "MiaoMiao Harem Anima v1.6", 5_480_000_000),
            Picture("krea2:turbo", "Krea 2 Turbo", 12_300_000_000),
            Picture("krea2:turbo-fp8", "Krea 2 Turbo (fp8)", 6_900_000_000),
            Picture("qwen-image:2.1", "Qwen-Image 2.1", 15_990_000_000),
            Picture("qwen-image:2.1-fp8", "Qwen-Image 2.1 (fp8)", 15_990_000_000),
            Picture("sdxl:base", "Stable Diffusion XL Base 1.0", 6_940_000_000),
            Picture("sdxl:illust", "Illustrious XL v2.0-STABLE", 6_940_000_000),
            Picture("sdxl:noob", "NoobAI XL v1.1", 7_110_000_000),
            Picture("sdxl:obsession", "One Obsession v24", 6_940_000_000),
            Picture("sdxl:wai", "WAI Illustrious v17", 14_220_000_000),
            new ModelEntry { Name = "cosyvoice", FullName = "Fun-CosyVoice3 0.5B", Category = Category.Speech, BytesTotal = 1_980_000_000 },
            new ModelEntry { Name = "indextts", FullName = "IndexTTS 2.5", Category = Category.Speech, BytesTotal = 2_400_000_000 },
            new ModelEntry { Name = "seedvc", FullName = "Seed-VC v2", Category = Category.Conversion, BytesTotal = 1_200_000_000 },
        };

        private static ModelEntry Picture(string name, string fullName, long bytes)
        {
            return new ModelEntry
            {
                Name = name,
                FullName = fullName,
                Category = Category.Image,
                BytesTotal = bytes,
                ManifestHere = true,
            };
        }

        /// <summary>
        /// Every published model, in the order the picker shows them: anima:turbo first, the rest
        /// in the order the library gave.
        /// </summary>
        public static List<ModelEntry> All()
        {
            return Known.ToList();
        }

        public static List<ModelEntry> For(Category category)
        {
            return Known.Where(entry => entry.Category == category).ToList();
        }

        public static ModelEntry Find(string name)
        {
            return Known.FirstOrDefault(entry => entry.Name == name);
        }

        /// <summary>What the model says about itself, which is read when one is chosen.</summary>
        public static ModelDescription Describe(string name)
        {
            var entry = Find(name);
            if (entry == null)
            {
                return null;
            }

            var description = new ModelDescription
            {
                Name = entry.Name,
                FullName = entry.FullName,
                Category = entry.Category,
            };

            switch (entry.Category)
            {
                case Category.Image:
                    DescribePicture(entry, description);
                    break;
                case Category.Speech:
                    DescribeVoice(entry, description);
                    break;
                default:
                    description.ConvertsStyle = true;
                    description.Steps = 30;
                    break;
            }

            return description;
        }

        private static void DescribePicture(ModelEntry entry, ModelDescription description)
        {
            var anima = entry.Name.StartsWith("anima:");
            var turbo = entry.Name.Contains("turbo");

            description.Sizes = new List<KeyValuePair<int, int>>
            {
                new KeyValuePair<int, int>(1024, 1024),
                new KeyValuePair<int, int>(832, 1216),
                new KeyValuePair<int, int>(1216, 832),
                new KeyValuePair<int, int>(1152, 896),
                new KeyValuePair<int, int>(896, 1152),
            };

            // A turbo release is distilled for few steps at no guidance; asked for thirty and
            // five it gives back a burnt, over-contrasted picture. The Away from box and the CFG
            // slider are not shown at all for one, because there is nothing for them to do.
            description.Alignment = entry.Name.StartsWith("sdxl:") ? 32 : 16;
            description.TakesGuidance = !turbo;
            description.Steps = turbo ? 10 : 28;
            description.Guidance = turbo ? 1.0 : 5.0;
            description.Prompt = anima
                ? "masterpiece, best quality, score_7, safe, 1girl, hatsune miku, vocaloid, solo, twintails, smile, looking at viewer"
                : "masterpiece, best quality, 1girl, solo, looking at viewer";
            description.Avoid = description.TakesGuidance
                ? "worst quality, low quality, artist name, blurry, jpeg artifacts, chromatic aberration"
                : null;

            // Anima's package carries an image encoder, but the layer that reads one is not
            // written yet, so it says why rather than offering a box to drop a picture on.
            description.CanStartFromPicture = !anima;
            description.WhyNotFromPicture = anima
                ? "Anima cannot start from a picture yet: its package carries an image encoder, but the layer that reads one is not written"
                : null;
        }

        private static void DescribeVoice(ModelEntry entry, ModelDescription description)
        {
            description.Speed = 1.0;
            description.Temperature = 1.0;
            description.TakesARecording = true;
            description.Styles = entry.Name == "cosyvoice"
                ? new List<KeyValuePair<string, string>>
                {
                    new KeyValuePair<string, string>("Cheerful", "Say it brightly, as if sharing good news."),
                    new KeyValuePair<string, string>("Calm", "Say it slowly and evenly, with long pauses."),
                    new KeyValuePair<string, string>("Whisper", "Whisper it."),
                }
                : new List<KeyValuePair<string, string>>();
        }
    }
}
