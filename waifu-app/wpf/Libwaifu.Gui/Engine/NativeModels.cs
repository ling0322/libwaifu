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
using System.IO;
using System.Linq;
using System.Windows.Threading;

namespace Libwaifu.Gui.Engine
{
    /// <summary>
    /// What the libraries say about models: the model manager's catalogue and defaults, read out
    /// of the JSON documents it answers with, and what waifu.dll says a model can do, asked of a
    /// <c>WaifuModel</c> one attribute at a time.
    /// </summary>
    public static class NativeModels
    {
        /// <summary>Every published model. A model that does two things is listed under both.</summary>
        public static List<ModelEntry> Catalog()
        {
            var found = new List<ModelEntry>();
            var text = Utf8.TakeFromModelManager(Native.waifu_modelmanager_catalog_json());
            if (text == null)
            {
                Log.Write("warn", "models", "no catalog: " + Library.ModelManagerLastError());
                return found;
            }

            foreach (var row in Json.Items(Json.Parse(text), "models"))
            {
                var kind = Json.Text(row, "kind");
                // A transcriber is for a task this app does not have; read as any other kind it
                // does not know, it would be offered as a picture model.
                if (kind == "transcription")
                {
                    continue;
                }

                found.Add(new ModelEntry
                {
                    Name = Json.Text(row, "name"),
                    FullName = Json.Text(row, "full_name") ?? Json.Text(row, "name"),
                    Category = CategoryNamed(kind),
                    Cached = Json.Flag(row, "cached"),
                    ManifestHere = Json.Flag(row, "manifest_here"),
                    BytesOnDisk = Json.Big(row, "bytes_on_disk"),
                });
            }

            // anima:turbo is what to try first, and the rest keep the order the library gave.
            return found
                .Select((entry, at) => new { entry, rank = entry.Name == "anima:turbo" ? 0 : 1, at })
                .OrderBy(each => each.rank).ThenBy(each => each.at)
                .Select(each => each.entry)
                .ToList();
        }

        /// <summary>The stand-in voice, which is in waifu.dll itself and has no manifest.</summary>
        public const string Tones = "tones";

        /// <summary>
        /// The manifest of a published model here -- fetched, or bundled -- or a manifest's own
        /// path, as it is: what waifu.dll is handed. Null where there is none here, with the
        /// reason in <see cref="Library.ModelManagerLastError"/>.
        /// </summary>
        public static string ManifestPath(string model)
        {
            if (model == Tones)
            {
                return model;
            }

            return Utf8.TakeFromModelManager(Native.waifu_modelmanager_manifest_path(Utf8.Bytes(model)));
        }

        /// <summary>What the catalogue calls a published model, or null for anything else.</summary>
        public static string PublishedFullName(string model)
        {
            var listed = Catalog().FirstOrDefault(entry => entry.Name == model);
            return listed == null ? null : listed.FullName;
        }

        /// <summary>
        /// Whether a load of <paramref name="model"/> has a download to do first: a published
        /// model not all here. The stand-in and a manifest on the disk have nothing to fetch.
        /// </summary>
        public static bool NeedsFetching(string model)
        {
            if (model == Tones || File.Exists(model))
            {
                return false;
            }

            var listed = Catalog().FirstOrDefault(entry => entry.Name == model);
            return listed != null && !listed.Cached;
        }

        /// <summary>
        /// What a model is, without reading it: what waifu.dll says of the manifest the model
        /// manager finds for it, whose weights need not be here. Null where there is no manifest
        /// here, or the library cannot say.
        /// </summary>
        public static ModelDescription Describe(string model)
        {
            var manifest = ManifestPath(model);
            if (manifest == null)
            {
                return null;
            }

            // Refused -- no manifest there, or one that cannot be read -- with the reason in
            // waifu_last_error, which is where Models.Describe looks for it.
            var handle = Native.waifu_model_open(Utf8.Bytes(manifest));
            if (handle == IntPtr.Zero)
            {
                return null;
            }

            try
            {
                // What the boxes start at is the model manager's; what the model can be asked
                // for is waifu.dll's. The two halves of one description.
                var defaults = Defaults(manifest);
                var description = new ModelDescription
                {
                    Name = model,
                    // One asked for by its published name is called what the catalogue calls it,
                    // and anything else by its manifest's file name.
                    FullName = PublishedFullName(model) ?? Path.GetFileName(manifest) ?? model,
                    Category = CategoryOf(handle),
                    OnDisk = Has(handle, Native.ModelAttribute.SizeOnDisk),
                    InMemory = false,
                };

                // A model that does more than one thing -- CosyVoice3 reads and converts -- is
                // described for each, so whichever page it is chosen on has what it needs. One
                // that is neither a voice nor a converter is described as a picture model, as it
                // always was; a picture it cannot draw is refused when the job loads it.
                if (description.Category == Category.Image)
                {
                    ReadPicture(handle, defaults, description);
                }

                if (Supports(handle, Native.ModelAttribute.SupportsTextToSpeech))
                {
                    ReadVoice(handle, defaults, description);
                }

                if (Supports(handle, Native.ModelAttribute.SupportsVoiceConversion))
                {
                    ReadConverter(handle, defaults, description);
                }

                return description;
            }
            finally
            {
                Native.waifu_model_free(handle);
            }
        }

        /// <summary>What waifu_modelmanager.dll says a model's boxes start at; empty where it cannot say.</summary>
        private static object Defaults(string manifest)
        {
            var text = Utf8.TakeFromModelManager(
                Native.waifu_modelmanager_defaults_json(Utf8.Bytes(manifest)));
            if (text == null)
            {
                Log.Write("warn", "models", manifest + ": " + Library.ModelManagerLastError());
                return new Dictionary<string, object>();
            }

            return Json.Parse(text);
        }

        /// <summary>
        /// Which page a model is for, which is what the first thing it does says: a voice, then a
        /// converter, and a picture model otherwise.
        /// </summary>
        private static Category CategoryOf(IntPtr model)
        {
            if (Supports(model, Native.ModelAttribute.SupportsTextToSpeech))
            {
                return Category.Speech;
            }

            if (Supports(model, Native.ModelAttribute.SupportsVoiceConversion))
            {
                return Category.Conversion;
            }

            return Category.Image;
        }

        // -- asking a WaifuModel -----------------------------------------------------------------

        /// <summary>Whether a SUPPORTS_ attribute is 1. Absent -- a library that does not know it -- is not.</summary>
        public static bool Supports(IntPtr model, Native.ModelAttribute attribute)
        {
            long value;
            return Native.waifu_model_get_int_attribute(model, attribute, out value) && value == 1;
        }

        /// <summary>Why a SUPPORTS_ attribute is 0, in words to show; null where nothing says.</summary>
        public static string WhyNot(IntPtr model, Native.ModelAttribute attribute)
        {
            return Utf8.Read(Native.waifu_model_get_string_attribute(model, attribute));
        }

        private static bool Has(IntPtr model, Native.ModelAttribute attribute)
        {
            long value;
            return Native.waifu_model_get_int_attribute(model, attribute, out value);
        }

        private static int Integer(IntPtr model, Native.ModelAttribute attribute, int fallback)
        {
            long value;
            return Native.waifu_model_get_int_attribute(model, attribute, out value) &&
                   value > 0 && value <= int.MaxValue
                ? (int)value
                : fallback;
        }

        private static void ReadPicture(IntPtr model, object node, ModelDescription description)
        {
            description.Steps = Json.Count(node, "steps", 20);
            description.Guidance = Json.Number(node, "guidance", 5.0);
            description.TakesGuidance = Json.Flag(node, "takes_guidance", true);
            description.Alignment = Integer(
                model, Native.ModelAttribute.ImageGenerationWidthHeightMultipleOf, 16);
            description.Prompt = Json.Text(node, "prompt");
            description.Avoid = Json.Text(node, "avoid");

            foreach (var size in Json.Items(node, "sizes"))
            {
                var pair = size as List<object>;
                if (pair != null && pair.Count == 2 && pair[0] is double && pair[1] is double)
                {
                    description.Sizes.Add(new KeyValuePair<int, int>(
                        (int)(double)pair[0], (int)(double)pair[1]));
                }
            }

            // The size the model itself suggests, where the list does not lead with it.
            var width = Json.Count(node, "width");
            var height = Json.Count(node, "height");
            if (width > 0 && height > 0)
            {
                var suggested = new KeyValuePair<int, int>(width, height);
                description.Sizes.Remove(suggested);
                description.Sizes.Insert(0, suggested);
            }

            // What it can be given an image for is said here rather than found out from a job
            // that fails, so the box is shown only where it does something.
            description.CanStartFromPicture = Supports(
                model, Native.ModelAttribute.SupportsImageGenerationFromStartImage);
            description.WhyNotFromPicture = description.CanStartFromPicture
                ? null
                : WhyNot(model, Native.ModelAttribute.SupportsImageGenerationFromStartImage);
        }

        private static void ReadVoice(IntPtr model, object node, ModelDescription description)
        {
            description.Speed = Json.Number(node, "speed", 1.0);
            description.Temperature = Json.Number(node, "temperature", 1.0);
            description.Rate = Integer(model, Native.ModelAttribute.SampleRate, 24000);
            description.TakesARecording = Supports(
                model, Native.ModelAttribute.SupportsTextToSpeechLikeReference);
            description.WhyNotARecording = description.TakesARecording
                ? null
                : WhyNot(model, Native.ModelAttribute.SupportsTextToSpeechLikeReference);

            // Whether it follows an instruction at all; what is offered is the app's own list.
            description.FollowsStyle = Supports(
                model, Native.ModelAttribute.SupportsTextToSpeechWithStyle);
        }

        private static void ReadConverter(IntPtr model, object node, ModelDescription description)
        {
            // A model that is a voice as well keeps its converter's defaults under "conversion".
            description.Steps = Json.Count(
                Json.Field(node, "conversion"), "steps", Json.Count(node, "steps", 30));
            description.Rate = Integer(model, Native.ModelAttribute.SampleRate, 24000);
            description.ConvertsStyle = Supports(
                model, Native.ModelAttribute.SupportsVoiceConversionOfStyle);
            description.WhyNotStyle = description.ConvertsStyle
                ? null
                : WhyNot(model, Native.ModelAttribute.SupportsVoiceConversionOfStyle);
        }

        public static Category CategoryNamed(string kind)
        {
            switch (kind)
            {
                case "speech": return Category.Speech;
                case "conversion": return Category.Conversion;
                default: return Category.Image;
            }
        }

        // -- where models are kept -------------------------------------------------------------

        public static string Directory()
        {
            return Utf8.TakeFromModelManager(Native.waifu_modelmanager_directory());
        }

        public static bool SetDirectory(string path)
        {
            return Native.waifu_modelmanager_set_directory(Utf8.Bytes(path)) == Native.ManagerStatus.Ok;
        }

        public static bool Remove(string name)
        {
            if (Native.waifu_modelmanager_remove(Utf8.Bytes(name)) == Native.ManagerStatus.Ok)
            {
                return true;
            }

            Log.Write("error", "models", name + ": " + Library.ModelManagerLastError());
            return false;
        }
    }

    /// <summary>
    /// A download, running on a thread of the library's. Progress and the end of it are handed
    /// to the dispatcher, because what reads them is the screen.
    /// </summary>
    public sealed class Fetch : IDisposable
    {
        private readonly Dispatcher dispatcher;
        private readonly Native.ManagerProgressCallback onProgress;
        private readonly Native.ManagerCompleteCallback onComplete;
        private IntPtr handle;
        private bool ended;

        private Fetch(Dispatcher dispatcher)
        {
            this.dispatcher = dispatcher;

            // Held in fields for as long as the library can call them.
            onProgress = Advanced;
            onComplete = Ended;
        }

        /// <summary>How far along, and the line to put under the bar.</summary>
        public event Action<double, string> Progress;

        /// <summary>Null when it worked, the reason when it did not, and "" when cancelled.</summary>
        public event Action<string> Finished;

        /// <summary>
        /// Starts fetching a model. Null where the library refused at the call -- which it does
        /// without ever calling back, so nothing is left waiting.
        /// </summary>
        public static Fetch Start(string model)
        {
            var fetch = new Fetch(Dispatcher.CurrentDispatcher);
            fetch.handle = Native.waifu_modelmanager_fetch_async(
                Utf8.Bytes(model), IntPtr.Zero, fetch.onProgress, fetch.onComplete);

            if (fetch.handle == IntPtr.Zero)
            {
                Log.Write("error", "models", model + ": " + Library.ModelManagerLastError());
                return null;
            }

            return fetch;
        }

        /// <summary>
        /// Takes effect between files: what arrived stays on the disk, and the next fetch carries
        /// on from there.
        /// </summary>
        public void Cancel()
        {
            if (handle != IntPtr.Zero)
            {
                Native.waifu_modelmanager_fetch_cancel(handle);
            }
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero)
            {
                // Cancels first if it has not had its completion, and waits for the thread.
                Native.waifu_modelmanager_fetch_free(handle);
                handle = IntPtr.Zero;
            }
        }

        private void Advanced(IntPtr user, ref Native.ManagerProgressEvent e)
        {
            var fraction = e.Fraction;
            var words = Utf8.Read(e.Words);
            var file = Utf8.Read(e.File);
            var line = words ?? file;

            dispatcher.BeginInvoke((Action)delegate
            {
                var handler = Progress;
                if (handler != null)
                {
                    handler(fraction, line);
                }
            });
        }

        private void Ended(IntPtr user, ref Native.ManagerCompletionEvent e)
        {
            var code = e.Status.Code;
            var message = Utf8.Read(e.Status.Message);

            dispatcher.BeginInvoke((Action)delegate
            {
                if (ended)
                {
                    return;
                }

                ended = true;
                var handler = Finished;
                if (handler != null)
                {
                    handler(code == Native.ManagerStatus.Ok
                        ? null
                        : (code == Native.ManagerStatus.Cancelled ? string.Empty : message ?? "it failed"));
                }

                Dispose();
            });
        }
    }
}
