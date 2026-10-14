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
    /// <summary>
    /// The models the screen offers, from the library where it is here and from the stand-in
    /// catalog where it is not.
    /// </summary>
    /// <remarks>
    /// One list of <see cref="ModelEntry"/> objects is kept and updated in place rather than
    /// rebuilt, because the Model Manager's rows hold on to them: a row that is downloading has
    /// to go on pointing at the same entry when the catalog is read again underneath it.
    /// </remarks>
    public static class Models
    {
        private static readonly List<ModelEntry> Known = new List<ModelEntry>();
        private static bool read;

        /// <summary>Whether these came from the library rather than from the stand-in.</summary>
        public static bool Live
        {
            get { return Library.Available; }
        }

        public static List<ModelEntry> All()
        {
            if (!read)
            {
                Reread();
            }

            return Known.ToList();
        }

        public static List<ModelEntry> For(Category category)
        {
            return All().Where(entry => entry.Category == category).ToList();
        }

        public static ModelEntry Find(string name)
        {
            return All().FirstOrDefault(entry => entry.Name == name);
        }

        /// <summary>
        /// Reads the catalog again -- after a download, a delete, or a change of folder -- and
        /// writes what changed into the entries that are already out there.
        /// </summary>
        public static void Reread()
        {
            read = true;
            var fresh = Live ? NativeModels.Catalog() : Catalog.All();

            // The stand-in hands back the same objects every time, and they are the ones it
            // mutates; nothing to merge.
            if (!Live)
            {
                Known.Clear();
                Known.AddRange(fresh);
                return;
            }

            foreach (var entry in fresh)
            {
                var already = Known.FirstOrDefault(
                    each => each.Name == entry.Name && each.Category == entry.Category);

                if (already == null)
                {
                    Known.Add(entry);
                    continue;
                }

                already.FullName = entry.FullName;
                already.Cached = entry.Cached;
                already.ManifestHere = entry.ManifestHere;
                already.BytesOnDisk = entry.BytesOnDisk;
            }

            Known.RemoveAll(each => !fresh.Any(
                entry => entry.Name == each.Name && entry.Category == each.Category));
        }

        /// <summary>
        /// What a model says about itself. Null, with <see cref="Trouble"/> set, where the
        /// library will not have it -- a name nobody published, or a manifest it cannot read.
        /// </summary>
        public static ModelDescription Describe(string name)
        {
            Trouble = null;

            if (!Live)
            {
                var described = Catalog.Describe(name);
                if (described == null)
                {
                    Trouble = name + ": nothing in the catalog goes by that name.";
                }

                return described;
            }

            var answer = NativeModels.Describe(name);
            if (answer == null)
            {
                Trouble = name + ": " + Library.LastError();
            }

            return answer;
        }

        /// <summary>Why the last describe failed, for the Model card's footnote.</summary>
        public static string Trouble { get; private set; }

        public static string Directory
        {
            get
            {
                if (Live)
                {
                    var folder = NativeModels.Directory();
                    if (!string.IsNullOrEmpty(folder))
                    {
                        return folder;
                    }
                }

                return Places.Models;
            }
        }

        public static bool SetDirectory(string path)
        {
            Places.Models = path;
            return !Live || NativeModels.SetDirectory(path);
        }

        public static bool Remove(ModelEntry entry)
        {
            if (Live)
            {
                if (!NativeModels.Remove(entry.Name))
                {
                    return false;
                }
            }
            else
            {
                entry.Cached = false;
                entry.BytesOnDisk = 0;
            }

            Reread();
            return true;
        }
    }
}
