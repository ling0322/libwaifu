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
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Threading;
using Libwaifu.Gui.Engine;

namespace Libwaifu.Gui.ViewModels
{
    /// <summary>One model's row: what it is, how much of it is here, and the one button for it.</summary>
    public class ModelRowViewModel : Observable
    {
        private readonly ModelEntry entry;
        private readonly ModelManagerViewModel manager;
        private Fetch fetch;
        private DispatcherTimer pretending;
        private double got = -1;
        private string line;

        public ModelRowViewModel(ModelManagerViewModel manager, ModelEntry entry)
        {
            this.manager = manager;
            this.entry = entry;

            Download = new Command(Begin, () => !manager.Busy);
            Cancel = new Command(Stop);
            Remove = new Command(Delete, () => !manager.Busy);
        }

        public string Name
        {
            get { return entry.Name; }
        }

        public string FullName
        {
            get { return entry.FullName; }
        }

        public Command Download { get; private set; }

        public Command Cancel { get; private set; }

        public Command Remove { get; private set; }

        /// <summary>A green pill beside the name, for the one model that is on the device.</summary>
        public bool Loaded
        {
            get { return manager.LoadedModel == entry.Name; }
        }

        public bool Downloading
        {
            get { return fetch != null || pretending != null; }
        }

        public bool Here
        {
            get { return !Downloading && entry.Cached; }
        }

        public bool NotHere
        {
            get { return !Downloading && !entry.Cached; }
        }

        public double Progress
        {
            get { return got * 100; }
        }

        /// <summary>
        /// Whether the bar can say how far along it is. A download knows once the first file's
        /// size has come back, and not before.
        /// </summary>
        public bool Measured
        {
            get { return got >= 0; }
        }

        public string ProgressLine
        {
            get { return line; }
        }

        /// <summary>How much room it takes, or how much of it arrived before something stopped.</summary>
        public string Size
        {
            get
            {
                if (entry.Cached)
                {
                    return Converters.Bytes.Format(entry.BytesOnDisk);
                }

                return entry.BytesOnDisk > 0
                    ? Converters.Bytes.Format(entry.BytesOnDisk) + " of it here"
                    : null;
            }
        }

        public void Refresh()
        {
            foreach (var name in new[] { "Loaded", "Here", "NotHere", "Downloading", "Size" })
            {
                Raise(name);
            }

            Download.Refresh();
            Remove.Refresh();
        }

        private void Begin()
        {
            if (fetch != null || pretending != null)
            {
                return;
            }

            got = -1;
            line = "starting";

            if (Models.Live)
            {
                fetch = Fetch.Start(entry.Name);
                if (fetch == null)
                {
                    line = Library.LastError();
                    Raise("ProgressLine");
                    return;
                }

                fetch.Progress += Along;
                fetch.Finished += Over;
            }
            else
            {
                Pretend();
            }

            Refresh();
        }

        private void Along(double fraction, string words)
        {
            got = fraction;
            line = words;
            Raise("Progress");
            Raise("Measured");
            Raise("ProgressLine");
        }

        private void Over(string trouble)
        {
            fetch = null;

            // "" is what a cancelled fetch says, and a cancelled fetch is not a failure: what
            // arrived stays on the disk and the next one carries on from there.
            if (!string.IsNullOrEmpty(trouble))
            {
                manager.Trouble = entry.Name + ": " + trouble;
            }

            Refresh();
            manager.ModelsChanged();
        }

        /// <summary>The stand-in's download, for a screen with no library behind it.</summary>
        private void Pretend()
        {
            got = (double)entry.BytesOnDisk / Math.Max(1, entry.BytesTotal);
            pretending = new DispatcherTimer(DispatcherPriority.Background)
            {
                Interval = TimeSpan.FromMilliseconds(120),
            };

            pretending.Tick += delegate
            {
                got = Math.Min(1.0, got + 0.03);
                entry.BytesOnDisk = (long)(entry.BytesTotal * got);
                Along(got,
                    Converters.Bytes.Format(entry.BytesOnDisk) + " of " +
                    Converters.Bytes.Format(entry.BytesTotal));

                if (got >= 1.0)
                {
                    entry.Cached = true;
                    Stop();
                }
            };

            pretending.Start();
        }

        /// <summary>
        /// Cancelling takes effect between files: what arrived stays on the disk and the next
        /// fetch carries on from there.
        /// </summary>
        private void Stop()
        {
            if (fetch != null)
            {
                // The library answers with its own completion, which is where the row is let go.
                fetch.Cancel();
                return;
            }

            if (pretending == null)
            {
                return;
            }

            pretending.Stop();
            pretending = null;
            Refresh();
            manager.ModelsChanged();
        }

        private void Delete()
        {
            var answer = MessageBox.Show(
                Converters.Bytes.Format(entry.BytesOnDisk) +
                " on the disk. It is downloaded again the next time it is used.",
                "Delete " + entry.Name + "?",
                MessageBoxButton.OKCancel,
                MessageBoxImage.None);

            if (answer != MessageBoxResult.OK)
            {
                return;
            }

            if (!Models.Remove(entry))
            {
                manager.Trouble = entry.Name + ": " + Library.LastError();
            }

            Refresh();
            manager.ModelsChanged();
        }
    }

    /// <summary>One of the three headings the models are listed under.</summary>
    public class ModelGroupViewModel
    {
        public ModelGroupViewModel(string title)
        {
            Title = title;
            Rows = new ObservableCollection<ModelRowViewModel>();
        }

        public string Title { get; private set; }

        public ObservableCollection<ModelRowViewModel> Rows { get; private set; }
    }

    /// <summary>The Model Manager page: where models are kept, and what is in there.</summary>
    public class ModelManagerViewModel : Observable
    {
        private readonly MainViewModel main;
        private string trouble;

        public ModelManagerViewModel(MainViewModel main)
        {
            this.main = main;

            if (Places.Models == null)
            {
                Places.Models = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    ".cache", "libwaifu", "models");
            }

            Groups = new ObservableCollection<ModelGroupViewModel>();
            Reveal = new Command(delegate
            {
                Places.Ensure(Models.Directory);
                Process.Start("explorer.exe", "\"" + Models.Directory + "\"");
            });

            Change = new Command(Move);
            Fill();
        }

        public ObservableCollection<ModelGroupViewModel> Groups { get; private set; }

        public Command Reveal { get; private set; }

        public Command Change { get; private set; }

        public string Directory
        {
            get { return Models.Directory ?? "the default folder"; }
        }

        /// <summary>A red line at the top of the page: what a download or a delete came to.</summary>
        public string Trouble
        {
            get { return trouble; }
            set { Set(ref trouble, value); Raise("HasTrouble"); }
        }

        public bool HasTrouble
        {
            get { return !string.IsNullOrEmpty(trouble); }
        }

        public string LoadedModel
        {
            get { return main.Engine.LoadedModel; }
        }

        /// <summary>Downloading and deleting are off while a job is running.</summary>
        public bool Busy
        {
            get { return main.Busy; }
        }

        public void Reread()
        {
            foreach (var group in Groups)
            {
                foreach (var row in group.Rows)
                {
                    row.Refresh();
                }
            }

            Raise("Busy");
        }

        /// <summary>
        /// What a download or a delete changes is not only this page: the Model popup groups
        /// models by whether they are here, so it is rebuilt too.
        /// </summary>
        public void ModelsChanged()
        {
            // What a download or a delete changed is the library's to say, so the catalog is read
            // again before anything is asked of it.
            Models.Reread();
            Reread();
            foreach (var task in main.Pages)
            {
                task.RefreshModelCard();
            }
        }

        private void Fill()
        {
            Groups.Clear();
            Add("Pictures", Category.Image);
            Add("Voices", Category.Speech);
            Add("Voice conversion", Category.Conversion);
        }

        private void Add(string title, Category category)
        {
            var group = new ModelGroupViewModel(title);
            foreach (var entry in Models.For(category))
            {
                group.Rows.Add(new ModelRowViewModel(this, entry));
            }

            if (group.Rows.Count > 0)
            {
                Groups.Add(group);
            }
        }

        private void Move()
        {
            using (var dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Where models are kept",
                SelectedPath = Places.Models,
            })
            {
                if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    if (!Models.SetDirectory(dialog.SelectedPath))
                    {
                        Trouble = Library.LastError();
                    }

                    Models.Reread();
                    Raise("Directory");
                    Reread();
                }
            }
        }
    }
}
