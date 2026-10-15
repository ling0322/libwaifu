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
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using Libwaifu.Gui.Engine;

using Lib = Libwaifu.Gui.Engine;

namespace Libwaifu.Gui.ViewModels
{
    /// <summary>One line of the Model popup. Most are models; two are headings, one is a rule.</summary>
    public class ModelChoice
    {
        public enum Sort
        {
            Model,
            Heading,
            Rule,
            Browse,
        }

        public Sort Kind { get; set; }
        public string Name { get; set; }
        public string Display { get; set; }

        public bool IsModel { get { return Kind == Sort.Model; } }
        public bool IsHeading { get { return Kind == Sort.Heading; } }
        public bool IsRule { get { return Kind == Sort.Rule; } }

        /// <summary>A heading and a rule are shown but cannot be picked.</summary>
        public bool Pickable { get { return Kind != Sort.Heading && Kind != Sort.Rule; } }

        public static ModelChoice Model(string name, string display)
        {
            return new ModelChoice { Kind = Sort.Model, Name = name, Display = display };
        }

        public static ModelChoice Heading(string text)
        {
            return new ModelChoice { Kind = Sort.Heading, Display = text };
        }

        public static ModelChoice Rule()
        {
            return new ModelChoice { Kind = Sort.Rule, Display = string.Empty };
        }

        public static ModelChoice Browse()
        {
            return new ModelChoice { Kind = Sort.Browse, Display = "Choose a Manifest..." };
        }
    }

    /// <summary>
    /// A page: the model card above its settings, the prompt and the button in the middle, and
    /// everything the task has ever produced underneath.
    /// </summary>
    public abstract class TaskViewModel : Observable
    {
        protected readonly MainViewModel Main;

        private ModelChoice chosen;
        private ModelDescription description;
        private string describeFailed;
        private ResultViewModel selected;
        private string progressLine;
        private double progressValue;
        private bool indeterminate;
        private bool showProgress;
        private string notice;
        private bool noticeIsError;
        private bool quiet;

        protected TaskViewModel(MainViewModel main, TaskKind kind)
        {
            Main = main;
            Kind = kind;
            Category = Lib.Tasks.CategoryOf(kind);
            Name = Lib.Tasks.Name(kind);

            ModelChoices = new ObservableCollection<ModelChoice>();
            Shown = new ListCollectionView(main.Results)
            {
                Filter = item =>
                {
                    var result = item as ResultViewModel;
                    return result != null && result.Category == Category;
                },
            };

            Go = new Command(OnGo);
            Generate = new Command(Start, () => CanGo);
            Stop = new Command(() => Main.CancelFor(this), () => Main.IsRunning(this) || Main.IsQueued(this));
            Save = new Command(OnSave, () => Selected != null);
            Reuse = new Command(OnReuse, () => Selected != null);
            Delete = new Command(OnDelete, () => Selected != null);
            CopyParameters = new Command(delegate
            {
                if (Selected != null && Selected.Parameters != null)
                {
                    Clipboard.SetText(Selected.Parameters);
                }
            });

            CopyPicture = new Command(delegate
            {
                if (Selected != null && Selected.Picture != null)
                {
                    Clipboard.SetImage(Selected.Picture);
                }
            });

            RebuildChoices();
            var remembered = main.Settings.ModelFor(Category);
            if (remembered != null && Models.Find(remembered) != null)
            {
                Adopt(remembered, false);
            }

            Selected = Shown.Cast<ResultViewModel>().FirstOrDefault();
            RefreshModelCard();
        }

        public TaskKind Kind { get; private set; }

        public Category Category { get; private set; }

        public string Name { get; private set; }

        /// <summary>The 20x20 square in the sidebar, which is one colour per task.</summary>
        public Geometry Icon
        {
            get { return (Geometry)Application.Current.Resources[IconKey]; }
        }

        public Brush IconBrush
        {
            get { return (Brush)Application.Current.Resources[BrushKey]; }
        }

        protected abstract string IconKey { get; }

        protected abstract string BrushKey { get; }

        // -- the model card --------------------------------------------------------------------

        public ObservableCollection<ModelChoice> ModelChoices { get; private set; }

        public ModelDescription Description
        {
            get { return description; }
        }

        public ModelEntry ChosenModel
        {
            get { return chosen == null || chosen.Name == null ? null : Models.Find(chosen.Name); }
        }

        public ModelChoice SelectedChoice
        {
            get { return chosen; }
            set
            {
                if (value == null || ReferenceEquals(value, chosen))
                {
                    return;
                }

                if (value.Kind == ModelChoice.Sort.Browse)
                {
                    BrowseForManifest();
                    return;
                }

                if (!value.IsModel)
                {
                    return;
                }

                Adopt(value.Name, true);
            }
        }

        public string[] Devices
        {
            get { return Lib.Devices.All; }
        }

        public string Device
        {
            get { return Main.Device; }
            set { Main.Device = value; }
        }

        public string ModelFootnote { get; private set; }

        public bool ModelFootnoteIsError { get; private set; }

        /// <summary>Everything in the settings column is off while a job is running or waiting.</summary>
        public bool SettingsEnabled
        {
            get { return !Main.Busy; }
        }

        // -- the big button ---------------------------------------------------------------------

        public Command Go { get; private set; }

        public Command Generate { get; private set; }

        public Command Stop { get; private set; }

        public Command Save { get; private set; }

        public Command Reuse { get; private set; }

        public Command Delete { get; private set; }

        public Command CopyParameters { get; private set; }

        public Command CopyPicture { get; private set; }

        public bool IsStopping
        {
            get { return quiet; }
        }

        /// <summary>True once a job of this task's is running or waiting, which is when the
        /// button turns red.</summary>
        public bool Working
        {
            get { return Main.IsRunning(this) || Main.IsQueued(this); }
        }

        public string ActionWord
        {
            get { return Working ? "Cancel" : Lib.Tasks.Verb(Kind); }
        }

        public abstract bool CanGo { get; }

        /// <summary>
        /// Whether the big button does anything. Stopping a run always does; starting one needs
        /// everything the task cannot go without.
        /// </summary>
        public bool GoEnabled
        {
            get { return Working || CanGo; }
        }

        /// <summary>The settings as they stand, which is what a job is.</summary>
        protected abstract JobRequest Build();

        protected abstract void AdoptDescription(ModelDescription model);

        /// <summary>Put a finished result's settings back into the form, seed and all.</summary>
        protected abstract void ReuseFrom(ResultRecord record);

        /// <summary>
        /// Another result is showing. A picture needs nothing done; a clip has to be handed to
        /// the player under it.
        /// </summary>
        protected virtual void SelectionChanged()
        {
        }

        // -- what came out ------------------------------------------------------------------------

        public ListCollectionView Shown { get; private set; }

        public ResultViewModel Selected
        {
            get { return selected; }
            set
            {
                if (Set(ref selected, value))
                {
                    Save.Refresh();
                    Reuse.Refresh();
                    Delete.Refresh();
                    Raise("HasResult");
                    SelectionChanged();
                }
            }
        }

        public bool HasResult
        {
            get { return selected != null; }
        }

        public string NothingYet
        {
            get { return Lib.Tasks.NothingYet(Category); }
        }

        // -- how far along ------------------------------------------------------------------------

        public bool ShowProgress
        {
            get { return showProgress; }
            private set { Set(ref showProgress, value); }
        }

        public bool Indeterminate
        {
            get { return indeterminate; }
            private set { Set(ref indeterminate, value); }
        }

        public double ProgressValue
        {
            get { return progressValue; }
            private set { Set(ref progressValue, value); }
        }

        public string ProgressLine
        {
            get { return progressLine; }
            private set { Set(ref progressLine, value); }
        }

        public string Notice
        {
            get { return notice; }
            private set { Set(ref notice, value); Raise("ShowNotice"); }
        }

        public bool NoticeIsError
        {
            get { return noticeIsError; }
            private set { Set(ref noticeIsError, value); }
        }

        public bool ShowNotice
        {
            get { return !string.IsNullOrEmpty(notice); }
        }

        // -- what the window calls -----------------------------------------------------------------

        public void JobStarted()
        {
            quiet = false;
            Notice = null;
            ShowProgress = true;
            Indeterminate = true;
            ProgressLine = "starting";
            BusyChanged();
        }

        public void JobQueued()
        {
            ShowProgress = true;
            Indeterminate = true;
            ProgressLine = "waiting for the job before it";
            BusyChanged();
        }

        public void JobDequeued()
        {
            ShowProgress = false;
            ProgressLine = null;
            BusyChanged();
        }

        public void Stopping()
        {
            quiet = true;
            ProgressLine = "stopping after this step...";
        }

        public void Advanced(EngineEvent e)
        {
            ShowProgress = true;
            Indeterminate = !e.Fraction.HasValue;
            ProgressValue = e.Fraction.HasValue ? e.Fraction.Value * 100 : 0;

            if (quiet)
            {
                ProgressLine = "stopping after this step...";
                return;
            }

            // Three fields joined by three spaces, and whatever is missing is left out: the line
            // reads "drawing   52%   4.1s", or just "fetching   2.0s" where nothing knows how far.
            var fields = new List<string> { e.Line };
            if (e.Fraction.HasValue)
            {
                fields.Add(((int)Math.Round(e.Fraction.Value * 100))
                    .ToString(CultureInfo.InvariantCulture) + "%");
            }

            fields.Add(e.Seconds.ToString("0.0", CultureInfo.InvariantCulture) + "s");
            ProgressLine = string.Join("   ", fields.ToArray());
        }

        public void JobEnded(JobOutcome outcome)
        {
            ShowProgress = false;
            quiet = false;

            if (outcome.Ending == Ending.Cancelled)
            {
                NoticeIsError = false;
                Notice = "stopped where it was";
            }
            else if (outcome.Ending == Ending.Failed)
            {
                NoticeIsError = true;
                Notice = outcome.Error;
            }

            RefreshModelCard();
            BusyChanged();
        }

        public void Select(ResultViewModel result)
        {
            Shown.Refresh();
            Selected = result;
        }

        public void BusyChanged()
        {
            Raise("SettingsEnabled");
            Raise("Working");
            Raise("ActionWord");
            Raise("GoEnabled");
            Generate.Refresh();
            Stop.Refresh();
            Go.Refresh();
        }

        /// <summary>
        /// The footnote under the two popups, which is the one place that says whether anything
        /// has been downloaded or read onto a device yet.
        /// </summary>
        public void RefreshModelCard()
        {
            var entry = ChosenModel;
            ModelFootnoteIsError = false;

            if (describeFailed != null)
            {
                ModelFootnoteIsError = true;
                ModelFootnote = describeFailed;
            }
            else if (entry == null)
            {
                ModelFootnote = "Choose a model for " + Name + ".";
            }
            else if (Main.Engine.LoadedModel == entry.Name && Main.Engine.LoadedDevice == Device)
            {
                ModelFootnote = "Loaded on " + Device + ".";
            }
            else if (entry.Cached)
            {
                ModelFootnote = "Downloaded. Read onto the device by the first job.";
            }
            else
            {
                ModelFootnote = "Not downloaded yet: the first job fetches it.";
            }

            ModelFootnote += Tight(entry);

            Raise("ModelFootnote");
            Raise("ModelFootnoteIsError");
            Raise("Device");
            Raise("CanGo");
            Raise("GoEnabled");
            Generate.Refresh();
            Go.Refresh();
        }

        /// <summary>
        /// A word about a model that very nearly fills the card. What is on the disk is close to
        /// what the weights take in memory, and what is left over has to hold the picture being
        /// drawn -- so a model at four fifths of the card is one a large size will not fit beside.
        /// </summary>
        private string Tight(ModelEntry entry)
        {
            // The machine is read after the pages are made, so the first time through there is
            // nothing to compare against yet; the reading raises this again when it arrives.
            var card = Main.Machine == null ? 0 : Main.Machine.VramTotal;
            if (entry == null || card == 0 || entry.BytesOnDisk <= 0 ||
                (ulong)entry.BytesOnDisk * 4 < card * 3)
            {
                return string.Empty;
            }

            return " It is " + Converters.Bytes.Format(entry.BytesOnDisk) + " and the card has " +
                   Converters.Bytes.Format(card) + ": a large size may stop for want of memory.";
        }

        protected void GoChanged()
        {
            Generate.Refresh();
            Go.Refresh();
            Raise("CanGo");
            Raise("GoEnabled");
        }

        // -- the model popup ------------------------------------------------------------------------

        private void RebuildChoices()
        {
            var keep = chosen;
            ModelChoices.Clear();

            if (keep == null || keep.Name == null)
            {
                ModelChoices.Add(ModelChoice.Model(null, "None"));
            }

            var mine = Models.For(Category);
            var here = mine.Where(entry => entry.Cached).ToList();
            var elsewhere = mine.Where(entry => !entry.Cached).ToList();

            if (here.Count > 0)
            {
                ModelChoices.Add(ModelChoice.Heading("Downloaded"));
                foreach (var entry in here)
                {
                    ModelChoices.Add(ModelChoice.Model(entry.Name, entry.Name));
                }
            }

            if (elsewhere.Count > 0)
            {
                ModelChoices.Add(ModelChoice.Heading("Not downloaded"));
                foreach (var entry in elsewhere)
                {
                    ModelChoices.Add(ModelChoice.Model(entry.Name, entry.Name));
                }
            }

            ModelChoices.Add(ModelChoice.Rule());
            ModelChoices.Add(ModelChoice.Browse());

            chosen = ModelChoices.FirstOrDefault(
                choice => choice.IsModel && choice.Name == (keep == null ? null : keep.Name));
            Raise("SelectedChoice");
        }

        /// <summary>
        /// Choosing a model reads what it says about itself and fills the form in from that. It
        /// does not download it and does not load it: the first job does both.
        /// </summary>
        private void Adopt(string name, bool remember)
        {
            description = Models.Describe(name);
            describeFailed = description == null ? Models.Trouble : null;

            if (description != null)
            {
                AdoptDescription(description);
            }

            if (remember)
            {
                Main.Settings.SetModelFor(Category, name);
                Main.Settings.Save();
            }

            RebuildChoices();
            chosen = ModelChoices.FirstOrDefault(choice => choice.IsModel && choice.Name == name);
            Raise("SelectedChoice");
            Raise("Description");
            RefreshModelCard();
            Main.Reread();
            GoChanged();
        }

        /// <summary>Choose a model by name, which is what <c>-m</c> on the command line does.</summary>
        public void ChooseModel(string name)
        {
            if (Models.Find(name) != null)
            {
                Adopt(name, true);
            }
        }

        private void BrowseForManifest()
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Manifest|*.yaml;*.yml",
                Title = "Choose a Manifest",
            };

            if (dialog.ShowDialog() == true)
            {
                // A manifest nobody published has no catalog entry; the popup shows its file name
                // and the footnote says what went wrong if the library cannot read it.
                describeFailed = null;
                description = null;
                RebuildChoices();
                ModelChoices.Insert(0, ModelChoice.Model(
                    dialog.FileName, Path.GetFileName(dialog.FileName)));
                chosen = ModelChoices[0];
                Raise("SelectedChoice");
                RefreshModelCard();
            }
            else
            {
                Raise("SelectedChoice");
            }
        }

        // -- the buttons ------------------------------------------------------------------------------

        private void OnGo()
        {
            if (Working)
            {
                Main.CancelFor(this);
            }
            else
            {
                Start();
            }
        }

        private void Start()
        {
            if (!CanGo)
            {
                return;
            }

            var job = Build();
            job.Model = ChosenModel == null ? null : ChosenModel.Name;
            job.ModelFullName = ChosenModel == null ? null : ChosenModel.FullName;
            job.Device = Device;
            job.Kind = Kind;
            Main.Enqueue(this, job);
        }

        private void OnSave()
        {
            var result = Selected;
            if (result == null || !File.Exists(result.Path))
            {
                return;
            }

            var extension = Category == Category.Image ? ".png" : ".wav";
            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                FileName = "waifu-" + DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture) + extension,
                Filter = Category == Category.Image ? "PNG|*.png" : "WAV|*.wav",
            };

            if (dialog.ShowDialog() == true)
            {
                File.Copy(result.Path, dialog.FileName, true);
            }
        }

        private void OnReuse()
        {
            if (Selected != null && Selected.Record.Made != null)
            {
                ReuseFrom(Selected.Record);
                GoChanged();
            }
        }

        private void OnDelete()
        {
            var result = Selected;
            if (result == null)
            {
                return;
            }

            var what = Category == Category.Image ? "picture" : "clip";
            var answer = MessageBox.Show(
                "Nothing else keeps a copy.",
                "Delete this " + what + "?",
                MessageBoxButton.OKCancel,
                MessageBoxImage.None);

            if (answer != MessageBoxResult.OK)
            {
                return;
            }

            Lib.Store.Delete(result.Record);
            Main.Results.Remove(result);
            Shown.Refresh();
            Selected = Shown.Cast<ResultViewModel>().FirstOrDefault();
        }
    }
}
