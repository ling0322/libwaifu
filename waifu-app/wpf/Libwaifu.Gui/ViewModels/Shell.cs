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
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows.Threading;
using Libwaifu.Gui.Engine;

// `Engine` is both a namespace here and the name of the one engine the window holds, so the
// namespace is spelled out where a static of its own is wanted.
using Lib = Libwaifu.Gui.Engine;

namespace Libwaifu.Gui.ViewModels
{
    /// <summary>
    /// The window: which page is showing, what is queued, and the one engine everything shares.
    /// </summary>
    public class MainViewModel : Observable
    {
        private readonly DispatcherTimer machineTick;
        private object page;
        private bool inspectorVisible = true;
        private bool sidebarVisible = true;
        private TaskViewModel runningOn;
        private TaskViewModel queuedOn;
        private JobRequest queued;

        public MainViewModel()
        {
            Settings = Lib.Settings.Read();
            Settings.Device = Lib.Devices.Known(Settings.Device);

            // The library where it is beside the program, and the stand-in where it is not --
            // the same interface either way, so nothing above here knows which it got.
            Engine = Library.Available ? (IEngine)new NativeEngine() : new MockEngine();
            Engine.Progress += OnProgress;
            Engine.Finished += OnFinished;

            Results = new ObservableCollection<ResultViewModel>(
                Lib.Store.ReadAll().Select(record => new ResultViewModel(record)));

            Pages = new List<TaskViewModel>
            {
                new ImageTaskViewModel(this, TaskKind.Txt2Img),
                new ImageTaskViewModel(this, TaskKind.Img2Img),
                new SpeechTaskViewModel(this),
                new ConversionTaskViewModel(this),
            };

            // The sidebar lists the tasks the session offers, which today is one of the four.
            Tasks = new ObservableCollection<TaskViewModel>(
                Pages.Where(task => Lib.Tasks.Tabs.Contains(task.Kind)));

            ModelManager = new ModelManagerViewModel(this);
            ModelPages = new List<ModelManagerViewModel> { ModelManager };
            Machine = new MachineViewModel();

            ToggleInspector = new Command(delegate { InspectorVisible = !InspectorVisible; });
            ToggleSidebar = new Command(delegate { SidebarVisible = !SidebarVisible; });
            ShowLog = new Command(delegate
            {
                var window = new Views.LogWindow { Owner = System.Windows.Application.Current.MainWindow };
                window.ShowDialog();
            });
            ShowOutputFolder = new Command(delegate
            {
                Places.Ensure(Places.Output);
                Process.Start("explorer.exe", "\"" + Places.Output + "\"");
            });

            page = Tasks.FirstOrDefault();

            machineTick = new DispatcherTimer(DispatcherPriority.Background)
            {
                Interval = TimeSpan.FromSeconds(2),
            };
            // How big the card is decides whether the Model card says a model may not fit on it,
            // and that answer only arrives once the machine has been read.
            Machine.PropertyChanged += delegate(object sender, System.ComponentModel.PropertyChangedEventArgs e)
            {
                if (e.PropertyName == "VramTotal")
                {
                    foreach (var task in Pages)
                    {
                        task.RefreshModelCard();
                    }
                }
            };

            machineTick.Tick += delegate { Machine.Refresh(); };
            machineTick.Start();
            Machine.Refresh();
        }

        public Settings Settings { get; private set; }

        public IEngine Engine { get; private set; }

        /// <summary>Every result ever, newest first, whichever task made it.</summary>
        public ObservableCollection<ResultViewModel> Results { get; private set; }

        public List<TaskViewModel> Pages { get; private set; }

        public ObservableCollection<TaskViewModel> Tasks { get; private set; }

        public ModelManagerViewModel ModelManager { get; private set; }

        /// <summary>
        /// The Models section of the sidebar, which holds the one row. It is a list rather than a
        /// row so that it has a selection of its own, and so that choosing in it clears the task
        /// list above.
        /// </summary>
        public List<ModelManagerViewModel> ModelPages { get; private set; }

        public MachineViewModel Machine { get; private set; }

        public Command ToggleInspector { get; private set; }

        public Command ToggleSidebar { get; private set; }

        public Command ShowLog { get; private set; }

        public Command ShowOutputFolder { get; private set; }

        /// <summary>The task page or the Model Manager, whichever the sidebar last chose.</summary>
        public object Page
        {
            get { return page; }
            set
            {
                if (value == null || !Set(ref page, value))
                {
                    return;
                }

                Raise("CurrentTask");
                Raise("OnModelManager");
                Raise("Title");
                Raise("Subtitle");
                Raise("InspectorButtonVisible");
                Raise("SelectedTask");
                Raise("SelectedModels");
            }
        }

        /// <summary>
        /// What the sidebar's task list is bound to. Choosing the Model Manager leaves it with
        /// nothing selected, which is how the two lists behave as one.
        /// </summary>
        public TaskViewModel SelectedTask
        {
            get { return page as TaskViewModel; }
            set
            {
                if (value != null)
                {
                    Page = value;
                }
            }
        }

        /// <summary>
        /// The Models list below the tasks, which holds one row. It is a list of its own so that
        /// choosing in it clears the task list's selection, the way one sidebar behaves.
        /// </summary>
        public ModelManagerViewModel SelectedModels
        {
            get { return page as ModelManagerViewModel; }
            set
            {
                if (value != null)
                {
                    Page = value;
                }
            }
        }

        public TaskViewModel CurrentTask
        {
            get { return page as TaskViewModel; }
        }

        public bool OnModelManager
        {
            get { return page is ModelManagerViewModel; }
        }

        public string Title
        {
            get { return OnModelManager ? "Model Manager" : CurrentTask.Name; }
        }

        /// <summary>"&lt;model&gt; on &lt;device&gt;", under the task's name in the toolbar.</summary>
        public string Subtitle
        {
            get
            {
                if (OnModelManager || CurrentTask == null)
                {
                    return null;
                }

                var chosen = CurrentTask.ChosenModel;
                if (chosen == null)
                {
                    return "no model chosen";
                }

                return chosen.FullName + " on " + Settings.Device;
            }
        }

        public bool InspectorVisible
        {
            get { return inspectorVisible; }
            set { Set(ref inspectorVisible, value); }
        }

        public bool SidebarVisible
        {
            get { return sidebarVisible; }
            set { Set(ref sidebarVisible, value); }
        }

        public bool InspectorButtonVisible
        {
            get { return !OnModelManager; }
        }

        public string Device
        {
            get { return Lib.Devices.Known(Settings.Device); }
            set
            {
                if (Settings.Device == value)
                {
                    return;
                }

                Settings.Device = value;
                Settings.Save();
                Raise();
                Raise("Subtitle");
                foreach (var task in Pages)
                {
                    task.RefreshModelCard();
                }
            }
        }

        // -- one job at a time, and one waiting ----------------------------------------------

        public bool Busy
        {
            get { return runningOn != null || queuedOn != null; }
        }

        /// <summary>
        /// Queue a job. There is room for one running and one waiting; a third press of the
        /// button does nothing, which is why the button is a Cancel by then.
        /// </summary>
        public void Enqueue(TaskViewModel task, JobRequest job)
        {
            if (runningOn == null)
            {
                runningOn = task;
                task.JobStarted();
                Engine.Begin(job);
            }
            else if (queuedOn == null)
            {
                queuedOn = task;
                queued = job;
                task.JobQueued();
            }

            Changed();
        }

        public void CancelFor(TaskViewModel task)
        {
            if (queuedOn == task)
            {
                // The waiting one is taken out of line at once; nothing has happened to it yet.
                queuedOn = null;
                queued = null;
                task.JobDequeued();
                Changed();
                return;
            }

            if (runningOn == task)
            {
                Engine.Cancel();
                task.Stopping();
            }
        }

        public bool IsRunning(TaskViewModel task)
        {
            return runningOn == task;
        }

        public bool IsQueued(TaskViewModel task)
        {
            return queuedOn == task;
        }

        private void OnProgress(EngineEvent e)
        {
            if (runningOn != null)
            {
                runningOn.Advanced(e);
            }
        }

        private void OnFinished(JobOutcome outcome)
        {
            var task = runningOn;
            runningOn = null;

            if (task != null)
            {
                task.JobEnded(outcome);
            }

            if (outcome.Ending == Ending.Done && outcome.Result != null)
            {
                var result = new ResultViewModel(outcome.Result);
                Results.Insert(0, result);
                if (task != null)
                {
                    task.Select(result);
                }
            }

            if (queuedOn != null)
            {
                var next = queuedOn;
                var job = queued;
                queuedOn = null;
                queued = null;
                runningOn = next;
                next.JobStarted();
                Engine.Begin(job);
            }

            Changed();
            Reread();
        }

        /// <summary>
        /// Something about the models changed -- one was chosen, fetched or deleted -- so the
        /// toolbar's subtitle and the Model Manager both have to be asked again.
        /// </summary>
        public void Reread()
        {
            Raise("Title");
            Raise("Subtitle");
            if (ModelManager != null)
            {
                ModelManager.Reread();
            }
        }

        private void Changed()
        {
            Raise("Busy");
            Raise("Subtitle");
            foreach (var task in Pages)
            {
                task.BusyChanged();
            }
        }
    }

    /// <summary>The lines under "This PC", read again every two seconds on a thread of their own.</summary>
    public class MachineViewModel : Observable
    {
        private MachineSnapshot now = new MachineSnapshot();
        private bool reading;

        public string CpuName
        {
            get { return now.CpuName; }
        }

        public string Threads
        {
            get
            {
                return now.Cores + (now.Cores == 1 ? " core" : " cores") + " · " +
                       now.Threads + (now.Threads == 1 ? " thread" : " threads");
            }
        }

        public string Memory
        {
            get
            {
                if (now.MemoryTotal == 0)
                {
                    return null;
                }

                return Converters.Bytes.Format(now.MemoryUsed) + " of " +
                       Converters.Bytes.Format(now.MemoryTotal);
            }
        }

        public double MemoryFraction
        {
            get { return now.MemoryTotal == 0 ? 0 : 100.0 * now.MemoryUsed / now.MemoryTotal; }
        }

        /// <summary>Past ninety percent the bar is the one thing on this screen that is red.</summary>
        public bool MemoryTight
        {
            get { return MemoryFraction > 90; }
        }

        public string GpuName
        {
            get { return now.GpuName ?? "no display adapter reported"; }
        }

        /// <summary>
        /// The line under the card's name: how much of its memory is in use where the driver
        /// says, that the two share one pool where they do, and otherwise the reason there is no
        /// bar to draw.
        /// </summary>
        public string Vram
        {
            get
            {
                if (now.Unified)
                {
                    return "memory is shared with the CPU";
                }

                if (now.VramKnown && now.Vram > 0)
                {
                    return Converters.Bytes.Format(now.VramUsed) + " of " +
                           Converters.Bytes.Format(now.Vram);
                }

                if (now.Vram > 0)
                {
                    return Converters.Bytes.Format(now.Vram) + " of video memory · " + now.WhyNoVram;
                }

                return now.WhyNoVram;
            }
        }

        /// <summary>How much the card has in all, for working out what will not fit on it.</summary>
        public ulong VramTotal
        {
            get { return now.Unified ? 0 : now.Vram; }
        }

        public double VramFraction
        {
            get { return now.Vram == 0 ? 0 : 100.0 * now.VramUsed / now.Vram; }
        }

        /// <summary>Whether there is a bar to draw at all, which is whether anything said.</summary>
        public bool VramMeasured
        {
            get { return !now.Unified && now.VramKnown && now.Vram > 0; }
        }

        public bool VramTight
        {
            get { return VramFraction > 90; }
        }

        public string Accelerators
        {
            get
            {
                if (now.Accelerators == null || now.Accelerators.Count == 0)
                {
                    return "none -- runs go to the processor";
                }

                return string.Join(" · ", now.Accelerators.ToArray());
            }
        }

        /// <summary>
        /// Reading the machine touches the registry and the kernel, so it happens off the UI
        /// thread; what comes back is handed to the dispatcher in one piece.
        /// </summary>
        public void Refresh()
        {
            if (reading)
            {
                return;
            }

            reading = true;
            var dispatcher = Dispatcher.CurrentDispatcher;
            ThreadPool.QueueUserWorkItem(delegate
            {
                MachineSnapshot read;
                try
                {
                    read = Machine.Read();
                }
                catch (Exception error)
                {
                    Log.Write("warn", "machine", error.Message);
                    read = new MachineSnapshot();
                }

                dispatcher.BeginInvoke((Action)delegate
                {
                    now = read;
                    reading = false;
                    foreach (var name in new[]
                    {
                        "CpuName", "Threads", "Memory", "MemoryFraction", "MemoryTight",
                        "GpuName", "Vram", "VramFraction", "VramMeasured", "VramTight",
                        "VramTotal", "Accelerators",
                    })
                    {
                        Raise(name);
                    }
                });
            });
        }
    }

    /// <summary>One finished result, and the picture or recording it stands for.</summary>
    public class ResultViewModel : Observable
    {
        private System.Windows.Media.Imaging.BitmapImage picture;

        public ResultViewModel(ResultRecord record)
        {
            Record = record;
        }

        public ResultRecord Record { get; private set; }

        public Category Category
        {
            get { return Record.Category; }
        }

        public string Path
        {
            get { return Store.PathOf(Record); }
        }

        /// <summary>
        /// Read when it is first shown rather than when the folder is scanned: a hundred results
        /// is a hundred files, and a window that reads them all to open is a window that does not
        /// open.
        /// </summary>
        public System.Windows.Media.Imaging.BitmapImage Picture
        {
            get
            {
                if (picture == null && Category == Category.Image && File.Exists(Path))
                {
                    try
                    {
                        picture = new System.Windows.Media.Imaging.BitmapImage();
                        picture.BeginInit();
                        picture.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
                        picture.UriSource = new Uri(Path);
                        picture.EndInit();
                        picture.Freeze();
                    }
                    catch (Exception error)
                    {
                        Log.Write("warn", "store", error.Message);
                        picture = null;
                    }
                }

                return picture;
            }
        }

        public string Parameters
        {
            get { return Record.Made == null ? null : Record.Made.Parameters; }
        }

        /// <summary>What the strip says when the pointer rests on one of them.</summary>
        public string Hover
        {
            get
            {
                var seed = Record.Made == null
                    ? 0
                    : Record.Made.Seed;
                return Record.FileName + " -- seed " + seed.ToString(CultureInfo.InvariantCulture);
            }
        }

        /// <summary>The first line of what was said, which is what names a clip in the strip.</summary>
        public string FirstLine
        {
            get
            {
                var text = Record.Made == null ? null : Record.Made.Text;
                if (string.IsNullOrEmpty(text))
                {
                    return Record.FileName;
                }

                var stop = text.IndexOfAny(new[] { '.', '!', '?', '\n' });
                return stop > 0 ? text.Substring(0, stop + 1) : text;
            }
        }

        public string Length
        {
            get
            {
                var seconds = Record.Made == null || !Record.Made.Length.HasValue
                    ? 0
                    : Record.Made.Length.Value;
                return Converters.Clock.Format(seconds);
            }
        }
    }
}
