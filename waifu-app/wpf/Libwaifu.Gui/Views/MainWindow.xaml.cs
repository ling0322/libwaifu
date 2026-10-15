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
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;
using Libwaifu.Gui.ViewModels;

namespace Libwaifu.Gui.Views
{
    public partial class MainWindow : Window
    {
        private readonly MainViewModel model = new MainViewModel();
        private GridLength inspectorWas = new GridLength(300);
        private GridLength sidebarWas = new GridLength(300);

        public MainWindow()
        {
            InitializeComponent();
            DataContext = model;
            model.PropertyChanged += OnModelChanged;

            // Said once, in the one place that is always on screen: without the library behind it
            // this window goes through the motions and draws nothing.
            if (!Engine.Library.Available)
            {
                Title += " - stand-in engine";
            }

            if (App.AskedForDevice != null)
            {
                model.Device = App.AskedForDevice;
            }

            // `-m` chooses a model and turns to the page of the task it is for.
            if (App.AskedForModel != null)
            {
                var entry = Engine.Models.Find(App.AskedForModel);
                if (entry != null)
                {
                    var page = model.Pages.FirstOrDefault(task => task.Category == entry.Category);
                    if (page != null)
                    {
                        page.ChooseModel(entry.Name);
                        model.Page = page;
                    }
                }
            }

            Layout();
        }

        protected override void OnSourceInitialized(EventArgs e)
        {
            base.OnSourceInitialized(e);
            DarkenTheTitleBar();
        }

        /// <summary>
        /// Asked for a second time once the window is on screen. Asking before it is shown is
        /// accepted and does nothing: the bar is built light and stays that way until the
        /// attribute is set on a window that already has a frame.
        /// </summary>
        protected override void OnContentRendered(EventArgs e)
        {
            base.OnContentRendered(e);
            Dispatcher.BeginInvoke(
                System.Windows.Threading.DispatcherPriority.ApplicationIdle,
                (Action)DarkenTheTitleBar);
        }

        /// <summary>
        /// The title bar is drawn by Windows and would otherwise be the light one whatever the
        /// rest of the window does, so it is asked for the dark one by hand.
        /// </summary>
        private void DarkenTheTitleBar()
        {
            try
            {
                var handle = new WindowInteropHelper(this).Handle;
                var on = 1;
                if (DwmSetWindowAttribute(handle, UseImmersiveDarkMode, ref on, sizeof(int)) != 0)
                {
                    // Before Windows 10 2004 the attribute had a different number.
                    DwmSetWindowAttribute(handle, UseImmersiveDarkModeOld, ref on, sizeof(int));
                }
            }
            catch (Exception error)
            {
                Engine.Log.Write("warn", "window", error.Message);
            }
        }

        private void OnModelChanged(object sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == "InspectorVisible" || e.PropertyName == "SidebarVisible" ||
                e.PropertyName == "Page")
            {
                Layout();
            }
        }

        /// <summary>
        /// Which of the three columns are there. The Model Manager is one page across the middle
        /// and the right, so it puts the settings column away whether or not it was showing.
        /// </summary>
        private void Layout()
        {
            var inspector = model.InspectorVisible && !model.OnModelManager;
            if (!inspector && InspectorColumn.Width.Value > 0)
            {
                inspectorWas = InspectorColumn.Width;
            }

            InspectorColumn.MinWidth = inspector ? 300 : 0;
            InspectorColumn.Width = inspector ? inspectorWas : new GridLength(0);
            Inspector.Visibility = inspector ? Visibility.Visible : Visibility.Collapsed;
            InspectorLine.Visibility = Inspector.Visibility;
            InspectorSplitter.Visibility = Inspector.Visibility;

            if (!model.SidebarVisible && SidebarColumn.Width.Value > 0)
            {
                sidebarWas = SidebarColumn.Width;
            }

            SidebarColumn.MinWidth = model.SidebarVisible ? 200 : 0;
            SidebarColumn.Width = model.SidebarVisible ? sidebarWas : new GridLength(0);
            SidebarSplitter.Visibility = model.SidebarVisible ? Visibility.Visible : Visibility.Collapsed;
        }

        /// <summary>
        /// The shortcuts. They are handled here rather than as input bindings because every one
        /// of them acts on whichever task page is showing, which an input binding declared in
        /// XAML cannot reach.
        /// </summary>
        protected override void OnPreviewKeyDown(KeyEventArgs e)
        {
            base.OnPreviewKeyDown(e);

            var task = model.CurrentTask;
            var control = (Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control;
            var shift = (Keyboard.Modifiers & ModifierKeys.Shift) == ModifierKeys.Shift;

            if (control && shift && e.Key == Key.O)
            {
                model.ShowOutputFolder.Execute(null);
                e.Handled = true;
                return;
            }

            // The three pages the sidebar does not list are still reachable, which is how they
            // are worked on until the session offers them.
            if (control && shift && e.Key >= Key.D1 && e.Key <= Key.D4)
            {
                model.Page = model.Pages[e.Key - Key.D1];
                e.Handled = true;
                return;
            }

            if (task == null)
            {
                return;
            }

            if (control && (e.Key == Key.Enter || e.Key == Key.Return))
            {
                if (task.Generate.CanExecute(null))
                {
                    task.Generate.Execute(null);
                }

                e.Handled = true;
            }
            else if ((control && e.Key == Key.OemPeriod) || e.Key == Key.Escape)
            {
                if (task.Stop.CanExecute(null))
                {
                    task.Stop.Execute(null);
                    e.Handled = true;
                }
            }
            else if (control && e.Key == Key.S)
            {
                if (task.Save.CanExecute(null))
                {
                    task.Save.Execute(null);
                }

                e.Handled = true;
            }
        }

        private const int UseImmersiveDarkMode = 20;
        private const int UseImmersiveDarkModeOld = 19;

        [DllImport("dwmapi.dll")]
        private static extern int DwmSetWindowAttribute(
            IntPtr window, int attribute, ref int value, int size);
    }
}
