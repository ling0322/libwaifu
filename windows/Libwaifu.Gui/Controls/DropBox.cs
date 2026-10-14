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
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace Libwaifu.Gui.Controls
{
    /// <summary>What was let go over a <see cref="DropBox"/>, or what a file dialog returned.</summary>
    public class DropBoxEventArgs : EventArgs
    {
        public DropBoxEventArgs(string path, BitmapSource pasted)
        {
            Path = path;
            Pasted = pasted;
        }

        /// <summary>The file, or null when a picture was dragged in as pixels rather than as a file.</summary>
        public string Path { get; private set; }

        /// <summary>The pixels, when that is what arrived.</summary>
        public BitmapSource Pasted { get; private set; }
    }

    /// <summary>
    /// The dashed box a picture or a recording is dropped on, which is also a button that opens a
    /// file dialog -- both ways of answering the same question, so both land in one event.
    /// </summary>
    /// <remarks>
    /// A bitmap dragged out of a browser arrives as pixels and not as a file, and refusing that
    /// would be refusing the most ordinary way of getting a picture onto this screen, so the
    /// clipboard's bitmap format is accepted alongside the file list.
    /// </remarks>
    public class DropBox : Button
    {
        public static readonly DependencyProperty CaptionProperty = DependencyProperty.Register(
            "Caption", typeof(string), typeof(DropBox), new PropertyMetadata(null));

        private static readonly DependencyPropertyKey IsDragOverKey = DependencyProperty.RegisterReadOnly(
            "IsDragOver", typeof(bool), typeof(DropBox), new PropertyMetadata(false));

        public static readonly DependencyProperty IsDragOverProperty = IsDragOverKey.DependencyProperty;

        /// <summary>The geometry drawn beside the caption: a picture for one, a waveform for the other.</summary>
        public static readonly DependencyProperty IconProperty = DependencyProperty.Register(
            "Icon", typeof(System.Windows.Media.Geometry), typeof(DropBox), new PropertyMetadata(null));

        /// <summary>Which file types the dialog offers, and so also what a drop is allowed to be.</summary>
        public static readonly DependencyProperty FilterProperty = DependencyProperty.Register(
            "Filter", typeof(string), typeof(DropBox), new PropertyMetadata("All files|*.*"));

        public event EventHandler<DropBoxEventArgs> Chosen;

        public DropBox()
        {
            AllowDrop = true;
            Click += delegate { Browse(); };
        }

        public string Caption
        {
            get { return (string)GetValue(CaptionProperty); }
            set { SetValue(CaptionProperty, value); }
        }

        public bool IsDragOver
        {
            get { return (bool)GetValue(IsDragOverProperty); }
        }

        public System.Windows.Media.Geometry Icon
        {
            get { return (System.Windows.Media.Geometry)GetValue(IconProperty); }
            set { SetValue(IconProperty, value); }
        }

        public string Filter
        {
            get { return (string)GetValue(FilterProperty); }
            set { SetValue(FilterProperty, value); }
        }

        private void Browse()
        {
            var dialog = new Microsoft.Win32.OpenFileDialog { Filter = Filter };
            if (dialog.ShowDialog() == true)
            {
                Raise(new DropBoxEventArgs(dialog.FileName, null));
            }
        }

        protected override void OnDragEnter(DragEventArgs e)
        {
            base.OnDragEnter(e);
            Look(e);
        }

        protected override void OnDragOver(DragEventArgs e)
        {
            base.OnDragOver(e);
            Look(e);
        }

        protected override void OnDragLeave(DragEventArgs e)
        {
            base.OnDragLeave(e);
            SetValue(IsDragOverKey, false);
        }

        protected override void OnDrop(DragEventArgs e)
        {
            base.OnDrop(e);
            SetValue(IsDragOverKey, false);
            e.Handled = true;

            var path = FirstFile(e.Data);
            if (path != null)
            {
                Raise(new DropBoxEventArgs(path, null));
                return;
            }

            if (e.Data.GetDataPresent(DataFormats.Bitmap))
            {
                var pixels = e.Data.GetData(DataFormats.Bitmap) as BitmapSource;
                if (pixels != null)
                {
                    Raise(new DropBoxEventArgs(null, pixels));
                }
            }
        }

        private void Look(DragEventArgs e)
        {
            var welcome = FirstFile(e.Data) != null || e.Data.GetDataPresent(DataFormats.Bitmap);
            SetValue(IsDragOverKey, welcome);
            e.Effects = welcome ? DragDropEffects.Copy : DragDropEffects.None;
            e.Handled = true;
        }

        /// <summary>The first dragged file whose extension the filter names, or null.</summary>
        private string FirstFile(IDataObject data)
        {
            if (!data.GetDataPresent(DataFormats.FileDrop))
            {
                return null;
            }

            var files = data.GetData(DataFormats.FileDrop) as string[];
            if (files == null)
            {
                return null;
            }

            return files.FirstOrDefault(Wanted);
        }

        private bool Wanted(string path)
        {
            var extension = System.IO.Path.GetExtension(path);
            if (string.IsNullOrEmpty(extension))
            {
                return false;
            }

            // The filter is the dialog's own format -- pairs of "what it is|*.a;*.b" -- and the
            // patterns are every other field of it.
            var fields = Filter.Split('|');
            for (var i = 1; i < fields.Length; i += 2)
            {
                foreach (var pattern in fields[i].Split(';'))
                {
                    var wanted = pattern.Trim();
                    if (wanted == "*.*" ||
                        string.Equals(wanted, "*" + extension, StringComparison.OrdinalIgnoreCase))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        private void Raise(DropBoxEventArgs e)
        {
            var handler = Chosen;
            if (handler != null)
            {
                handler(this, e);
            }
        }
    }
}
