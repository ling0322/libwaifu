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

using System.Windows.Controls;
using Libwaifu.Gui.Controls;
using Libwaifu.Gui.ViewModels;

namespace Libwaifu.Gui.Views
{
    public partial class ImageInspector : UserControl
    {
        public ImageInspector()
        {
            InitializeComponent();
        }

        /// <summary>
        /// A picture arrived, either dropped on the box or chosen in the dialog it opens. Reading
        /// it is the page's business rather than the box's, which only says what was handed over.
        /// </summary>
        private void OnPictureChosen(object sender, DropBoxEventArgs e)
        {
            var page = DataContext as ImageTaskViewModel;
            if (page != null)
            {
                page.TakePicture(e.Path, e.Pasted);
            }
        }
    }
}
