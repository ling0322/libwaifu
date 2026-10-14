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

using System.Windows;
using System.Windows.Controls;
using Libwaifu.Gui.ViewModels;

namespace Libwaifu.Gui.Views
{
    /// <summary>
    /// Which settings the right-hand column shows. The middle of the window picks its view by
    /// the page's type, which WPF does on its own; the column needs a second view of the same
    /// page, so it asks for one by name.
    /// </summary>
    public class InspectorSelector : DataTemplateSelector
    {
        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            string key = null;

            if (item is ImageTaskViewModel)
            {
                key = "Inspector.Image";
            }
            else if (item is SpeechTaskViewModel)
            {
                key = "Inspector.Speech";
            }
            else if (item is ConversionTaskViewModel)
            {
                key = "Inspector.Conversion";
            }

            return key == null ? null : (DataTemplate)Application.Current.Resources[key];
        }
    }
}
