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

namespace Libwaifu.Gui.Controls
{
    /// <summary>One line of a <see cref="FormGroup"/>: a label on the left, a control on the right.</summary>
    /// <remarks>
    /// In the settings column the label column is a fixed 86 wide, so that every control in the
    /// column starts and ends on the same two lines however short its label is; a label longer
    /// than that pushes its control right rather than being cut. In the middle of the window and
    /// in the Model Manager the label takes what it needs and the control is pushed to the right
    /// edge, which is what <c>Auto</c> here means.
    /// </remarks>
    public class SettingRow : ContentControl
    {
        public static readonly DependencyProperty LabelProperty = DependencyProperty.Register(
            "Label", typeof(string), typeof(SettingRow), new PropertyMetadata(null));

        public static readonly DependencyProperty LabelWidthProperty = DependencyProperty.Register(
            "LabelWidth", typeof(GridLength), typeof(SettingRow),
            new PropertyMetadata(new GridLength(86)));

        public string Label
        {
            get { return (string)GetValue(LabelProperty); }
            set { SetValue(LabelProperty, value); }
        }

        public GridLength LabelWidth
        {
            get { return (GridLength)GetValue(LabelWidthProperty); }
            set { SetValue(LabelWidthProperty, value); }
        }
    }
}
