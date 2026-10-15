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
    /// <summary>
    /// A titled card of rows with a footnote under it -- the shape every group of settings on
    /// this screen takes.
    /// </summary>
    /// <remarks>
    /// The rows are the items, which is what makes the hairline between them the container's
    /// business rather than each row's: the template draws a separator above every row but the
    /// first, which comes to the same thing as one below every row but the last and needs only
    /// the index to decide. That is why <see cref="ItemsControl.AlternationCount"/> is set high
    /// in the default style -- it makes AlternationIndex the real index rather than a cycle.
    ///
    /// A group with no rows draws no card at all, only its title and footnote, which is how the
    /// model card says "this model cannot start from a picture" with nothing to operate.
    /// </remarks>
    public class FormGroup : ItemsControl
    {
        public static readonly DependencyProperty TitleProperty = DependencyProperty.Register(
            "Title", typeof(string), typeof(FormGroup), new PropertyMetadata(null));

        public static readonly DependencyProperty FootnoteProperty = DependencyProperty.Register(
            "Footnote", typeof(string), typeof(FormGroup), new PropertyMetadata(null));

        /// <summary>Red rather than secondary, for a footnote that is a reason something failed.</summary>
        public static readonly DependencyProperty FootnoteIsErrorProperty = DependencyProperty.Register(
            "FootnoteIsError", typeof(bool), typeof(FormGroup), new PropertyMetadata(false));

        /// <summary>
        /// How much air a row has. The middle of the window gives rows 8 above and below, the
        /// settings column 5, which is what keeps a column of twelve settings on one screen.
        /// </summary>
        public static readonly DependencyProperty RowPaddingProperty = DependencyProperty.Register(
            "RowPadding", typeof(Thickness), typeof(FormGroup),
            new PropertyMetadata(new Thickness(12, 8, 12, 8)));

        public string Title
        {
            get { return (string)GetValue(TitleProperty); }
            set { SetValue(TitleProperty, value); }
        }

        public string Footnote
        {
            get { return (string)GetValue(FootnoteProperty); }
            set { SetValue(FootnoteProperty, value); }
        }

        public bool FootnoteIsError
        {
            get { return (bool)GetValue(FootnoteIsErrorProperty); }
            set { SetValue(FootnoteIsErrorProperty, value); }
        }

        public Thickness RowPadding
        {
            get { return (Thickness)GetValue(RowPaddingProperty); }
            set { SetValue(RowPaddingProperty, value); }
        }
        /// <summary>
        /// Rows are wrapped in a container of our own rather than in a bare
        /// <see cref="System.Windows.Controls.ContentPresenter"/>, because a presenter cannot be
        /// given a template and the separator above each row is drawn by the template.
        /// </summary>
        protected override DependencyObject GetContainerForItemOverride()
        {
            return new FormRowContainer();
        }

        protected override bool IsItemItsOwnContainerOverride(object item)
        {
            return item is FormRowContainer;
        }

        /// <summary>
        /// A row that is not shown takes its wrapper with it. Without this, a setting that
        /// belongs to another model -- the guidance slider on a model that takes no guidance --
        /// leaves an empty strip and a hairline behind in the card.
        /// </summary>
        protected override void PrepareContainerForItemOverride(DependencyObject element, object item)
        {
            base.PrepareContainerForItemOverride(element, item);

            var container = element as FormRowContainer;
            var row = item as UIElement;
            if (container != null && row != null)
            {
                container.SetBinding(
                    VisibilityProperty,
                    new System.Windows.Data.Binding("Visibility") { Source = row });
            }
        }
    }

    /// <summary>One row's wrapper: the hairline above it, and the padding its group asks for.</summary>
    public class FormRowContainer : ContentControl
    {
    }
}
