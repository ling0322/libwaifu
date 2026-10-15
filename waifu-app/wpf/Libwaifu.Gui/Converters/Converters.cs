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
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace Libwaifu.Gui.Converters
{
    /// <summary>Shows a thing only when there is something to say.</summary>
    public class EmptyToCollapsed : IValueConverter
    {
        public object Convert(object value, Type target, object parameter, CultureInfo culture)
        {
            var text = value as string;
            var there = value is string ? !string.IsNullOrEmpty(text) : value != null;
            return there ? Visibility.Visible : Visibility.Collapsed;
        }

        public object ConvertBack(object value, Type target, object parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }

    /// <summary>The other way round, for the placeholder that shows when a box is empty.</summary>
    public class EmptyToVisible : IValueConverter
    {
        public object Convert(object value, Type target, object parameter, CultureInfo culture)
        {
            var text = value as string;
            var there = value is string ? !string.IsNullOrEmpty(text) : value != null;
            return there ? Visibility.Collapsed : Visibility.Visible;
        }

        public object ConvertBack(object value, Type target, object parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }

    public class BoolToCollapsed : IValueConverter
    {
        public object Convert(object value, Type target, object parameter, CultureInfo culture)
        {
            return Equals(value, true) ? Visibility.Collapsed : Visibility.Visible;
        }

        public object ConvertBack(object value, Type target, object parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }

    public class Not : IValueConverter
    {
        public object Convert(object value, Type target, object parameter, CultureInfo culture)
        {
            return !Equals(value, true);
        }

        public object ConvertBack(object value, Type target, object parameter, CultureInfo culture)
        {
            return !Equals(value, true);
        }
    }

    /// <summary>
    /// Shown only where every one of several things is true: a setting that belongs to one task
    /// and to one kind of model at once has two conditions and not one.
    /// </summary>
    public class AllTrue : IMultiValueConverter
    {
        public object Convert(object[] values, Type target, object parameter, CultureInfo culture)
        {
            foreach (var value in values)
            {
                if (!Equals(value, true))
                {
                    return Visibility.Collapsed;
                }
            }

            return Visibility.Visible;
        }

        public object[] ConvertBack(object value, Type[] targets, object parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }

    /// <summary>
    /// How much room something takes, written the way a disk is sold rather than the way it is
    /// addressed: decimal, so 5.48 GB here is 5.48 GB on the model's page.
    /// </summary>
    public class Bytes : IValueConverter
    {
        public static string Format(double bytes)
        {
            if (bytes >= 1e9)
            {
                return (bytes / 1e9).ToString("0.0", CultureInfo.InvariantCulture) + " GB";
            }

            if (bytes >= 1e6)
            {
                return (bytes / 1e6).ToString("0", CultureInfo.InvariantCulture) + " MB";
            }

            return (bytes / 1e3).ToString("0", CultureInfo.InvariantCulture) + " kB";
        }

        public object Convert(object value, Type target, object parameter, CultureInfo culture)
        {
            return Format(System.Convert.ToDouble(value, CultureInfo.InvariantCulture));
        }

        public object ConvertBack(object value, Type target, object parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }

    /// <summary>Seconds as m:ss, for the player's two ends.</summary>
    public class Clock : IValueConverter
    {
        public static string Format(double seconds)
        {
            if (double.IsNaN(seconds) || double.IsInfinity(seconds) || seconds < 0)
            {
                seconds = 0;
            }

            var whole = (int)Math.Floor(seconds);
            return (whole / 60) + ":" + (whole % 60).ToString("00", CultureInfo.InvariantCulture);
        }

        public object Convert(object value, Type target, object parameter, CultureInfo culture)
        {
            return Format(System.Convert.ToDouble(value, CultureInfo.InvariantCulture));
        }

        public object ConvertBack(object value, Type target, object parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }
}
