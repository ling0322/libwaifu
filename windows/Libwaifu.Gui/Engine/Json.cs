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
using System.Globalization;
using System.Text;

namespace Libwaifu.Gui.Engine
{
    /// <summary>
    /// Reads the JSON the library answers with.
    /// </summary>
    /// <remarks>
    /// Small on purpose. The documents are the library's own and the shapes are in docs/ffi.md;
    /// what is needed of them is "give me this field if it is there", which is a dictionary and
    /// not a set of generated classes. .NET Framework's own reader would do most of it, but it
    /// turns a JSON object into an array of key-value pairs rather than a map, and `why_not` is
    /// a map whose keys are the model's to choose.
    /// </remarks>
    public static class Json
    {
        public static object Parse(string text)
        {
            var at = 0;
            var value = Read(text, ref at);
            Skip(text, ref at);
            return value;
        }

        /// <summary>A field of an object, or null where the object has nothing under that name.</summary>
        public static object Field(object node, string name)
        {
            var map = node as Dictionary<string, object>;
            object value;
            if (map != null && map.TryGetValue(name, out value))
            {
                return value;
            }

            return null;
        }

        public static string Text(object node, string name)
        {
            return Field(node, name) as string;
        }

        public static bool Flag(object node, string name, bool fallback = false)
        {
            var value = Field(node, name);
            return value is bool ? (bool)value : fallback;
        }

        public static double Number(object node, string name, double fallback = 0)
        {
            var value = Field(node, name);
            return value is double ? (double)value : fallback;
        }

        public static int Count(object node, string name, int fallback = 0)
        {
            var value = Field(node, name);
            return value is double ? (int)Math.Round((double)value) : fallback;
        }

        public static long Big(object node, string name, long fallback = 0)
        {
            var value = Field(node, name);
            return value is double ? (long)(double)value : fallback;
        }

        public static List<object> Items(object node, string name)
        {
            return Field(node, name) as List<object> ?? new List<object>();
        }

        public static Dictionary<string, object> Map(object node)
        {
            return node as Dictionary<string, object> ?? new Dictionary<string, object>();
        }

        // -- the reader --------------------------------------------------------------------

        private static object Read(string text, ref int at)
        {
            Skip(text, ref at);
            if (at >= text.Length)
            {
                return null;
            }

            switch (text[at])
            {
                case '{':
                    return ReadObject(text, ref at);
                case '[':
                    return ReadArray(text, ref at);
                case '"':
                    return ReadString(text, ref at);
                case 't':
                    at += 4;
                    return true;
                case 'f':
                    at += 5;
                    return false;
                case 'n':
                    at += 4;
                    return null;
                default:
                    return ReadNumber(text, ref at);
            }
        }

        private static Dictionary<string, object> ReadObject(string text, ref int at)
        {
            var map = new Dictionary<string, object>();
            at++;
            Skip(text, ref at);

            if (at < text.Length && text[at] == '}')
            {
                at++;
                return map;
            }

            while (at < text.Length)
            {
                Skip(text, ref at);
                var key = ReadString(text, ref at);
                Skip(text, ref at);
                at++;  // the colon
                map[key] = Read(text, ref at);
                Skip(text, ref at);

                if (at < text.Length && text[at] == ',')
                {
                    at++;
                    continue;
                }

                at++;  // the closing brace
                break;
            }

            return map;
        }

        private static List<object> ReadArray(string text, ref int at)
        {
            var items = new List<object>();
            at++;
            Skip(text, ref at);

            if (at < text.Length && text[at] == ']')
            {
                at++;
                return items;
            }

            while (at < text.Length)
            {
                items.Add(Read(text, ref at));
                Skip(text, ref at);

                if (at < text.Length && text[at] == ',')
                {
                    at++;
                    continue;
                }

                at++;  // the closing bracket
                break;
            }

            return items;
        }

        private static string ReadString(string text, ref int at)
        {
            var built = new StringBuilder();
            at++;

            while (at < text.Length && text[at] != '"')
            {
                if (text[at] != '\\')
                {
                    built.Append(text[at++]);
                    continue;
                }

                at++;
                var escape = text[at++];
                switch (escape)
                {
                    case 'n': built.Append('\n'); break;
                    case 't': built.Append('\t'); break;
                    case 'r': built.Append('\r'); break;
                    case 'b': built.Append('\b'); break;
                    case 'f': built.Append('\f'); break;
                    case 'u':
                        built.Append((char)int.Parse(
                            text.Substring(at, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                        at += 4;
                        break;
                    default:
                        built.Append(escape);
                        break;
                }
            }

            at++;
            return built.ToString();
        }

        private static object ReadNumber(string text, ref int at)
        {
            var from = at;
            while (at < text.Length && "+-.eE0123456789".IndexOf(text[at]) >= 0)
            {
                at++;
            }

            double value;
            if (double.TryParse(
                text.Substring(from, at - from),
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out value))
            {
                return value;
            }

            return null;
        }

        private static void Skip(string text, ref int at)
        {
            while (at < text.Length && char.IsWhiteSpace(text[at]))
            {
                at++;
            }
        }
    }
}
