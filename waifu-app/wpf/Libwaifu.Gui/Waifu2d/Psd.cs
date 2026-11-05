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
//
// A Photoshop document's layers, read: as much of the format as a drawing in layers needs.
// A port of waifu-app/swiftui/Sources/Waifu/PSD.swift, which reads what
// waifu/src/see_through/psd.rs writes.
//
// Version 1 -- the .psd Photoshop saves; a .psb is version 2 -- eight bits a channel, RGB, or
// grayscale read as RGB. Each layer is a rectangle of pixels with a name, an opacity and whether
// it is shown; channel data raw or PackBits. Groups are read through: a layer in a hidden group
// is hidden, and the group's own records are not layers. Masks, effects and blend modes are left
// out. The layout is Adobe's "Photoshop File Formats Specification", big-endian throughout.

using System;
using System.Collections.Generic;
using System.Text;

namespace Libwaifu.Gui.Waifu2d
{
    /// One layer: where its top left corner is on the canvas, its pixels -- RGBA, not
    /// premultiplied, as large as its rectangle -- and how it is shown.
    sealed class PsdLayer
    {
        public string Name = "";
        public int Left;
        public int Top;
        public RgbaImage Image;
        /// 0 to 1.
        public double Opacity;
        /// Shown: neither it nor a group it is in is hidden.
        public bool Visible;
        public int Width => Image.Width;
        public int Height => Image.Height;
    }

    sealed class PsdDocument
    {
        public int Width;
        public int Height;
        /// Bottom first, as they are drawn.
        public List<PsdLayer> Layers = new List<PsdLayer>();

        public static PsdDocument Read(byte[] data) => new Reader(data).Document();

        /// Its layers, and apart from them the depth img2layers keeps in its hidden group: a
        /// hidden layer named `depth/<name>`, red its gray, 0 near to 255 far. Both in the
        /// document's order.
        public (List<PsdLayer> layers, List<PsdLayer> depths) SplitDepth()
        {
            var prefix = "depth/";
            bool IsDepth(PsdLayer l) => !l.Visible && l.Name.StartsWith(prefix);
            var layers = Layers.FindAll(l => !IsDepth(l));
            var depths = Layers.FindAll(IsDepth);
            return (layers, depths);
        }

        /// Reads one document front to back.
        sealed class Reader
        {
            readonly byte[] bytes;
            int at;
            /// Whether the document is grayscale rather than RGB.
            bool gray;

            public Reader(byte[] bytes) { this.bytes = bytes; }

            void Need(int count)
            {
                if (count < 0 || at + count > bytes.Length)
                    throw new Exception("the document ends where more of it was expected");
            }

            int U8() { Need(1); return bytes[at++]; }

            int U16() { Need(2); int v = bytes[at] << 8 | bytes[at + 1]; at += 2; return v; }

            int U32()
            {
                Need(4);
                int v = bytes[at] << 24 | bytes[at + 1] << 16 | bytes[at + 2] << 8 | bytes[at + 3];
                at += 4;
                return v;
            }

            int I32() => unchecked((int)(uint)U32());

            void Skip(int count) { Need(count); at += count; }

            string Ascii(int count)
            {
                Need(count);
                string s = Encoding.ASCII.GetString(bytes, at, count);
                at += count;
                return s;
            }

            public PsdDocument Document()
            {
                if (Ascii(4) != "8BPS") throw new Exception("this is not a Photoshop document");
                switch (U16())
                {
                    case 1: break;
                    case 2: throw new Exception("this is a large document (.psb): save it as a .psd");
                    case int v: throw new Exception($"this is a Photoshop document of version {v}, which is not read here");
                }
                Skip(6);
                U16(); // channels of the merged picture
                int height = U32();
                int width = U32();
                int depth = U16();
                int mode = U16();
                if (depth != 8) throw new Exception($"this document has {depth} bits a channel: save it with 8");
                if (mode != 3 && mode != 1) throw new Exception("this document is not RGB: convert it to RGB colour");
                gray = mode == 1;

                Skip(U32()); // colour mode data
                Skip(U32()); // image resources
                int section = U32();
                int end = at + section;
                var doc = new PsdDocument { Width = width, Height = height };
                if (section <= 0) return doc;
                doc.Layers = LayerInfo();
                at = end;
                return doc;
            }

            /// A layer's record: everything but its pixels, which come after every record.
            sealed class Record
            {
                public string Name;
                public int Top, Left, Bottom, Right;
                public List<(int id, int length)> Channels;
                public int Opacity;
                public bool Hidden;
                /// 1 or 2 opens a group -- going up the stack, its last record -- and 3 closes one.
                public int Divider;
            }

            List<PsdLayer> LayerInfo()
            {
                int length = U32();
                if (length <= 0) return new List<PsdLayer>();
                int count = Math.Abs((short)U16());
                var records = new List<Record>();
                for (int i = 0; i < count; i++) records.Add(ReadRecord());

                var pixels = new List<RgbaImage>();
                foreach (var record in records) pixels.Add(ChannelsOf(record));

                // Top first, so that a group's record comes before what is in it: a stack of
                // whether each open group is hidden.
                var layers = new List<PsdLayer>();
                var groups = new List<bool>();
                for (int i = records.Count - 1; i >= 0; i--)
                {
                    var record = records[i];
                    var image = pixels[i];
                    switch (record.Divider)
                    {
                        case 1:
                        case 2:
                            groups.Add(record.Hidden || (groups.Count > 0 && groups[groups.Count - 1]));
                            continue;
                        case 3:
                            if (groups.Count > 0) groups.RemoveAt(groups.Count - 1);
                            continue;
                    }
                    if (image == null) continue;
                    layers.Add(new PsdLayer
                    {
                        Name = record.Name,
                        Left = record.Left,
                        Top = record.Top,
                        Image = image,
                        Opacity = record.Opacity / 255.0,
                        Visible = !record.Hidden && !(groups.Count > 0 && groups[groups.Count - 1]),
                    });
                }
                layers.Reverse();
                return layers;
            }

            Record ReadRecord()
            {
                int top = I32(), left = I32(), bottom = I32(), right = I32();
                var channels = new List<(int id, int length)>();
                for (int i = 0, n = U16(); i < n; i++) channels.Add(((short)U16(), U32()));
                Skip(8); // "8BIM" and the blend mode
                int opacity = U8();
                U8(); // clipping
                int flags = U8();
                Skip(1);
                int extra = U32();
                int end = at + extra;
                Skip(U32()); // mask
                Skip(U32()); // blending ranges
                int @short = U8();
                string name = Ascii(@short);
                Skip((4 - (@short + 1) % 4) % 4);
                int divider = 0;
                // What else the layer says, by key: its name in full, and whether it opens or
                // closes a group.
                while (at + 12 <= end)
                {
                    string signature = Ascii(4);
                    if (signature != "8BIM" && signature != "8B64") break;
                    string key = Ascii(4);
                    int size = U32();
                    int start = at;
                    switch (key)
                    {
                        case "luni":
                            int characters = U32();
                            Need(characters * 2);
                            var units = new char[characters];
                            for (int i = 0; i < characters; i++)
                                units[i] = (char)(bytes[at + i * 2] << 8 | bytes[at + i * 2 + 1]);
                            name = new string(units);
                            break;
                        case "lsct":
                        case "lsdk":
                            divider = U32();
                            break;
                    }
                    at = start + size;
                }
                at = end;
                return new Record
                {
                    Name = name,
                    Top = top,
                    Left = left,
                    Bottom = bottom,
                    Right = right,
                    Channels = channels,
                    Opacity = opacity,
                    Hidden = (flags & 2) != 0,
                    Divider = divider,
                };
            }

            /// A layer's pixels, out of its channels: null for one with no area, a group's record
            /// among them.
            RgbaImage ChannelsOf(Record record)
            {
                int width = record.Right - record.Left;
                int height = record.Bottom - record.Top;
                int area = Math.Max(0, width) * Math.Max(0, height);
                var rgba = area > 0 ? new byte[area * 4] : null;
                if (rgba != null) for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255;
                foreach (var channel in record.Channels)
                {
                    int end = at + channel.length;
                    // Its pixels are its rectangle's: a mask's are not, and are not read.
                    if (area > 0 && channel.id >= -1 && channel.id <= (gray ? 0 : 2))
                    {
                        var plane = Plane(width, height, channel.length);
                        if (channel.id == -1)
                            for (int i = 0; i < area; i++) rgba[i * 4 + 3] = plane[i];
                        else if (gray)
                            for (int i = 0; i < area; i++)
                            {
                                rgba[i * 4] = plane[i];
                                rgba[i * 4 + 1] = plane[i];
                                rgba[i * 4 + 2] = plane[i];
                            }
                        else
                            for (int i = 0; i < area; i++) rgba[i * 4 + channel.id] = plane[i];
                    }
                    at = end;
                }
                return area > 0 ? new RgbaImage(width, height, rgba) : null;
            }

            /// One channel's `width` by `height` bytes: raw, or PackBits row by row.
            byte[] Plane(int width, int height, int length)
            {
                int compression = U16();
                switch (compression)
                {
                    case 0:
                        Need(width * height);
                        var raw = new byte[width * height];
                        Buffer.BlockCopy(bytes, at, raw, 0, raw.Length);
                        at += raw.Length;
                        return raw;
                    case 1:
                        var counts = new int[height];
                        for (int i = 0; i < height; i++) counts[i] = U16();
                        var plane = new byte[width * height];
                        for (int row = 0; row < height; row++)
                        {
                            int end = at + counts[row];
                            Need(counts[row]);
                            int x = 0;
                            while (at < end && x < width)
                            {
                                int n = (sbyte)bytes[at++];
                                if (n >= 0)
                                {
                                    int run = Math.Min(n + 1, Math.Min(width - x, end - at));
                                    Buffer.BlockCopy(bytes, at, plane, row * width + x, run);
                                    at += n + 1;
                                    x += run;
                                }
                                else if (n != -128)
                                {
                                    if (at >= end) break;
                                    int run = Math.Min(1 - n, width - x);
                                    byte v = bytes[at++];
                                    for (int i = 0; i < run; i++) plane[row * width + x + i] = v;
                                    x += run;
                                }
                            }
                            at = end;
                        }
                        return plane;
                    default:
                        throw new Exception("a layer is compressed with ZIP, which is not read here: save it without");
                }
            }
        }
    }
}
