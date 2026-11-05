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
// A drawing in layers rigged to move, the way Anime2.5DRig (https://github.com/852wa/Anime2.5DRig,
// MIT, Copyright (c) 2026 hakoniwa) rigs one: the names its layers are known by, where each sits
// in depth and whether it is the head's or the body's, the anchors it moves about, the strands its
// hair swings in, and the closed eyes it is given where it has none. Its numbers are
// Anime2.5DRig's (lib/rigger.js); how it moves is RigMotion's.
//
// A port of waifu-app/swiftui/Sources/Waifu/Rig.swift.
//
// A blink is not a lid. The open eye -- its white, iris and lashes -- is squashed toward a
// line across it, and when it is a quarter open it fades into a closed eye drawn apart: the
// document's own, or a stock one (ClosedParts) fitted to the eye and tinted to its lashes. The
// iris stays inside the white as the white is squashed. Which eye is which is the picture's: an
// eye layer drawn as one is cut in two at the middle of the face.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;

namespace Libwaifu.Gui.Waifu2d
{
    /// Which eye, as the picture has them: `Left` is the character's right.
    enum Side { Left, Right }

    /// What a layer is, for the parts of the rig that treat it as its own: an eye's, a brow, the
    /// face, and everything else.
    enum Role { EyeWhite, Irides, EyeLash, EyeClose, Eyebrow, Face, Other }

    /// Whether a layer moves with the head or with the body.
    enum RigGroup { Head, Body }

    /// How a layer swings: hair, or a thing hung from something -- an earring, a tail.
    enum Phys { Hair, Sway }

    /// What a kind of layer is to the rig: Anime2.5DRig's `SLOTS`. Depth is 1 at the face and more
    /// in front of it: how far a layer moves as the head turns.
    sealed class Slot
    {
        public float Depth;
        /// Null for one that is the head's or the body's by where it is.
        public RigGroup? Group;
        public Phys? Physics;
        public Slot(float depth, RigGroup? group, Phys? phys = null)
        {
            Depth = depth;
            Group = group;
            Physics = phys;
        }
    }

    /// One strand a layer of hair swings in: where it hangs, its root and its tip, on the canvas.
    struct Strand
    {
        public float X, RootY, TipY;
    }

    /// How a layer of hair or a hanging thing swings: its strands, each vertex's weight towards
    /// each one, and how far down it each vertex is, 0 at the root to 1 at the tip.
    sealed class Swing
    {
        public Phys Kind;
        /// Front hair, side hair and ahoge swing as front hair does.
        public bool Front;
        public Strand[] Strands;
        /// `Strands.Length` a vertex, vertex by vertex.
        public float[] Weights;
        public float[] Along;
    }

    sealed class RigPart
    {
        public string Name;
        /// Its slot's name -- "front hair", "topwear" -- or the name as it is folded where it has
        /// none.
        public string Slot;
        public Role Role;
        public Side? Side;
        /// Where it is on the canvas.
        public Rect Rect;
        /// RGBA, not premultiplied, the size of `Rect`.
        public RgbaImage Pixels;
        public double Opacity;
        public float Depth;
        public RigGroup Group;
        /// The grid it is drawn on, moved vertex by vertex as it moves.
        public RigMesh Mesh;
        public Swing Swing;
    }

    /// A layer's grid: its vertices at rest, on the canvas, where each one reads the layer's
    /// picture, and the triangles between them. Anime2.5DRig's: cells of 42 pixels at a 768
    /// canvas -- 30 for what swings -- as wide as the canvas is over 768 but never under 0.6 of
    /// that, at least two a side.
    sealed class RigMesh
    {
        public Vector2[] Rest;
        public Vector2[] Uv;
        public uint[] Indices;

        public static RigMesh Grid(Rect rect, int canvas, bool swings = false)
        {
            double cell = (swings ? 30 : 42) * Math.Max(0.6, canvas / 768.0);
            int nx = Math.Max(2, (int)Math.Round(rect.Width / cell));
            int ny = Math.Max(2, (int)Math.Round(rect.Height / cell));
            var rest = new List<Vector2>();
            var uv = new List<Vector2>();
            for (int j = 0; j <= ny; j++)
            for (int i = 0; i <= nx; i++)
            {
                float u = (float)i / nx, v = (float)j / ny;
                uv.Add(new Vector2(u, v));
                rest.Add(new Vector2(
                    (float)(rect.X + u * rect.Width),
                    (float)(rect.Y + v * rect.Height)));
            }
            var indices = new List<uint>();
            for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
            {
                uint a = (uint)(j * (nx + 1) + i);
                uint b = a + 1;
                uint c = a + (uint)(nx + 1);
                uint d = c + 1;
                indices.Add(a); indices.Add(b); indices.Add(c);
                indices.Add(b); indices.Add(d); indices.Add(c);
            }
            return new RigMesh { Rest = rest.ToArray(), Uv = uv.ToArray(), Indices = indices.ToArray() };
        }
    }

    /// An eye at rest: the box of its white, the middle of its iris, and the line it closes onto.
    sealed class EyeAnchor
    {
        public Rect White;
        public Point Iris;
        public double CloseY;
    }

    /// Where the rig moves about: the face, the neck it turns on, the body's foot and the chest.
    sealed class Anchors
    {
        public Rect Face;
        public Point FaceCenter;
        /// What the head turns about: 85 % of the way down the neck.
        public Point NeckPivot;
        public double NeckTop;
        public double NeckBottom;
        /// What the body leans about: under the neck, at the canvas's foot.
        public Point BodyPivot;
        /// The middle of the chest and how far it reaches across and down.
        public Point Chest;
        public Size ChestReach;
    }

    struct Point
    {
        public double X, Y;
        public Point(double x, double y) { X = x; Y = y; }
    }

    struct Size
    {
        public double Width, Height;
        public Size(double w, double h) { Width = w; Height = h; }
    }

    struct Rect
    {
        public double X, Y, Width, Height;
        public Rect(double x, double y, double w, double h) { X = x; Y = y; Width = w; Height = h; }
        public double MinX => X;
        public double MinY => Y;
        public double MaxX => X + Width;
        public double MaxY => Y + Height;
        public double MidX => X + Width / 2;
        public double MidY => Y + Height / 2;
        public Rect Offset(double dx, double dy) => new Rect(X + dx, Y + dy, Width, Height);
    }

    sealed class Rig
    {
        public int Width;
        public int Height;
        /// Bottom first, the hidden ones left out.
        public List<RigPart> Parts = new List<RigPart>();
        public Dictionary<Side, EyeAnchor> Eyes = new Dictionary<Side, EyeAnchor>();
        public Anchors Anchors;
        /// How wide the face is over 333: what every distance the rig moves by is scaled by.
        public double FaceScale;
        /// What it could not do, to say under the picture.
        public List<string> Notes = new List<string>();

        /// The white of `side`'s eye, which its iris is kept inside.
        public RigPart White(Side side) =>
            Parts.LastOrDefault(p => p.Role == Role.EyeWhite && p.Side == side);

        // MARK: - names

        /// Anime2.5DRig's slots, by name.
        static readonly Dictionary<string, Slot> slots = new Dictionary<string, Slot>
        {
            ["wings"] = new Slot(0.48f, RigGroup.Body),
            ["tail"] = new Slot(0.50f, RigGroup.Body, Phys.Sway),
            ["back hair"] = new Slot(0.55f, RigGroup.Head, Phys.Hair),
            ["footwear"] = new Slot(0.83f, RigGroup.Body),
            ["legwear"] = new Slot(0.84f, RigGroup.Body),
            ["bottomwear"] = new Slot(0.88f, RigGroup.Body),
            ["neck"] = new Slot(0.95f, RigGroup.Body),
            ["topwear"] = new Slot(0.90f, RigGroup.Body),
            ["neckwear"] = new Slot(0.98f, RigGroup.Body),
            ["handwear"] = new Slot(0.86f, RigGroup.Body),
            ["objects"] = new Slot(1.00f, null),
            ["earwear"] = new Slot(0.97f, RigGroup.Head, Phys.Sway),
            ["ears"] = new Slot(0.96f, RigGroup.Head),
            ["face"] = new Slot(1.00f, RigGroup.Head),
            ["facedetail"] = new Slot(1.02f, RigGroup.Head),
            ["headwear"] = new Slot(1.20f, RigGroup.Head),
            ["mouth_close"] = new Slot(1.08f, RigGroup.Head),
            ["mouth_open"] = new Slot(1.08f, RigGroup.Head),
            ["nose"] = new Slot(1.15f, RigGroup.Head),
            ["eyewhite"] = new Slot(1.06f, RigGroup.Head),
            ["eyebrow"] = new Slot(1.14f, RigGroup.Head),
            ["irides"] = new Slot(1.08f, RigGroup.Head),
            ["eyelash"] = new Slot(1.12f, RigGroup.Head),
            ["eye_close"] = new Slot(1.12f, RigGroup.Head),
            ["eye_close2"] = new Slot(1.12f, RigGroup.Head),
            ["eyewear"] = new Slot(1.18f, RigGroup.Head),
            ["side hair"] = new Slot(1.22f, RigGroup.Head, Phys.Hair),
            ["front hair"] = new Slot(1.28f, RigGroup.Head, Phys.Hair),
            ["ahoge"] = new Slot(1.30f, RigGroup.Head, Phys.Hair),
        };

        /// The names each slot is known by once folded (see `Fold`): Anime2.5DRig's, with
        /// See-through's among them.
        static readonly Dictionary<string, string> aliases = BuildAliases();

        static Dictionary<string, string> BuildAliases()
        {
            var groups = new (string slot, string names)[]
            {
                ("front hair", "fronthair hairfront bangs fringe 前髪 まえがみ"),
                ("back hair", "backhair hairback 後ろ髪 後髪 うしろがみ"),
                ("side hair", "sidehair hairside sidelock sidelocks 横髪 サイド髪 もみあげ"),
                ("ahoge", "ahoge アホ毛 あほ毛"),
                ("hair", "hair 髪 髪の毛 かみ"),
                ("face", "face 顔 かお 輪郭 facebase"),
                ("facedetail", "facedetail blush cheek cheeks 頬 頬染め チーク"),
                ("eyewhite", "eyewhite eyewhites sclera 白目"),
                ("irides", "irides iris irises pupil pupils 瞳 黒目 虹彩"),
                ("eyelash", "eyelash eyelashes lash lashes eyeline まつ毛 まつげ 睫毛 アイライン"),
                ("eyebrow", "eyebrow eyebrows brow brows 眉 眉毛 まゆ まゆげ"),
                ("eye_close", "eyeclose eyeclosed eyesclosed closedeye closedeyes eyelashc 閉じ目 目閉じ 閉眼"),
                ("mouth_open", "mouthopen openmouth mouth 口 開き口 口開き 開口"),
                ("mouth_close", "mouthclose mouthclosed closedmouth mouthc 閉じ口 口閉じ"),
                ("nose", "nose 鼻 はな"),
                ("ears", "ears ear 耳"),
                ("earwear", "earwear earring earrings イヤリング ピアス 耳飾り"),
                ("neck", "neck 首"),
                ("neckwear", "neckwear necktie tie necklace choker ネクタイ ネックレス チョーカー 首飾り"),
                ("topwear", "topwear top tops shirt body torso clothes 服 上着 体 胴体 上半身 トップス"),
                ("bottomwear", "bottomwear bottom bottoms skirt pants スカート ズボン ボトムス 下半身"),
                ("handwear", "handwear hand hands arm arms gloves 手 腕 手袋"),
                ("legwear", "legwear leg legs socks tights 脚 足 靴下 ソックス タイツ"),
                ("footwear", "footwear shoes shoe boots 靴 ブーツ"),
                ("headwear", "headwear hat cap hairpin hairband 帽子 カチューシャ 髪飾り ヘアピン"),
                ("eyewear", "eyewear glasses 眼鏡 メガネ めがね"),
                ("tail", "tail しっぽ 尻尾 シッポ"),
                ("wings", "wings wing 羽 翼 羽根"),
                ("objects", "objects object 小物 持ち物"),
            };
            var map = new Dictionary<string, string>();
            foreach (var (slot, names) in groups)
                foreach (var name in names.Split(' '))
                    map[Fold(name)] = slot;
            return map;
        }

        /// A layer's name as the aliases are written: compatibility-normalized, lower case,
        /// without a copy's suffix or the marks between words.
        static string Fold(string name)
        {
            string folded = name.Normalize(NormalizationForm.FormKC)
                .ToLowerInvariant().Trim();
            foreach (var suffix in new[] { "のコピー" })
                if (folded.EndsWith(suffix)) folded = folded.Substring(0, folded.Length - suffix.Length);
            var copy = Regex.Match(folded, @"\s+copy(\s+\d+)?$");
            if (copy.Success) folded = folded.Substring(0, copy.Index);
            return new string(folded.Where(c => !" _-・.".Contains(c)).ToArray());
        }

        /// A layer's slot and which eye it is of, by its name: "前髪2" is a piece of `front hair`,
        /// "eye_close2" the long closed eye, See-through's "mouth" an open one. The side is the
        /// character's in the name -- See-through's "-r", "右" -- and the picture's here: the
        /// character's right is on the picture's left. A name no slot has comes back folded, its
        /// own.
        static (string slot, Side? side) SlotOf(string name0)
        {
            string name = name0.Normalize(NormalizationForm.FormKC)
                .ToLowerInvariant().Trim();
            Side? side = null;
            var marks = new (string mark, Side which)[]
            {
                ("-r", Side.Left), ("_r", Side.Left), (" r", Side.Left), (".r", Side.Left),
                (" right", Side.Left), ("右", Side.Left),
                ("-l", Side.Right), ("_l", Side.Right), (" l", Side.Right), (".l", Side.Right),
                (" left", Side.Right), ("左", Side.Right),
            };
            foreach (var (mark, which) in marks)
            {
                if (name.EndsWith(mark))
                {
                    name = name.Substring(0, name.Length - mark.Length);
                    side = which;
                    break;
                }
                if (mark.Length == 1 && name.StartsWith(mark))
                {
                    name = name.Substring(1);
                    side = which;
                    break;
                }
            }
            string folded = Fold(name);
            if (folded == "eyeclose2" || folded == "閉じ目2") return ("eye_close2", side);
            if (aliases.TryGetValue(folded, out string slot)) return (slot, side);
            // A number on the end is another piece of the same thing: "前髪2", "eyelash_2".
            string bare = Regex.Replace(folded, @"\d+$", "");
            if (aliases.TryGetValue(bare, out string slot2)) return (slot2, side);
            return (folded, side);
        }

        /// What a layer is, for the eyes, brows and face, and which eye.
        static (Role, Side?) RoleOf(string name)
        {
            var (slot, side) = SlotOf(name);
            return (RoleOfSlot(slot), side);
        }

        static Role RoleOfSlot(string slot)
        {
            switch (slot)
            {
                case "eyewhite": return Role.EyeWhite;
                case "irides": return Role.Irides;
                case "eyelash": return Role.EyeLash;
                case "eye_close":
                case "eye_close2": return Role.EyeClose;
                case "eyebrow": return Role.Eyebrow;
                case "face": return Role.Face;
                default: return Role.Other;
            }
        }

        /// Whether a layer's name has a number on its end: one strand group among several.
        static bool IsNumbered(string name) => Regex.IsMatch(Fold(name), @"\d+$");

        // MARK: - pixels

        static class Pixels
        {
            /// The box of what is drawn, alpha above `floor`; null for nothing.
            public static Rect? Box(RgbaImage image, byte floor = 16)
            {
                int x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1;
                for (int y = 0; y < image.Height; y++)
                for (int x = 0; x < image.Width; x++)
                {
                    if (image.Rgba[(y * image.Width + x) * 4 + 3] <= floor) continue;
                    x0 = Math.Min(x0, x); x1 = Math.Max(x1, x);
                    y0 = Math.Min(y0, y); y1 = Math.Max(y1, y);
                }
                if (x1 < 0) return null;
                return new Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
            }

            /// Clears every speck: each piece of what is drawn above `floor`, its pixels touching
            /// side to side, smaller than `smallest`. A model's layers are scattered with them,
            /// faint and alone, all over the canvas -- enough to make every layer's box the whole
            /// of it.
            public static void Denoise(RgbaImage image, byte floor = 16, int smallest = 40)
            {
                int w = image.Width, h = image.Height;
                var seen = new bool[w * h];
                var px = image.Rgba;
                var piece = new List<int>();
                for (int start = 0; start < w * h; start++)
                {
                    if (seen[start] || px[start * 4 + 3] <= floor) continue;
                    piece.Clear();
                    piece.Add(start);
                    seen[start] = true;
                    for (int next = 0; next < piece.Count; next++)
                    {
                        int p = piece[next];
                        int x = p % w, y = p / w;
                        if (x > 0 && !seen[p - 1] && px[(p - 1) * 4 + 3] > floor) { seen[p - 1] = true; piece.Add(p - 1); }
                        if (x < w - 1 && !seen[p + 1] && px[(p + 1) * 4 + 3] > floor) { seen[p + 1] = true; piece.Add(p + 1); }
                        if (y > 0 && !seen[p - w] && px[(p - w) * 4 + 3] > floor) { seen[p - w] = true; piece.Add(p - w); }
                        if (y < h - 1 && !seen[p + w] && px[(p + w) * 4 + 3] > floor) { seen[p + w] = true; piece.Add(p + w); }
                    }
                    if (piece.Count < smallest)
                        foreach (int p in piece) px[p * 4 + 3] = 0;
                }
            }

            /// The middle of what is drawn above `floor`, each pixel weighed by its alpha.
            public static Point? Centroid(RgbaImage image, byte floor = 16)
            {
                double sx = 0, sy = 0, sw = 0;
                for (int y = 0; y < image.Height; y++)
                for (int x = 0; x < image.Width; x++)
                {
                    byte alpha = image.Rgba[(y * image.Width + x) * 4 + 3];
                    if (alpha <= floor) continue;
                    sx += alpha * (x + 0.5);
                    sy += alpha * (y + 0.5);
                    sw += alpha;
                }
                return sw > 0 ? new Point(sx / sw, sy / sw) : (Point?)null;
            }

            /// The colour of what is drawn, weighed by alpha: 0 to 1 each.
            public static (double, double, double)? MeanColour(RgbaImage image)
            {
                double r = 0, g = 0, b = 0, w = 0;
                for (int i = 0; i < image.Rgba.Length; i += 4)
                {
                    double a = image.Rgba[i + 3];
                    r += a * image.Rgba[i];
                    g += a * image.Rgba[i + 1];
                    b += a * image.Rgba[i + 2];
                    w += a;
                }
                return w > 0 ? ((r / w / 255, g / w / 255, b / w / 255)) : ((double, double, double)?)null;
            }

            /// The part of it inside `box`, which is in its own pixels.
            public static RgbaImage Crop(RgbaImage image, Rect box)
            {
                int x0 = (int)box.MinX, y0 = (int)box.MinY, w = (int)box.Width, h = (int)box.Height;
                var rgba = new byte[w * h * 4];
                for (int y = 0; y < h; y++)
                {
                    int from = ((y0 + y) * image.Width + x0) * 4;
                    Buffer.BlockCopy(image.Rgba, from, rgba, y * w * 4, w * 4);
                }
                return new RgbaImage(w, h, rgba);
            }
        }

        // MARK: - strands

        /// Where a layer of hair hangs in strands: Anime2.5DRig's `detectStrands`. The bottom edge
        /// of what is drawn, column by column, smoothed over 41 columns, has a peak at each
        /// strand's tip: up to `want` of the most prominent, at least `minSep` apart, and more
        /// where the layer is wide enough that a column is that far from every one. Each strand
        /// runs from the top of its column to the bottom, in the layer's own pixels.
        static List<Strand> Strands(RgbaImage image, int want, int minSep)
        {
            int w = image.Width, h = image.Height;
            var top = new int[w];
            var bottom = new int[w];
            for (int x = 0; x < w; x++)
            {
                top[x] = -1;
                for (int y = 0; y < h; y++)
                {
                    if (image.Rgba[(y * w + x) * 4 + 3] <= 16) continue;
                    if (top[x] < 0) top[x] = y;
                    bottom[x] = y;
                }
            }
            var painted = Enumerable.Range(0, w).Where(x => top[x] >= 0).ToList();
            if (painted.Count == 0) return new List<Strand>();
            int minX = painted[0], maxX = painted[painted.Count - 1];

            // Box-smoothed over 41 columns, always divided by 41, beyond the edges 0.
            var smoothed = new double[w];
            for (int x = 0; x < w; x++)
            {
                double sum = 0;
                for (int k = -20; k <= 20; k++)
                    if (x + k >= 0 && x + k < w) sum += bottom[x + k];
                smoothed[x] = sum / 41;
            }
            var picked = new List<int>();
            foreach (int peak in Peaks(smoothed, minSep, 10))
            {
                if (picked.Count >= want) break;
                picked.Add(peak);
            }

            // More, at whichever of 40 columns across it is farthest from every strand, while that
            // is at least minSep away.
            int inset = Math.Min(30, (maxX - minX) / 4);
            var candidates = Enumerable.Range(0, 40).Select(k =>
            {
                int x = minX + inset + (int)Math.Round((maxX - minX - 2 * inset) * (double)k / 39);
                return painted.OrderBy(p => Math.Abs(p - x)).First();
            }).ToList();
            for (int i = 0; i < 50 && picked.Count < want; i++)
            {
                int Distance(int x) => picked.Count == 0 ? int.MaxValue : picked.Min(p => Math.Abs(p - x));
                int far = -1, farD = -1;
                foreach (int c in candidates)
                {
                    int d = Distance(c);
                    if (d > farD) { farD = d; far = c; }
                }
                if (far < 0 || farD < Math.Max(1, minSep)) break;
                picked.Add(far);
            }
            var columns = picked.Select(x => painted.OrderBy(p => Math.Abs(p - x)).First())
                .Distinct().OrderBy(x => x).ToList();
            return columns.Select(x => new Strand { X = x, RootY = top[x], TipY = bottom[x] }).ToList();
        }

        /// The peaks of `a`, most prominent first, none within `minSep` of one kept: a rise to a
        /// flat top counts at the top's middle, unless it rises again after. A peak's prominence
        /// is how far it stands over the higher of the lowest points on either side before
        /// something higher.
        static List<int> Peaks(double[] a, int minSep, double minProminence)
        {
            var found = new List<(int p, double prom)>();
            int i = 1;
            while (i < a.Length - 1)
            {
                if (a[i] <= a[i - 1]) { i++; continue; }
                int j = i;
                while (j + 1 < a.Length && a[j + 1] == a[i]) j++;
                if (j + 1 < a.Length && a[j + 1] < a[i])
                {
                    int p = (i + j) / 2;
                    double left = a[p], right = a[p];
                    for (int k = p - 1; k >= 0 && a[k] <= a[p]; k--) left = Math.Min(left, a[k]);
                    for (int k = p + 1; k < a.Length && a[k] <= a[p]; k++) right = Math.Min(right, a[k]);
                    double prominence = a[p] - Math.Max(left, right);
                    if (prominence >= minProminence) found.Add((p, prominence));
                }
                i = j + 1;
            }
            var kept = new List<int>();
            foreach (var (p, _) in found.OrderByDescending(f => f.prom))
                if (kept.All(k => Math.Abs(k - p) >= minSep)) kept.Add(p);
            return kept;
        }

        /// Each vertex's weight towards each strand -- a Gaussian over 0.6 of the strands' median
        /// spacing, summing to 1 -- and how far down the weighted strand it is.
        static Swing MakeSwing(Phys kind, bool front, List<Strand> strands, RigMesh mesh)
        {
            var gaps = strands.Skip(1).Zip(strands, (a, b) => a.X - b.X).OrderBy(g => g).ToList();
            float spacing = gaps.Count == 0 ? 120 : gaps[gaps.Count / 2];
            float sigma = Math.Max(1, spacing * 0.6f);
            var weights = new List<float>();
            var along = new List<float>();
            foreach (var p in mesh.Rest)
            {
                var w = strands.Select(s => (float)Math.Exp(-Math.Pow((s.X - p.X) / sigma, 2))).ToArray();
                float sum = w.Sum();
                if (sum < 1e-6f)
                    for (int i = 0; i < w.Length; i++) w[i] = i == 0 ? 1 : 0;
                else
                    for (int i = 0; i < w.Length; i++) w[i] /= sum;
                float root = w.Zip(strands, (a, s) => a * s.RootY).Sum();
                float tip = w.Zip(strands, (a, s) => a * s.TipY).Sum();
                weights.AddRange(w);
                along.Add(Math.Min(Math.Max((p.Y - root) / Math.Max(1, tip - root), 0), 1));
            }
            return new Swing
            {
                Kind = kind,
                Front = front,
                Strands = strands.ToArray(),
                Weights = weights.ToArray(),
                Along = along.ToArray(),
            };
        }

        // MARK: - building

        /// What moves in `document`, and how. Anything it does not know is drawn as it is, with
        /// the head or the body by where it is.
        public static Rig Build(PsdDocument document)
        {
            // Each shown layer, trimmed to what is drawn.
            var pieces = new List<Piece>();
            foreach (var layer in document.Layers.Where(l => l.Visible))
            {
                var image = layer.Image;
                Pixels.Denoise(image);
                var box = Pixels.Box(image);
                if (box == null) continue;
                var (slot, side) = SlotOf(layer.Name);
                pieces.Add(new Piece
                {
                    Name = layer.Name,
                    Slot = slot,
                    Side = side,
                    Left = layer.Left + (int)box.Value.MinX,
                    Top = layer.Top + (int)box.Value.MinY,
                    Image = Pixels.Crop(image, box.Value),
                    Opacity = layer.Opacity,
                });
            }
            // A plain "hair" is the front hair above the face and the back hair under it.
            int faceAt = pieces.FindLastIndex(p => p.Slot == "face");
            if (faceAt >= 0)
                for (int i = 0; i < pieces.Count; i++)
                    if (pieces[i].Slot == "hair")
                        pieces[i].Slot = i > faceAt ? "front hair" : "back hair";

            // The face: its box and middle, or where a face would be.
            var face = pieces.LastOrDefault(p => p.Slot == "face");
            var faceBox = face != null
                ? Pixels.Box(face.Image, 8)?.Offset(face.Left, face.Top)
                : null;
            var faceRect = faceBox ?? new Rect(
                0.35 * document.Width, 0.1 * document.Height,
                0.3 * document.Width, 0.4 * document.Height);
            var faceCenter = face != null
                ? Pixels.Centroid(face.Image).Let(c => new Point(c.X + face.Left, c.Y + face.Top))
                : null;
            var faceMid = faceCenter ?? new Point(document.Width / 2.0, 0.3 * document.Height);

            // Eyes drawn as one layer each, cut at the middle of the face.
            var eyeish = new HashSet<string> { "eyewhite", "irides", "eyelash", "eye_close", "eye_close2", "eyebrow" };
            var cut = new List<Piece>();
            foreach (var piece in pieces)
            {
                if (!eyeish.Contains(piece.Slot) || piece.Side != null)
                {
                    cut.Add(piece);
                    continue;
                }
                int at = Math.Min(Math.Max(0, (int)Math.Round(faceMid.X) - piece.Left), piece.Image.Width);
                foreach (var (side, from, to) in new[] { (Side.Left, 0, at), (Side.Right, at, piece.Image.Width) })
                {
                    if (to <= from) continue;
                    var half = Pixels.Crop(piece.Image, new Rect(from, 0, to - from, piece.Image.Height));
                    var box = Pixels.Box(half);
                    if (box == null) continue;
                    cut.Add(new Piece
                    {
                        Name = piece.Name,
                        Slot = piece.Slot,
                        Side = side,
                        Left = piece.Left + from + (int)box.Value.MinX,
                        Top = piece.Top + (int)box.Value.MinY,
                        Image = Pixels.Crop(half, box.Value),
                        Opacity = piece.Opacity,
                    });
                }
            }
            pieces = cut;

            // Each eye's anchor: its white's box, its iris's middle, and where it closes -- its
            // own closed eye's middle, or 62 % of the way down its white.
            var eyes = new Dictionary<Side, EyeAnchor>();
            var notes = new List<string>();
            foreach (var side in new[] { Side.Left, Side.Right })
            {
                var white = pieces.LastOrDefault(p => p.Slot == "eyewhite" && p.Side == side);
                var box = white != null ? Pixels.Box(white.Image) : null;
                if (box == null) continue;
                var rect = box.Value.Offset(white.Left, white.Top);
                var iris = pieces.LastOrDefault(p => p.Slot == "irides" && p.Side == side);
                var middle = iris != null
                    ? Pixels.Centroid(iris.Image).Let(c => new Point(c.X + iris.Left, c.Y + iris.Top))
                    : null;
                var mid = middle ?? new Point(rect.MidX, rect.MidY);
                var closed = pieces.LastOrDefault(p => p.Slot == "eye_close" && p.Side == side);
                var closeY = closed != null
                    ? Pixels.Centroid(closed.Image).Let(c => c.Y + closed.Top)
                    : null;
                eyes[side] = new EyeAnchor
                {
                    White = rect,
                    Iris = mid,
                    CloseY = closeY ?? rect.MinY + 0.62 * rect.Height,
                };
            }
            if (eyes.Count == 0)
                notes.Add("No eye white layer (eyewhite, 白目): the eyes cannot blink.");

            // The neck the head turns on: 85 % down it, or under the face where there is none.
            var neck = pieces.LastOrDefault(p => p.Slot == "neck");
            var neckBox = neck != null ? Pixels.Box(neck.Image, 8)?.Offset(neck.Left, neck.Top) : null;
            var neckCenterX = neck != null
                ? Pixels.Centroid(neck.Image).Let(c => c.X + neck.Left)
                : null;
            var neckPivot = neckBox != null
                ? new Point(neckCenterX ?? neckBox.Value.MidX, neckBox.Value.MinY + 0.85 * neckBox.Value.Height)
                : new Point(faceMid.X, faceRect.MaxY + 20);
            double neckTop = neckBox?.MinY ?? faceRect.MaxY;
            double neckBottom = neckBox?.MaxY ?? faceRect.MaxY + 60;

            // A closed eye for each eye without one: the stock drawing, 1.1 times as wide as the
            // white, its middle on the white's, 55 % of it above the line it closes onto, in the
            // colour of that eye's lashes (or brow).
            var made = new List<Piece>();
            foreach (var side in new[] { Side.Left, Side.Right })
            {
                if (!eyes.TryGetValue(side, out var eye)) continue;
                if (pieces.Any(p => p.Slot == "eye_close" && p.Side == side)) continue;
                var lash = pieces.LastOrDefault(p => p.Slot == "eyelash" && p.Side == side)
                    ?? pieces.LastOrDefault(p => p.Slot == "eyebrow" && p.Side == side);
                var tint = lash != null ? Pixels.MeanColour(lash.Image) : null;
                var tintC = tint ?? (0.2, 0.15, 0.15);
                var drawing = ClosedEye(side, eye.White.Width * 1.1, tintC);
                if (drawing == null) continue;
                made.Add(new Piece
                {
                    Name = "eye_close (made)",
                    Slot = "eye_close",
                    Side = side,
                    Left = (int)Math.Round(eye.White.MidX - drawing.Width / 2.0),
                    Top = (int)Math.Round(eye.CloseY - 0.55 * drawing.Height),
                    Image = drawing,
                    Opacity = 1,
                });
            }
            if (made.Count > 0)
            {
                // Over the lashes, or the irides, or the face: where a closed eye would be drawn.
                int at = pieces.FindLastIndex(p => p.Slot == "eyelash");
                if (at < 0) at = pieces.FindLastIndex(p => p.Slot == "irides");
                if (at < 0) at = pieces.FindLastIndex(p => p.Slot == "face");
                pieces.InsertRange(at >= 0 ? at + 1 : pieces.Count, made);
            }

            var parts = pieces.Select(piece =>
            {
                var rect = new Rect(piece.Left, piece.Top, piece.Image.Width, piece.Image.Height);
                slots.TryGetValue(piece.Slot, out var slot);
                // One no slot places is the head's above the bottom of the face and the body's
                // under it.
                RigGroup group;
                if (slot?.Group != null)
                    group = slot.Group.Value;
                else
                {
                    var cy = Pixels.Centroid(piece.Image).Let(c => c.Y + piece.Top) ?? rect.MidY;
                    group = cy < faceRect.MaxY ? RigGroup.Head : RigGroup.Body;
                }
                var mesh = RigMesh.Grid(rect, document.Width, swings: slot?.Physics != null);
                Swing swing = null;
                if (slot?.Physics is Phys phys)
                {
                    int width = piece.Image.Width;
                    int want, minSep;
                    if (phys == Phys.Hair)
                    {
                        want = IsNumbered(piece.Name)
                            ? Math.Min(Math.Max((int)Math.Round(width / 110.0), 2), 6)
                            : 6;
                        minSep = Math.Max(30, (int)Math.Round(width / (want * 1.6)));
                    }
                    else
                    {
                        want = Math.Min(Math.Max((int)Math.Round(width / 140.0), 1), 4);
                        minSep = Math.Max(20, (int)Math.Round(width / (want * 1.6)));
                    }
                    var found = Strands(piece.Image, want, minSep).Select(s => new Strand
                    {
                        X = s.X + piece.Left,
                        RootY = s.RootY + piece.Top,
                        TipY = s.TipY + piece.Top,
                    }).ToList();
                    if (found.Count > 0)
                    {
                        bool front = piece.Slot == "front hair" || piece.Slot == "side hair" || piece.Slot == "ahoge";
                        swing = MakeSwing(phys, front, found, mesh);
                    }
                }
                return new RigPart
                {
                    Name = piece.Name,
                    Slot = piece.Slot,
                    Role = RoleOfSlot(piece.Slot),
                    Side = piece.Side,
                    Rect = rect,
                    Pixels = piece.Image,
                    Opacity = piece.Opacity,
                    Depth = slot?.Depth ?? 1,
                    Group = group,
                    Mesh = mesh,
                    Swing = swing,
                };
            }).ToList();

            double faceHeight = faceRect.Height;
            var anchors = new Anchors
            {
                Face = faceRect,
                FaceCenter = faceMid,
                NeckPivot = neckPivot,
                NeckTop = neckTop,
                NeckBottom = neckBottom,
                BodyPivot = new Point(neckPivot.X, document.Height),
                Chest = new Point(neckPivot.X, neckBottom + faceHeight * 0.60),
                ChestReach = new Size(Math.Max(1, faceRect.Width * 0.60), Math.Max(1, faceHeight * 0.45)),
            };
            double faceWidth = face != null
                ? Pixels.Box(face.Image)?.Width ?? document.Width / 3.0
                : document.Width / 3.0;
            return new Rig
            {
                Width = document.Width,
                Height = document.Height,
                Parts = parts,
                Eyes = eyes,
                Anchors = anchors,
                FaceScale = faceWidth / 333,
                Notes = notes,
            };
        }

        /// Each shown layer, trimmed to what is drawn.
        sealed class Piece
        {
            public string Name;
            public string Slot;
            public Side? Side;
            public int Left;
            public int Top;
            public RgbaImage Image;
            public double Opacity;
        }

        /// The stock closed eye for `side`, scaled to `width` and painted `tint`.
        static RgbaImage ClosedEye(Side side, double width, (double, double, double) tint) =>
            Stock(side == Side.Left ? ClosedParts.Left : ClosedParts.Right, width, tint);

        /// A stock drawing's ink scaled to `width` and painted `tint`, each pixel as light as it
        /// was against the ink's mean, at most 2.2 times; the skin it is edged in left out.
        static RgbaImage Stock(ClosedParts.Drawing drawing, double width, (double, double, double) tint)
        {
            var source = drawing.GetRgba();
            if (source == null || width < 1) return null;
            int w = Math.Max(1, (int)Math.Round(width));
            int h = Math.Max(1, (int)Math.Round(drawing.Height * width / drawing.Width));
            double Luma(int i) => (0.299 * source[i] + 0.587 * source[i + 1] + 0.114 * source[i + 2]) / 255;
            // How much of a pixel is ink: the drawings are edged in skin -- a line's soft edge, a
            // lower lip's highlight -- which, tinted, would be a red blot on the face. Skin, at a
            // luma of 0.85 or more, is left out; the line, under 0.5, is kept whole.
            double Ink(double luma) => Smoothing.Smooth((0.85 - luma) / 0.35);
            double sum = 0, weight = 0;
            for (int i = 0; i < source.Length; i += 4)
            {
                double a = source[i + 3] * Ink(Luma(i));
                sum += a * Luma(i);
                weight += a;
            }
            double mean = weight > 0 ? Math.Max(sum / weight, 1e-3) : 1;
            // Bilinear, in premultiplied alpha so that the edge does not darken, then back.
            var out_ = new byte[w * h * 4];
            double Sample(double x, double y, int c)
            {
                double fx = Math.Min(Math.Max(x, 0), drawing.Width - 1);
                double fy = Math.Min(Math.Max(y, 0), drawing.Height - 1);
                int x0 = (int)fx, y0 = (int)fy;
                int x1 = Math.Min(x0 + 1, drawing.Width - 1), y1 = Math.Min(y0 + 1, drawing.Height - 1);
                double tx = fx - x0, ty = fy - y0;
                double At(int x_, int y_)
                {
                    int i = (y_ * drawing.Width + x_) * 4;
                    double a = source[i + 3] / 255.0;
                    return c == 3 ? a : a * Luma(i);
                }
                return (At(x0, y0) * (1 - tx) + At(x1, y0) * tx) * (1 - ty)
                    + (At(x0, y1) * (1 - tx) + At(x1, y1) * tx) * ty;
            }
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double sx = (x + 0.5) * drawing.Width / (double)w - 0.5;
                double sy = (y + 0.5) * drawing.Height / (double)h - 0.5;
                double covered = Sample(sx, sy, 3);
                if (covered <= 0) continue;
                double lightness = Sample(sx, sy, 0) / covered;
                double a = covered * Ink(lightness);
                if (a <= 0) continue;
                double shade = Math.Min(2.2, lightness / mean);
                int i = (y * w + x) * 4;
                out_[i] = (byte)Math.Min(255, Math.Round(tint.Item1 * shade * 255));
                out_[i + 1] = (byte)Math.Min(255, Math.Round(tint.Item2 * shade * 255));
                out_[i + 2] = (byte)Math.Min(255, Math.Round(tint.Item3 * shade * 255));
                out_[i + 3] = (byte)Math.Min(255, Math.Round(a * 255));
            }
            return new RgbaImage(w, h, out_);
        }
    }

    static class RgbaExt
    {
        /// `f(v)` where `v` has a value; null where it does not.
        public static TOut? Let<TIn, TOut>(this TIn? v, Func<TIn, TOut> f)
            where TIn : struct where TOut : struct
            => v.HasValue ? f(v.Value) : (TOut?)null;
    }

    /// How open the eyes are from moment to moment: Anime2.5DRig's blink. Closed in 80 ms, held
    /// 340 ms, open in 160 ms; the next one 1.6 to 5.4 s after this one began, or -- one time in
    /// six or so -- right after it; the first 1.8 s in. Every change is eased by a low-pass of
    /// 71 ms, and an eye just opened gives its iris a small bounce.
    sealed class Blinker
    {
        /// Seconds since it began, at its last step.
        public double Now { get; private set; }
        double nextBlink = 1.8;
        /// Seconds into the blink going on, or null.
        double? blinking;
        bool bounced;
        double? bounceTime;
        /// How open the eyes are, 0 shut to 1 open, eased.
        public double Openness { get; private set; } = 1.0;
        /// The iris's scale across and down, from its bounce.
        public Size Bounce { get; private set; } = new Size(1, 1);
        readonly Random rng = new Random();

        /// Blinks once, now, whatever the clock says.
        public void BlinkNow()
        {
            if (blinking == null)
            {
                blinking = 0;
                bounced = false;
            }
        }

        /// Where the blink is `seconds` into it: how open the eyes are, or null once it is over.
        static double? Shape(double seconds)
        {
            const double hold = 0.34;
            if (seconds < 0.08) return 1 - seconds / 0.08;
            if (seconds < 0.08 + hold) return 0;
            if (seconds < 0.24 + hold) return (seconds - 0.08 - hold) / 0.16;
            return null;
        }

        /// Moves on by `dt` seconds. `open` is as open as the eyes may be; `automatic` blinks on
        /// its own.
        public void Step(double dt, double open, bool automatic)
        {
            dt = Math.Max(0, Math.Min(dt, 0.25));
            Now += dt;
            if (automatic && blinking == null && Now > nextBlink)
            {
                blinking = 0;
                bounced = false;
                nextBlink = Now + 1.6 + rng.NextDouble() * 3.8;
                if (rng.NextDouble() < 0.18) nextBlink = Now + 0.28;
            }
            double target = open;
            if (blinking is double bt)
            {
                bt += dt;
                var v = Shape(bt);
                if (v.HasValue)
                {
                    blinking = bt;
                    target = Math.Min(target, v.Value);
                    if (bt >= 0.08 + 0.34 && v.Value > 0.12 && !bounced)
                    {
                        bounced = true;
                        bounceTime = 0;
                    }
                }
                else
                {
                    blinking = null;
                }
            }
            Openness += (target - Openness) * (1 - Math.Exp(-dt * 14));

            Bounce = new Size(1, 1);
            if (bounceTime is double t0)
            {
                t0 += dt;
                double p = t0 / 0.52;
                if (p >= 1)
                {
                    bounceTime = null;
                }
                else
                {
                    bounceTime = t0;
                    double damp = Math.Exp(-2.2 * p);
                    double scale = 1 + 0.18 * Math.Sin(4 * Math.PI * p) * damp;
                    double squash = 0.10 * Math.Sin(4 * Math.PI * p + Math.PI / 2) * damp;
                    Bounce = new Size(scale * (1 + squash), scale * (1 - squash));
                }
            }
        }
    }

    /// Smoothstep, clamped: 0 below 0, 1 above 1.
    static class Smoothing
    {
        public static double Smooth(double t)
        {
            t = Math.Min(Math.Max(t, 0), 1);
            return t * t * (3 - 2 * t);
        }
    }
}
