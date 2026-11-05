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
// A rig drawn with Skia, as waifu-app/swiftui's RigMetal draws one with Metal (and as
// Anime2.5DRig draws one with WebGL, lib/renderer.js): every layer a grid of triangles over its
// picture, moved vertex by vertex on the CPU, in premultiplied alpha.
//
// The eye whites are drawn first into a mask of their own -- the left one's into one surface,
// the right one's into another -- which the irises are cut by as they are drawn (a layer masked
// by the surface, destination-in), so that an iris stays inside its white with a smooth edge
// whatever order the layers are in and however faint the white is.
//
// A port of waifu-app/swiftui/Sources/Waifu/RigMetal.swift.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using SkiaSharp;

namespace Libwaifu.Gui.Waifu2d
{
    sealed class RigRenderer : IDisposable
    {
        sealed class Mesh : IDisposable
        {
            public SKBitmap Texture;
            public System.Runtime.InteropServices.GCHandle Pixels;
            public SKPoint[] Uv;
            public ushort[] Indices;
            public void Dispose() { Texture?.Dispose(); if (Pixels.IsAllocated) Pixels.Free(); }
        }

        Rig rig;
        readonly List<Mesh> meshes = new List<Mesh>();
        /// The whites, left and right, as mask surfaces the size of the target.
        SKSurface maskL, maskR;
        int maskW, maskH;

        public Rig Rig => rig;

        /// Hands `rig`'s pictures and grids to the renderer, once: its pictures premultiplied.
        public void Load(Rig value)
        {
            DisposeMeshes();
            rig = value;
            foreach (var part in rig.Parts)
            {
                int w = Math.Max(1, part.Pixels.Width), h = Math.Max(1, part.Pixels.Height);
                var premultiplied = new byte[w * h * 4];
                if (part.Pixels.Rgba.Length == w * h * 4)
                {
                    for (int i = 0; i < premultiplied.Length; i += 4)
                    {
                        int a = part.Pixels.Rgba[i + 3];
                        premultiplied[i] = (byte)((part.Pixels.Rgba[i] * a + 127) / 255);
                        premultiplied[i + 1] = (byte)((part.Pixels.Rgba[i + 1] * a + 127) / 255);
                        premultiplied[i + 2] = (byte)((part.Pixels.Rgba[i + 2] * a + 127) / 255);
                        premultiplied[i + 3] = (byte)a;
                    }
                }
                var info = new SKImageInfo(w, h, SKColorType.Rgba8888, SKAlphaType.Premul);
                var texture = new SKBitmap(info);
                var handle = System.Runtime.InteropServices.GCHandle.Alloc(premultiplied,
                    System.Runtime.InteropServices.GCHandleType.Pinned);
                texture.InstallPixels(info, handle.AddrOfPinnedObject());

                meshes.Add(new Mesh
                {
                    Texture = texture,
                    Pixels = handle,
                    // Skia takes its texture coordinates in pixels; the mesh keeps them 0 to 1.
                    Uv = part.Mesh.Uv.Select(p => new SKPoint(p.X * w, p.Y * h)).ToArray(),
                    Indices = part.Mesh.Indices.Select(i => (ushort)i).ToArray(),
                });
            }
        }

        void DisposeMeshes()
        {
            foreach (var m in meshes) m.Dispose();
            meshes.Clear();
            rig = null;
        }

        /// The canvas fitted into `size` pixels, in the middle of it.
        public static SKRect Fit(float canvasW, float canvasH, float width, float height)
        {
            float scale = Math.Min(width / Math.Max(1, canvasW), height / Math.Max(1, canvasH));
            float w = canvasW * scale, h = canvasH * scale;
            return new SKRect((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2);
        }

        /// Encodes `frame` of the loaded rig into `canvas`, fitted into it, over `clear`.
        public void Encode(SKCanvas canvas, RigFrame frame, int targetW, int targetH, SKColor clear)
        {
            if (rig == null || meshes.Count != rig.Parts.Count || frame.Positions.Count != meshes.Count)
                return;
            var fitted = Fit(rig.Width, rig.Height, targetW, targetH);
            var xform = SKMatrix.CreateScaleTranslation(
                fitted.Width / rig.Width, fitted.Height / rig.Height, fitted.Left, fitted.Top);

            // The whites, into the mask surfaces, as they are now -- whether they show or not --
            // where an iris will be drawn that needs one.
            bool clipped = Enumerable.Range(0, rig.Parts.Count).Any(i =>
                rig.Parts[i].Role == Role.Irides && rig.Parts[i].Side != null && frame.Alphas[i] >= 0.004);
            var masked = new HashSet<Side>();
            if (clipped)
            {
                var maskFor = MaskSurfaces(targetW, targetH);
                foreach (var side in new[] { Side.Left, Side.Right })
                {
                    var surface = side == Side.Left ? maskFor.Item1 : maskFor.Item2;
                    if (surface == null) continue;
                    var mc = surface.Canvas;
                    mc.Clear(SKColors.Transparent);
                    mc.Save();
                    mc.Concat(xform);
                    for (int i = 0; i < rig.Parts.Count; i++)
                    {
                        var part = rig.Parts[i];
                        if (part.Role != Role.EyeWhite || part.Side != side) continue;
                        DrawMesh(mc, i, frame.Positions[i], 1f);
                        masked.Add(side);
                    }
                    mc.Restore();
                    mc.Flush();
                }
            }

            canvas.Clear(clear);
            canvas.Save();
            canvas.Concat(xform);
            for (int i = 0; i < rig.Parts.Count; i++)
            {
                if (frame.Alphas[i] < 0.004) continue;
                var part = rig.Parts[i];
                if (part.Role == Role.Irides && part.Side is Side side && masked.Contains(side))
                {
                    // The iris, in a layer cut by its white's mask.
                    canvas.SaveLayer((SKPaint)null);
                    DrawMesh(canvas, i, frame.Positions[i], frame.Alphas[i]);
                    using (var paint = new SKPaint { BlendMode = SKBlendMode.DstIn, IsAntialias = true })
                    using (var snapshot = (side == Side.Left ? maskL : maskR).Snapshot())
                    {
                        canvas.Save();
                        // Undo the canvas transform: the mask is in target pixels.
                        canvas.ResetMatrix();
                        canvas.DrawImage(snapshot, new SKRect(0, 0, targetW, targetH), paint);
                        canvas.Restore();
                    }
                    canvas.Restore();
                }
                else
                {
                    DrawMesh(canvas, i, frame.Positions[i], frame.Alphas[i]);
                }
            }
            canvas.Restore();
        }

        /// One part's triangles textured by its picture.
        void DrawMesh(SKCanvas canvas, int i, Vector2[] positions, float alpha)
        {
            var mesh = meshes[i];
            if (mesh.Indices.Length == 0) return;
            var points = positions.Select(p => new SKPoint(p.X, p.Y)).ToArray();
            using (var vertices = SKVertices.CreateCopy(
                SKVertexMode.Triangles, points, mesh.Uv, null, mesh.Indices))
            using (var paint = new SKPaint())
            {
                paint.Shader = mesh.Texture.ToShader(SKShaderTileMode.Clamp, SKShaderTileMode.Clamp);
                paint.Color = new SKColor(255, 255, 255, (byte)Math.Min(255, Math.Max(0, alpha * 255 + 0.5f)));
                paint.IsAntialias = true;
                // DrawVertices grew no SKSamplingOptions overload in SkiaSharp 3, so the old
                // paint property is still the only way to ask for the better filter.
#pragma warning disable CS0618
                paint.FilterQuality = SKFilterQuality.High;
#pragma warning restore CS0618
                canvas.DrawVertices(vertices, SKBlendMode.Modulate, paint);
            }
        }

        /// The mask surfaces, the size of what is drawn into: the left white's and the right's.
        (SKSurface, SKSurface) MaskSurfaces(int w, int h)
        {
            if (maskL != null && (maskW != w || maskH != h))
            {
                maskL.Dispose(); maskL = null;
                maskR?.Dispose(); maskR = null;
            }
            if (maskL == null)
            {
                maskW = w; maskH = h;
                var info = new SKImageInfo(w, h, SKColorType.Rgba8888, SKAlphaType.Premul);
                maskL = SKSurface.Create(info);
                maskR = SKSurface.Create(info);
            }
            return (maskL, maskR);
        }

        public void Dispose()
        {
            DisposeMeshes();
            maskL?.Dispose();
            maskR?.Dispose();
        }
    }
}
