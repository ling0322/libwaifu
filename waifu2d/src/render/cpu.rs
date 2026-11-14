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

//! The backend contract, executable: a rasterizer on the CPU that every GPU backend is held to.
//!
//! It draws what a GPU draws, the way one does:
//!
//! - **Vertices** go from canvas pixels to clip space in `f32`, as the vertex shader does, then
//!   through the viewport to target pixels, and are snapped to a grid of 1/256 of a pixel -- the
//!   8 bits of sub-pixel precision Direct3D 11 asks for -- so that edges are decided in exact
//!   integer arithmetic.
//! - **Coverage**: a pixel is drawn when its centre, at `+0.5`, is inside the triangle, with the
//!   top-left rule for a centre on an edge, and inside the viewport, which clips as clip space
//!   does. Either winding; nothing is culled.
//! - **Sampling**: trilinear, clamp to edge, at the level of detail a GPU picks. The mapping from
//!   a triangle to its picture is affine, so the derivatives of `uv * size` (level 0's texels)
//!   across and down the target are the same over the whole triangle: one LOD a triangle, `log2`
//!   of the longest stretch of a pixel's footprint -- the larger singular value of those
//!   derivatives -- clamped to the chain; at or under 0 it is level 0, bilinear. That is what an
//!   Apple GPU picks, as near as it was measured. Where a picture is only scaled it is the
//!   textbook `max(|d/dx|, |d/dy|)`; where it is squeezed and also turned or sheared, as a bent
//!   grid is, the GPU follows the footprint's long axis -- a little short of it -- and the
//!   textbook would pick a level up to a quarter of one finer, enough to fail conformance.
//! - **Blending**: premultiplied, One and OneMinusSourceAlpha, in `f32` on what the target holds,
//!   which is 8 bits a channel: every draw's result is rounded to the nearest of 255 steps,
//!   `floor(v * 255 + 0.5)`, clamped to 0..1 first, as a GPU writes a UNORM8 target. The clear
//!   colour is rounded the same way.
//! - **Masks**: one R8 coverage image a mask the size of the target -- layers of one array, grown
//!   when a frame has more masks than it -- cleared to 0, its parts' texel alpha blended in as
//!   above. A clipped draw reads its mask's coverage at its own pixel.

use std::collections::HashMap;

use super::{Api, Assets, Backend, DrawList, Loaded, PartAssets, Surface};
use crate::Error;

/// Sub-pixel steps a pixel: where a vertex is snapped to.
const STEPS: i64 = 256;
const SUBPIXEL: f32 = STEPS as f32;

pub struct CpuBackend {
    size: [u32; 2],
    assets: Option<(Assets, Loaded)>,
    /// The target, RGBA8 premultiplied, top row first.
    target: Vec<u8>,
    /// The masks, `layers` of them, each `size[0] * size[1]` coverage bytes.
    masks: Vec<u8>,
    layers: usize,
}

impl CpuBackend {
    pub fn new(surface: Surface) -> Result<CpuBackend, Error> {
        match surface {
            Surface::Offscreen { width, height } => Ok(CpuBackend {
                size: [width, height],
                assets: None,
                target: vec![0; width as usize * height as usize * 4],
                masks: Vec::new(),
                layers: 0,
            }),
            _ => Err(Error::Unsupported(
                "the CPU backend draws offscreen only".into(),
            )),
        }
    }
}

impl Backend for CpuBackend {
    fn api(&self) -> Api {
        Api::Cpu
    }

    fn load(&mut self, assets: &Assets) -> Result<(), Error> {
        self.assets = Some((assets.clone(), Loaded::of(assets)));
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        self.size = [width, height];
        self.target = vec![0; width as usize * height as usize * 4];
        self.masks = Vec::new();
        self.layers = 0;
        Ok(())
    }

    fn draw(&mut self, list: &DrawList, clear: [f32; 4]) -> Result<(), Error> {
        if list.target != self.size {
            return Err(Error::InvalidArgument(format!(
                "a frame for {:?} pixels drawn into {:?}",
                list.target, self.size
            )));
        }
        let (assets, loaded) = self
            .assets
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument("drawn before a rig was loaded".into()))?;
        loaded.check(list)?;
        let [w, h] = [self.size[0] as usize, self.size[1] as usize];

        // The masks, each into a layer of its own.
        if self.layers < list.masks.len() {
            self.layers = list.masks.len();
            self.masks = vec![0; self.layers * w * h];
        }
        let mut layer_of = HashMap::new();
        for (layer, mask) in list.masks.iter().enumerate() {
            layer_of.insert(mask.id, layer);
            let coverage = &mut self.masks[layer * w * h..(layer + 1) * w * h];
            coverage.fill(0);
            for &part in &mask.parts {
                let vertices = list.frame.vertices(part);
                rasterize(
                    list,
                    &assets.parts[part.index()],
                    vertices,
                    |x, y, texel| {
                        let m = &mut coverage[y * w + x];
                        *m = unorm8(texel[3] + *m as f32 / 255.0 * (1.0 - texel[3]));
                    },
                );
            }
        }

        // Then the parts, over the clear colour.
        let clear = clear.map(unorm8);
        for pixel in self.target.chunks_exact_mut(4) {
            pixel.copy_from_slice(&clear);
        }
        for draw in &list.draws {
            let coverage = match draw.mask {
                Some(id) => {
                    let layer = *layer_of.get(&id).ok_or_else(|| {
                        Error::InvalidArgument(format!("a draw cut by {id:?}, which is not listed"))
                    })?;
                    Some(&self.masks[layer * w * h..(layer + 1) * w * h])
                }
                None => None,
            };
            let target = &mut self.target;
            let vertices = list.frame.vertices(draw.part);
            rasterize(
                list,
                &assets.parts[draw.part.index()],
                vertices,
                |x, y, texel| {
                    let m = coverage.map_or(1.0, |c| c[y * w + x] as f32 / 255.0);
                    let k = draw.alpha * m;
                    let source = texel.map(|c| c * k);
                    let pixel = &mut target[(y * w + x) * 4..(y * w + x) * 4 + 4];
                    for c in 0..4 {
                        pixel[c] = unorm8(source[c] + pixel[c] as f32 / 255.0 * (1.0 - source[3]));
                    }
                },
            );
        }
        Ok(())
    }

    fn read_rgba(&mut self, out: &mut [u8]) -> Result<(), Error> {
        if out.len() != self.target.len() {
            return Err(Error::InvalidArgument(format!(
                "{} bytes for {}",
                out.len(),
                self.target.len()
            )));
        }
        out.copy_from_slice(&self.target);
        Ok(())
    }
}

/// A value in 0 to 1 as an 8-bit UNORM target stores it: rounded to the nearest step.
fn unorm8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
}

/// A vertex in target pixels as the vertex shader and the viewport put it, in sub-pixel steps.
fn to_target(list: &DrawList, p: [f32; 2]) -> [i64; 2] {
    let canvas = [list.canvas_size[0] as f32, list.canvas_size[1] as f32];
    let c = [p[0] / canvas[0] * 2.0 - 1.0, p[1] / canvas[1] * 2.0 - 1.0];
    // Clip space y is up: the shader writes -c.y, which the viewport turns down again.
    let clip = [c[0], -c[1]];
    let v = &list.viewport;
    let x = v.x as f32 + (clip[0] + 1.0) * 0.5 * v.width as f32;
    let y = v.y as f32 + (1.0 - clip[1]) * 0.5 * v.height as f32;
    [(x * SUBPIXEL).round() as i64, (y * SUBPIXEL).round() as i64]
}

/// Twice the signed area of `a b p`, positive where `p` is on the inside of an edge of a triangle
/// wound as the rasterizer has it (clockwise on a target whose y is down).
fn edge(a: [i64; 2], b: [i64; 2], p: [i64; 2]) -> i64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Whether an edge from `a` to `b` of such a triangle is a top edge (flat, the inside under it)
/// or a left one (going up): what owns a pixel centre exactly on it.
fn top_left(a: [i64; 2], b: [i64; 2]) -> bool {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    (dy == 0 && dx > 0) || dy < 0
}

/// Every pixel of `part`'s triangles at `vertices`, with the texel it samples there.
fn rasterize(
    list: &DrawList,
    part: &PartAssets,
    vertices: &[[f32; 2]],
    mut fragment: impl FnMut(usize, usize, [f32; 4]),
) {
    let [tw, th] = [list.target[0] as i64, list.target[1] as i64];
    let v = &list.viewport;
    // The viewport clips as clip space does: centres inside it, the far edges left out.
    let clip_lo = [v.x, v.y];
    let clip_hi = [v.x + v.width, v.y + v.height];
    let size = [part.mips[0].width as f64, part.mips[0].height as f64];
    for triangle in part.indices.chunks_exact(3) {
        let mut p = [0usize, 1, 2].map(|k| to_target(list, vertices[triangle[k] as usize]));
        let mut uv = [0usize, 1, 2].map(|k| part.uv[triangle[k] as usize]);
        let mut area = edge(p[0], p[1], p[2]);
        if area == 0 {
            continue;
        }
        if area < 0 {
            p.swap(1, 2);
            uv.swap(1, 2);
            area = -area;
        }

        // The level of detail: how many texels of level 0 a pixel spans, across or down.
        let f = p.map(|q| [q[0] as f64 / SUBPIXEL as f64, q[1] as f64 / SUBPIXEL as f64]);
        let det =
            (f[1][0] - f[0][0]) * (f[2][1] - f[0][1]) - (f[2][0] - f[0][0]) * (f[1][1] - f[0][1]);
        let gradient = |axis: usize| -> [f64; 2] {
            let t = [0, 1, 2].map(|k| uv[k][axis] as f64 * size[axis]);
            let ddx =
                ((t[1] - t[0]) * (f[2][1] - f[0][1]) - (t[2] - t[0]) * (f[1][1] - f[0][1])) / det;
            let ddy =
                ((t[2] - t[0]) * (f[1][0] - f[0][0]) - (t[1] - t[0]) * (f[2][0] - f[0][0])) / det;
            [ddx, ddy]
        };
        let (du, dv) = (gradient(0), gradient(1));
        // The longest stretch of a pixel's footprint: the larger singular value of the Jacobian
        // [du; dv], from its squared norm and its determinant.
        let sum = du[0] * du[0] + du[1] * du[1] + dv[0] * dv[0] + dv[1] * dv[1];
        let det = du[0] * dv[1] - du[1] * dv[0];
        let major = ((sum + (sum * sum - 4.0 * det * det).max(0.0).sqrt()) / 2.0).sqrt();
        let lod = major.log2() as f32;

        let lo = [0, 1].map(|a| p.iter().map(|q| q[a]).min().unwrap());
        let hi = [0, 1].map(|a| p.iter().map(|q| q[a]).max().unwrap());
        // Pixels whose centre, half a pixel in, can be inside.
        let first = |l: i64| (l - STEPS / 2 + STEPS - 1).div_euclid(STEPS);
        let last = |h: i64| (h - STEPS / 2).div_euclid(STEPS);
        let x0 = first(lo[0]).max(0);
        let x1 = last(hi[0]).min(tw - 1);
        let y0 = first(lo[1]).max(0);
        let y1 = last(hi[1]).min(th - 1);
        let owns = [
            top_left(p[1], p[2]),
            top_left(p[2], p[0]),
            top_left(p[0], p[1]),
        ];
        for y in y0..=y1 {
            let cy = y as f64 + 0.5;
            if cy < clip_lo[1] || cy >= clip_hi[1] {
                continue;
            }
            for x in x0..=x1 {
                let cx = x as f64 + 0.5;
                if cx < clip_lo[0] || cx >= clip_hi[0] {
                    continue;
                }
                let centre = [x * STEPS + STEPS / 2, y * STEPS + STEPS / 2];
                let e = [
                    edge(p[1], p[2], centre),
                    edge(p[2], p[0], centre),
                    edge(p[0], p[1], centre),
                ];
                if (0..3).any(|k| e[k] < 0 || (e[k] == 0 && !owns[k])) {
                    continue;
                }
                let b = e.map(|e| e as f64 / area as f64);
                let at = [0, 1].map(|a| {
                    (b[0] * uv[0][a] as f64 + b[1] * uv[1][a] as f64 + b[2] * uv[2][a] as f64)
                        as f32
                });
                fragment(x as usize, y as usize, trilinear(part, at, lod));
            }
        }
    }
}

/// `part`'s picture at `uv` and level of detail `lod`: bilinear in the two levels about it,
/// mixed by how far between them it is, clamped to edge.
fn trilinear(part: &PartAssets, uv: [f32; 2], lod: f32) -> [f32; 4] {
    let top = (part.mips.len() - 1) as f32;
    // Magnified -- or a picture squeezed to nothing, whose LOD is -inf or NaN: level 0.
    if lod.is_nan() || lod <= 0.0 {
        return bilinear(part, 0, uv);
    }
    let lod = lod.min(top);
    let level = lod.floor() as usize;
    let t = lod - level as f32;
    let a = bilinear(part, level, uv);
    if t == 0.0 || level + 1 >= part.mips.len() {
        return a;
    }
    let b = bilinear(part, level + 1, uv);
    [0, 1, 2, 3].map(|c| a[c] + (b[c] - a[c]) * t)
}

/// Level `level` of `part`'s picture at `uv`, the four texels about it weighed by distance.
fn bilinear(part: &PartAssets, level: usize, uv: [f32; 2]) -> [f32; 4] {
    let image = &part.mips[level];
    let (w, h) = (image.width as i64, image.height as i64);
    let x = uv[0] * w as f32 - 0.5;
    let y = uv[1] * h as f32 - 0.5;
    let (fx, fy) = (x.floor(), y.floor());
    let (tx, ty) = (x - fx, y - fy);
    let (x0, y0) = (fx as i64, fy as i64);
    let texel = |x: i64, y: i64| -> [f32; 4] {
        let at = ((y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) * 4) as usize;
        [0, 1, 2, 3].map(|c| image.rgba[at + c] as f32 / 255.0)
    };
    let (a, b, c, d) = (
        texel(x0, y0),
        texel(x0 + 1, y0),
        texel(x0, y0 + 1),
        texel(x0 + 1, y0 + 1),
    );
    [0, 1, 2, 3].map(|k| {
        let top = a[k] + (b[k] - a[k]) * tx;
        let bottom = c[k] + (d[k] - c[k]) * tx;
        top + (bottom - top) * ty
    })
}
