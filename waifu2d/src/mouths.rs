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

//! A mouth that moves through the shapes libwaifu's text to speech says -- `A E I O U @ MBP REST`
//! -- for a rig whose document has only its one mouth, sized to the drawn mouth and in its
//! colours. **An experiment**: what it looks like decides whether it stays.
//!
//! The drawn mouth is kept, and is the rest shape. Over it is a mouth made of four parts, each a
//! grid that never changes but whose vertices are worked out again every frame: the inside, a
//! superellipse hung from the mouth's line -- flat on top and round below, as an anime mouth opens
//! by its jaw -- the line around it, the upper teeth and the tongue, the last two cut by the inside
//! as an iris is by its white. Every shape is a set of numbers -- how wide, how open, how square,
//! how much tooth and tongue -- and the mouth is drawn from those of the shapes it is between,
//! mixed by how far it is from each: it opens, rounds and shuts rather than cutting from one
//! picture to the next. At rest it is the drawn line, width, curve and weight, and fades into it.

use crate::geom::{Rect, RgbaImage};
use crate::rig::{MaskDef, MaskId, PartId, Pixels, Rig, RigMesh, RigPart, Role};

/// A mouth shape: libwaifu's `WaifuViseme`, number for number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Viseme {
    A = 0,
    E = 1,
    I = 2,
    O = 3,
    U = 4,
    /// Half open and not spread: Mandarin's e, English's unstressed vowel.
    Schwa = 5,
    /// The lips pressed shut: b, p, m.
    Mbp = 6,
    /// Relaxed and lightly closed: the drawn mouth.
    Rest = 7,
}

impl Viseme {
    pub const ALL: [Viseme; 8] = [
        Viseme::A,
        Viseme::E,
        Viseme::I,
        Viseme::O,
        Viseme::U,
        Viseme::Schwa,
        Viseme::Mbp,
        Viseme::Rest,
    ];

    pub fn from_index(i: i32) -> Option<Viseme> {
        Viseme::ALL.get(usize::try_from(i).ok()?).copied()
    }

    /// The weights of this shape alone, by `Viseme as usize`.
    pub fn alone(self) -> [f32; 8] {
        let mut w = [0.0; 8];
        w[self as usize] = 1.0;
        w
    }
}

/// A shape as numbers, in widths of the drawn mouth. The mouth between shapes is drawn from these
/// mixed.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shape {
    /// The opening's width and height; a height of 0 is shut, a line.
    width: f64,
    height: f64,
    /// How square the opening's top and bottom are: 2 an ellipse, more flatter.
    top: f64,
    bottom: f64,
    /// How much of the opening's height the upper teeth fill, from its top.
    teeth: f64,
    /// How wide the tongue is, as much of the opening's width.
    tongue: f64,
    /// How thick the upper lip's line is, in the drawn line's thickness; the lower is 0.6 of it,
    /// so that shut the two make 1.6 of this.
    line: f64,
    /// How much of the opening is above the drawn line: the upper lip lifting as the jaw drops.
    /// An anime mouth opens mostly downward, but not all of it: a quarter to a third of a wide
    /// open one is above where the line was, and a thin or small one sits about it.
    lift: f64,
}

fn shape(v: Viseme) -> Shape {
    let s = |width, height, top, bottom, teeth, tongue, line, lift| Shape {
        width,
        height,
        top,
        bottom,
        teeth,
        tongue,
        line,
        lift,
    };
    match v {
        Viseme::A => s(0.62, 0.46, 3.5, 2.2, 0.18, 0.62, 1.15, 0.30),
        Viseme::E => s(0.70, 0.26, 3.0, 2.2, 0.30, 0.00, 1.15, 0.30),
        Viseme::I => s(0.78, 0.15, 3.2, 2.4, 0.55, 0.00, 1.15, 0.40),
        Viseme::O => s(0.44, 0.40, 2.3, 2.1, 0.12, 0.55, 1.15, 0.30),
        Viseme::U => s(0.30, 0.22, 2.2, 2.0, 0.00, 0.00, 1.15, 0.45),
        Viseme::Schwa => s(0.50, 0.22, 3.0, 2.2, 0.15, 0.50, 1.15, 0.30),
        // Pressed: shorter than at rest, and heavier.
        Viseme::Mbp => s(0.58, 0.0, 2.0, 2.0, 0.0, 0.0, 1.1, 0.0),
        // The drawn line: as wide, and shut as thick as it, 1.6 of 0.625.
        Viseme::Rest => s(1.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.625, 0.0),
    }
}

/// The shapes mixed by `weights`, by `Viseme as usize`.
fn blend(weights: &[f32; 8]) -> Shape {
    let total: f64 = weights.iter().map(|&w| w.max(0.0) as f64).sum();
    if total <= 0.0 {
        return shape(Viseme::Rest);
    }
    let mut m = Shape {
        width: 0.0,
        height: 0.0,
        top: 0.0,
        bottom: 0.0,
        teeth: 0.0,
        tongue: 0.0,
        line: 0.0,
        lift: 0.0,
    };
    for v in Viseme::ALL {
        let w = weights[v as usize].max(0.0) as f64 / total;
        let s = shape(v);
        m.width += w * s.width;
        m.height += w * s.height;
        m.top += w * s.top;
        m.bottom += w * s.bottom;
        m.teeth += w * s.teeth;
        m.tongue += w * s.tongue;
        m.line += w * s.line;
        m.lift += w * s.lift;
    }
    m
}

/// The drawn mouth's colours: its line, and the inside of an open mouth made from it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Colours {
    line: [f64; 3],
    inside: [f64; 3],
    teeth: [f64; 3],
    tongue: [f64; 3],
}

/// The darker two fifths of what is drawn of `image`, as its line's colour.
fn line_colour(image: &RgbaImage) -> [f64; 3] {
    let mut drawn: Vec<[f64; 3]> = image
        .rgba
        .chunks_exact(4)
        .filter(|p| p[3] > 128)
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    if drawn.is_empty() {
        return [90.0, 40.0, 45.0];
    }
    drawn.sort_by(|a, b| (a[0] + a[1] + a[2]).total_cmp(&(b[0] + b[1] + b[2])));
    let dark = &drawn[..(drawn.len() * 2 / 5).max(1)];
    let n = dark.len() as f64;
    [0, 1, 2].map(|c| dark.iter().map(|p| p[c]).sum::<f64>() / n)
}

fn colours(mouth: &RgbaImage) -> Colours {
    let line = line_colour(mouth);
    let mix = |a: [f64; 3], b: [f64; 3], t: f64| [0, 1, 2].map(|c| a[c] * (1.0 - t) + b[c] * t);
    Colours {
        line,
        inside: mix(line, [150.0, 45.0, 58.0], 0.6),
        teeth: [250.0, 248.0, 244.0],
        tongue: mix([222.0, 108.0, 118.0], line, 0.15),
    }
}

/// Where the drawn mouth's line is: its middle, the height of the line across it, and its width;
/// how thick it is drawn, and how far its ends rise above its middle -- the smile in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    x: f64,
    y: f64,
    width: f64,
    stroke: f64,
    bend: f64,
}

impl Line {
    /// The line of the mouth drawn in `image`, whose top left corner is at (`left`, `top`):
    /// column by column, where its ink is and how much of it there is, and the curve
    /// `y = middle - bend * u^2` fitted to the middles, `u` -1 to 1 across it.
    fn measure(image: &RgbaImage, left: f64, top: f64) -> Option<Line> {
        let bounds = Pixels::bounds(image, 16)?;
        let mut columns = Vec::new();
        for x in 0..image.width {
            let (mut ink, mut moment) = (0.0, 0.0);
            for y in 0..image.height {
                let a = image.rgba[(y * image.width + x) * 4 + 3] as f64 / 255.0;
                ink += a;
                moment += a * (y as f64 + 0.5);
            }
            if ink > 0.3 {
                columns.push((x as f64 + 0.5, moment / ink, ink));
            }
        }
        if columns.len() < 3 {
            return None;
        }
        let (mid_x, half) = (bounds.mid_x(), (bounds.width / 2.0).max(1.0));
        // Least squares of y = c - bend * u^2 over the columns.
        let n = columns.len() as f64;
        let (mut su, mut sy, mut suu, mut suy) = (0.0, 0.0, 0.0, 0.0);
        for &(x, y, _) in &columns {
            let u2 = ((x - mid_x) / half).powi(2);
            su += u2;
            sy += y;
            suu += u2 * u2;
            suy += u2 * y;
        }
        let slope = (n * suy - su * sy) / (n * suu - su * su).max(1e-9);
        let middle = (sy - slope * su) / n;
        let mut inks: Vec<f64> = columns.iter().map(|c| c.2).collect();
        inks.sort_by(f64::total_cmp);
        Some(Line {
            x: left + mid_x,
            y: top + middle,
            width: bounds.width.max(8.0),
            stroke: inks[inks.len() / 2].max(0.8),
            bend: -slope,
        })
    }

    /// How far above its middle the line is at `dx` across from it.
    fn rise(&self, dx: f64) -> f64 {
        let u = (dx / (self.width / 2.0)).clamp(-1.0, 1.0);
        self.bend * u * u
    }
}

/// How many points go round the opening, and round the tongue; how many columns the teeth are.
const ROUND: usize = 64;
const TONGUE_ROUND: usize = 32;
const TEETH_COLUMNS: usize = 24;

/// A rig's made mouth: the line it is drawn about, the drawn mouth it rests as, and its four
/// parts.
#[derive(Clone, Debug, PartialEq)]
pub struct MouthRig {
    line: Line,
    pub drawn: PartId,
    pub inside: PartId,
    pub teeth: PartId,
    pub tongue: PartId,
    pub outline: PartId,
}

/// The mouth `s` is, laid about `line`: where its opening's top is and how far it is across and
/// down, and its line's thickness above and below.
struct Laid {
    a: f64,
    b: f64,
    mid_y: f64,
    upper: f64,
    lower: f64,
}

fn lay(line: &Line, s: &Shape) -> Laid {
    let w = line.width;
    let (a, b) = (s.width * w / 2.0, s.height * w / 2.0);
    // The opening is `lift` of it above the line and the rest below, the jaw's share; shut, it is
    // the line.
    let top = line.y - s.lift * 2.0 * b;
    let upper = (line.stroke * s.line).min(0.05 * w);
    Laid {
        a,
        b,
        mid_y: top + b,
        upper,
        lower: 0.6 * upper,
    }
}

/// The point at `theta` round a superellipse `a` across and `b` down from (0, 0), as square on top
/// and below as `s` says; y down, so the top is where `sin(theta)` is negative.
fn round(a: f64, b: f64, s: &Shape, theta: f64) -> (f64, f64) {
    let (c, n) = (theta.cos(), theta.sin());
    let p = if n < 0.0 { s.top } else { s.bottom };
    let f = |t: f64| t.signum() * t.abs().powf(2.0 / p);
    (a * f(c), b * f(n))
}

/// A point of the mouth on the canvas: `(dx, dy)` from the opening's middle, bent with the line.
fn place(line: &Line, laid: &Laid, dx: f64, dy: f64) -> [f32; 2] {
    [
        (line.x + dx) as f32,
        (laid.mid_y + dy - line.rise(dx)) as f32,
    ]
}

/// Part `part`'s vertices, at rest on the canvas, with the mouth's shapes weighed as `weights`
/// says; `None` for a part that is not one of the made mouth's.
pub fn shaped(rig: &Rig, part: PartId, weights: &[f32; 8]) -> Option<Vec<[f32; 2]>> {
    let m = rig.mouth.as_ref()?;
    let s = blend(weights);
    let line = &m.line;
    let laid = lay(line, &s);
    let theta = |i: usize, n: usize| i as f64 / n as f64 * std::f64::consts::TAU;
    if part == m.inside {
        // The middle, then round the opening.
        let mut out = vec![place(line, &laid, 0.0, 0.0)];
        for i in 0..ROUND {
            let (x, y) = round(laid.a, laid.b, &s, theta(i, ROUND));
            out.push(place(line, &laid, x, y));
        }
        Some(out)
    } else if part == m.outline {
        // Round the opening, and round it grown by the line: thicker above than below.
        let mut out = Vec::with_capacity(2 * ROUND);
        for i in 0..ROUND {
            let t = theta(i, ROUND);
            let (x, y) = round(laid.a, laid.b, &s, t);
            out.push(place(line, &laid, x, y));
            let g = laid.lower + (laid.upper - laid.lower) * (1.0 - t.sin()) / 2.0;
            let (x, y) = round(laid.a + g, laid.b + g, &s, t);
            out.push(place(line, &laid, x, y));
        }
        Some(out)
    } else if part == m.teeth {
        // A band from over the opening's top down as far as the teeth reach, wider than it: the
        // inside cuts it.
        let reach = laid.a + laid.upper + 2.0;
        let mut out = Vec::with_capacity(2 * (TEETH_COLUMNS + 1));
        for i in 0..=TEETH_COLUMNS {
            let dx = -reach + 2.0 * reach * i as f64 / TEETH_COLUMNS as f64;
            out.push(place(line, &laid, dx, -laid.b - laid.upper - 2.0));
            out.push(place(line, &laid, dx, -laid.b + s.teeth * 2.0 * laid.b));
        }
        Some(out)
    } else if part == m.tongue {
        // An ellipse about the opening's bottom, which the inside cuts to its top half; as it
        // goes, it sinks as well as narrows, rather than leaving a nub.
        let (rx, ry) = (s.tongue * laid.a, 0.55 * laid.b * (s.tongue / 0.6).min(1.0));
        let mut out = vec![place(line, &laid, 0.0, laid.b)];
        for i in 0..TONGUE_ROUND {
            let t = theta(i, TONGUE_ROUND);
            out.push(place(line, &laid, rx * t.cos(), laid.b + ry * t.sin()));
        }
        Some(out)
    } else {
        None
    }
}

/// A fan's triangles: its middle, vertex 0, and each pair of the `n` round it.
fn fan(n: usize) -> Vec<u32> {
    (0..n as u32)
        .flat_map(|i| [0, 1 + i, 1 + (i + 1) % n as u32])
        .collect()
}

/// A ring's triangles, inner and outer vertices taking turns, `n` of each.
fn ring(n: usize) -> Vec<u32> {
    (0..n as u32)
        .flat_map(|i| {
            let (a, b) = (2 * i, 2 * i + 1);
            let (c, d) = (2 * ((i + 1) % n as u32), 2 * ((i + 1) % n as u32) + 1);
            [a, b, c, b, d, c]
        })
        .collect()
}

/// A strip's triangles, top and bottom vertices taking turns, `columns` quads of them.
fn strip(columns: usize) -> Vec<u32> {
    (0..columns as u32)
        .flat_map(|i| {
            let (a, b, c, d) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
            [a, b, c, b, d, c]
        })
        .collect()
}

/// A disc of `colour`, `size` across, its edge softened over a texel and a half: what a fan's
/// rim reads, so that the shape's edge is smooth with no help from the GPU.
fn disc(size: usize, colour: [f64; 3]) -> RgbaImage {
    let mut rgba = Vec::with_capacity(size * size * 4);
    for y in 0..size {
        for x in 0..size {
            let (u, v) = (
                (x as f64 + 0.5) / size as f64 - 0.5,
                (y as f64 + 0.5) / size as f64 - 0.5,
            );
            let edge = (0.5 - (u * u + v * v).sqrt()) * size as f64 / 1.5;
            let a = (edge.clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba.extend([colour[0] as u8, colour[1] as u8, colour[2] as u8, a]);
        }
    }
    RgbaImage::new(size, size, rgba)
}

/// A line's cross-section: solid from its inside edge, softened toward its outside one.
fn profile(colour: [f64; 3]) -> RgbaImage {
    let alphas = [255u8, 255, 200, 70];
    let mut rgba = Vec::new();
    for _ in 0..2 {
        for a in alphas {
            rgba.extend([colour[0] as u8, colour[1] as u8, colour[2] as u8, a]);
        }
    }
    RgbaImage::new(alphas.len(), 2, rgba)
}

fn solid(colour: [f64; 3]) -> RgbaImage {
    RgbaImage::new(
        2,
        2,
        [colour[0] as u8, colour[1] as u8, colour[2] as u8, 255].repeat(4),
    )
}

/// The rig's drawn mouth: the last layer in a mouth slot.
pub fn drawn_mouth(rig: &Rig) -> Option<PartId> {
    rig.parts
        .iter()
        .rposition(|p| matches!(p.slot.as_str(), "mouth_open" | "mouth_close" | "mouth"))
        .map(|i| PartId(i as u32))
}

/// `rig` with a made mouth over its drawn one, at rest until it is told a shape. As it is where it
/// has no drawn mouth to size one by.
pub fn with_drawn_mouths(mut rig: Rig) -> Rig {
    if rig.mouth.is_some() {
        return rig;
    }
    let Some(drawn_id) = drawn_mouth(&rig) else {
        return rig;
    };
    let drawn = rig.part(drawn_id).clone();
    let Some(line) = Line::measure(&drawn.pixels, drawn.rect.x, drawn.rect.y) else {
        return rig;
    };
    let colours = colours(&drawn.pixels);
    let at = drawn_id.index() + 1;
    let ids = |k: u32| PartId(at as u32 + k);
    let (inside, teeth, tongue, outline) = (ids(0), ids(1), ids(2), ids(3));
    let mouth = MouthRig {
        line,
        drawn: drawn_id,
        inside,
        teeth,
        tongue,
        outline,
    };

    // Each part's grid, as it is at rest: the topology is what never changes.
    let rest = Viseme::Rest.alone();
    let mut probe = rig.clone();
    probe.mouth = Some(mouth.clone());
    let mut made = Vec::new();
    let mut part =
        |name: &str, id: PartId, pixels: RgbaImage, uv: Vec<[f32; 2]>, indices: Vec<u32>| {
            let rest_points = shaped(&probe, id, &rest).expect("one of the mouth's parts");
            let (lo, hi) =
                rest_points
                    .iter()
                    .fold(([f32::MAX; 2], [f32::MIN; 2]), |(lo, hi), p| {
                        (
                            [lo[0].min(p[0]), lo[1].min(p[1])],
                            [hi[0].max(p[0]), hi[1].max(p[1])],
                        )
                    });
            let rect = Rect::new(
                lo[0] as f64,
                lo[1] as f64,
                (hi[0] - lo[0]) as f64,
                (hi[1] - lo[1]) as f64,
            );
            made.push(RigPart {
                name: format!("{name} (made)"),
                slot: name.to_string(),
                role: Role::Other,
                side: None,
                rect,
                pixels,
                opacity: 1.0,
                depth: drawn.depth,
                group: drawn.group,
                mesh: RigMesh {
                    rest: rest_points,
                    uv,
                    indices,
                },
                swing: None,
            });
        };
    let round_uv = |n: usize| {
        let mut uv = vec![[0.5, 0.5]];
        for i in 0..n {
            let t = i as f64 / n as f64 * std::f64::consts::TAU;
            uv.push([(0.5 + 0.5 * t.cos()) as f32, (0.5 + 0.5 * t.sin()) as f32]);
        }
        uv
    };
    part(
        "mouth_inside",
        inside,
        disc(64, colours.inside),
        round_uv(ROUND),
        fan(ROUND),
    );
    let teeth_uv = (0..=TEETH_COLUMNS)
        .flat_map(|_| [[0.5, 0.5], [0.5, 0.5]])
        .collect();
    part(
        "mouth_teeth",
        teeth,
        solid(colours.teeth),
        teeth_uv,
        strip(TEETH_COLUMNS),
    );
    part(
        "mouth_tongue",
        tongue,
        disc(32, colours.tongue),
        round_uv(TONGUE_ROUND),
        fan(TONGUE_ROUND),
    );
    let ring_uv = (0..ROUND).flat_map(|_| [[0.0, 0.5], [1.0, 0.5]]).collect();
    part(
        "mouth_line",
        outline,
        profile(colours.line),
        ring_uv,
        ring(ROUND),
    );

    let n = made.len() as u32;
    rig.parts.splice(at..at, made);
    // The masks name parts by place, and every place after the drawn mouth has moved on.
    let moved = |p: &mut PartId| {
        if p.index() >= at {
            p.0 += n;
        }
    };
    for MaskDef { parts, clips, .. } in &mut rig.masks {
        parts.iter_mut().for_each(moved);
        clips.iter_mut().for_each(moved);
    }
    // The teeth and the tongue inside the mouth, as an iris is inside its white.
    let id = rig.masks.iter().map(|m| m.id.0 + 1).max().unwrap_or(0);
    rig.masks.push(MaskDef {
        id: MaskId(id),
        name: "mouth".to_string(),
        parts: vec![inside],
        clips: vec![teeth, tongue],
    });
    rig.mouth = Some(mouth);
    rig.id = crate::rig::next_rig_id();
    rig
}

/// Each part's alpha with the mouth's shapes weighed as `weights` says: the made mouth whole as
/// soon as it is a sixth of the way from rest -- still all but shut, and where the drawn line is
/// -- and the drawn mouth only in the last sixth before rest, so that the one fades into the other
/// where they are alike and neither is ever a faint wide smear. Nothing where the rig has no made
/// mouth.
pub fn weigh(rig: &Rig, alphas: &mut [f32], weights: &[f32; 8]) {
    let Some(m) = &rig.mouth else {
        return;
    };
    let rest = weights[Viseme::Rest as usize].clamp(0.0, 1.0);
    let made = (6.0 * (1.0 - rest)).min(1.0);
    let drawn = (6.0 * rest - 5.0).clamp(0.0, 1.0);
    alphas[m.drawn.index()] *= drawn;
    for p in [m.inside, m.teeth, m.tongue, m.outline] {
        alphas[p.index()] *= made;
    }
}
