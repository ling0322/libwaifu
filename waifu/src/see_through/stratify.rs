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

//! Depth-guided layer stratification: one part cut into a front and a back by its own depth.
//!
//! One depth a part cannot say that a part is in front of another in one place and behind it in
//! another -- the back of a collar behind the neck and its flaps in front, the paper's "sandwich".
//! The paper's answer (section 4.2.2) is to cluster each part's pixel depth into two, near and far,
//! and make each its own layer; its PSD export does this to hair, hands, tops and bottoms.
//! Upstream ships it only as a step to run by hand, `heuristic_partseg.py seg_wdepth`, which is
//! `cluster_inpaint_part` in `common/utils/torchcv.py`; this is that, with its own `cv2.inpaint`
//! branch rather than LaMa, which is a model of its own:
//!
//! - k-means (k = 2) on the depth of what is drawn, alpha above 10;
//! - the near cluster is the front, `<tag>-0`, its edge feathered by a 3 x 3 Gaussian;
//! - the back, `<tag>-1`, is the whole part with the front's place, grown by a 7 x 7 ellipse,
//!   filled in from around it -- by Lvmin's pyramid ([`pad_rgb`]) where upstream has
//!   Navier-Stokes -- and its depth there the median of the rest;
//! - the front takes the part's place in the document, and the back goes down past the layers
//!   under it nearer than its own median depth; the rest of the order is the document's.
//!
//! It runs on a document after the fact -- one img2layers wrote, which keeps each part's depth in a
//! hidden group -- as upstream's `seg_wdepth_psd` runs on a document it wrote.
//!
//! Some things differ, on purpose:
//!
//! - The k-means is exact and the same every time: depth is a byte, so it runs on a histogram of
//!   256 values from the two ends, where upstream's is seeded at random on 2000 samples.
//! - Specks -- pieces of what is drawn under 40 pixels, which a layer is scattered with all over
//!   the canvas -- are not clustered, and stay with the back. Clustered with the rest they are a
//!   cluster of their own, and the part is cut into itself and specks.
//! - A side under a twentieth of what is clustered (64 pixels at least) leaves the part whole: a
//!   hundredth was a fringe of depth noise along the edge of hair or a hand, cut off for nothing.
//! - The front keeps the part's own alpha under its feathered mask, where upstream's front is the
//!   mask alone and loses the part's soft edge.
//! - The back keeps the part's own outline: only its colour is filled in. Upstream fills its alpha
//!   in from around the front too, which where the front reaches the part's edge brings the
//!   transparency outside it in, and leaves a faint ghost of the part.

use super::compose::Part;
use super::image::{bounding_rect, connected_components, median, pad_rgb, Image, Rgba};

/// The parts the paper stratifies when it writes a document: hair, hands, tops and bottoms -- with
/// this model's two hairs, and hands split into a left and a right.
pub const STRATIFIED: [&str; 7] = [
    "front hair",
    "back hair",
    "handwear",
    "handwear-l",
    "handwear-r",
    "topwear",
    "bottomwear",
];

/// What counts as drawn, as upstream's `seg_wdepth_psd` has it.
const DRAWN: u8 = 10;
/// How much of the part, beyond the front's feathered mask, the back has to keep to be read from:
/// `(alpha - mask) > 50`.
const KEPT: i32 = 50;
/// A piece of what is drawn smaller than this, its pixels touching, is a speck and not clustered:
/// Anime2.5DRig's floor for the same specks.
const SPECK: usize = 40;
/// Neither side may be smaller than this, nor than a twentieth of what is clustered.
const SMALLEST: usize = 64;

/// A layer of a document and its depth: what a document img2layers wrote keeps, the depth in its
/// hidden group, and what is stratified after the fact.
#[derive(Clone, Debug)]
pub struct DepthLayer {
    pub name: String,
    /// Where its top left corner is on the canvas.
    pub left: i32,
    pub top: i32,
    /// RGBA, not premultiplied.
    pub pixels: Rgba,
    /// Gray, 0 near to 255 far, the same size.
    pub depth: Image<u8>,
}

/// `layers`, bottom first, with every one in [`STRATIFIED`] cut into its front and back.
///
/// The document's order is kept: what a person or the pipeline's fix-ups put where, stays. A
/// front takes its layer's place, and a back goes down from there past every layer right under it
/// that is nearer than it -- the median depth of what it holds against theirs.
pub fn stratify(layers: Vec<DepthLayer>) -> Vec<DepthLayer> {
    // Each with the median depth of what is drawn of it, which is what a back is laid against.
    let mut out: Vec<(DepthLayer, f32)> = Vec::with_capacity(layers.len() + STRATIFIED.len());
    for layer in layers {
        let (left, top) = (layer.left, layer.top);
        // At the canvas's corner while it is cut, and put back where it was after.
        let part = Part {
            tag: layer.name.clone(),
            left: 0,
            top: 0,
            pixels: layer.pixels,
            depth: layer.depth,
            depth_median: 0.0,
        };
        let drawn: Vec<bool> = (0..part.pixels.width * part.pixels.height)
            .map(|p| part.pixels.data[p * 4 + 3] > DRAWN)
            .collect();
        let median = median_depth(&part, &drawn).unwrap_or(1.0);
        let cut = STRATIFIED
            .contains(&part.tag.as_str())
            .then(|| split(&part))
            .flatten();
        let back_to = |p: Part| {
            let depth_median = p.depth_median;
            let layer = DepthLayer {
                name: p.tag,
                left: left + p.left as i32,
                top: top + p.top as i32,
                pixels: p.pixels,
                depth: p.depth,
            };
            (layer, depth_median)
        };
        match cut {
            Some((front, back)) => {
                let back = back_to(back);
                let mut at = out.len();
                while at > 0 && out[at - 1].1 < back.1 {
                    at -= 1;
                }
                out.insert(at, back);
                out.push(back_to(front));
            }
            None => out.push((
                DepthLayer {
                    name: part.tag,
                    left,
                    top,
                    pixels: part.pixels,
                    depth: part.depth,
                },
                median,
            )),
        }
    }
    out.into_iter().map(|(layer, _)| layer).collect()
}

/// The two means of `histogram`'s values, nearest first, and where they divide: everything at or
/// below the returned bound is the near cluster. None where one cluster would be empty.
fn two_means(histogram: &[u64; 256]) -> Option<(f64, f64, usize)> {
    let lowest = histogram.iter().position(|&n| n > 0)?;
    let highest = histogram.iter().rposition(|&n| n > 0)?;
    if lowest == highest {
        return None;
    }
    let (mut near, mut far) = (lowest as f64, highest as f64);
    let mut bound = 0;
    for _ in 0..256 {
        bound = ((near + far) / 2.0).floor() as usize;
        let mean = |range: std::ops::RangeInclusive<usize>| {
            let (mut sum, mut count) = (0.0, 0u64);
            for v in range {
                sum += v as f64 * histogram[v] as f64;
                count += histogram[v];
            }
            (count > 0).then(|| sum / count as f64)
        };
        let (Some(n), Some(f)) = (mean(0..=bound), mean(bound + 1..=255)) else {
            return None;
        };
        if n == near && f == far {
            break;
        }
        (near, far) = (n, f);
    }
    Some((near, far, bound))
}

/// `cv2.getStructuringElement(MORPH_ELLIPSE, (7, 7))`, row by row.
const ELLIPSE_7: [&str; 7] = [
    "0001000", "0111110", "1111111", "1111111", "1111111", "0111110", "0001000",
];

/// `cv2.dilate` by [`ELLIPSE_7`].
fn dilate_ellipse(mask: &Image<bool>) -> Image<bool> {
    let (w, h) = (mask.width as i64, mask.height as i64);
    let mut out = Image::new(mask.width, mask.height, 1);
    for y in 0..h {
        for x in 0..w {
            if !mask.data[(y * w + x) as usize] {
                continue;
            }
            for (dy, row) in ELLIPSE_7.iter().enumerate() {
                for (dx, cell) in row.bytes().enumerate() {
                    let (nx, ny) = (x + dx as i64 - 3, y + dy as i64 - 3);
                    if cell == b'1' && nx >= 0 && ny >= 0 && nx < w && ny < h {
                        out.data[(ny * w + nx) as usize] = true;
                    }
                }
            }
        }
    }
    out
}

/// `cv2.GaussianBlur(mask, (3, 3), sigma 0.7)` of a mask in 0 and 255, borders reflected.
fn feather(mask: &Image<bool>) -> Image<f32> {
    let k1 = (-1.0f32 / (2.0 * 0.7 * 0.7)).exp();
    let kernel = [
        k1 / (1.0 + 2.0 * k1),
        1.0 / (1.0 + 2.0 * k1),
        k1 / (1.0 + 2.0 * k1),
    ];
    let (w, h) = (mask.width, mask.height);
    let at = |i: i64, n: usize| -> usize {
        // BORDER_REFLECT_101: -1 is 1, n is n - 2.
        match i {
            i if i < 0 => (-i) as usize,
            i if i >= n as i64 => (2 * n as i64 - 2 - i).max(0) as usize,
            i => i as usize,
        }
    };
    let value = |x: usize, y: usize| if mask.data[y * w + x] { 255.0 } else { 0.0 };
    let mut across = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            across[y * w + x] = (0..3)
                .map(|k| kernel[k] * value(at(x as i64 + k as i64 - 1, w), y))
                .sum();
        }
    }
    let mut out = Image::new(w, h, 1);
    for y in 0..h {
        for x in 0..w {
            out.data[y * w + x] = (0..3)
                .map(|k| kernel[k] * across[at(y as i64 + k as i64 - 1, h) * w + x])
                .sum();
        }
    }
    out
}

/// `values` where `known`, and elsewhere filled in from around them by [`pad_rgb`]: up to three
/// channels a pixel, in 0 to 255.
fn fill(values: &[[f32; 3]], known: &[bool], width: usize, height: usize) -> Vec<[f32; 3]> {
    let mut rgba = Image::new(width, height, 4);
    for p in 0..width * height {
        if known[p] {
            for c in 0..3 {
                rgba.data[p * 4 + c] = values[p][c] / 255.0;
            }
            rgba.data[p * 4 + 3] = 1.0;
        }
    }
    let padded = pad_rgb(&rgba);
    (0..width * height)
        .map(|p| {
            if known[p] {
                values[p]
            } else {
                [0, 1, 2].map(|c| (padded.data[p * 3 + c] * 255.0).clamp(0.0, 255.0))
            }
        })
        .collect()
}

/// The median of `part`'s depth where `mask` is, in `[0, 1]`.
fn median_depth(part: &Part, mask: &[bool]) -> Option<f32> {
    let mut values: Vec<f32> = (0..mask.len())
        .filter(|&p| mask[p])
        .map(|p| part.depth.data[p] as f32 / 255.0)
        .collect();
    median(&mut values)
}

/// `part` cut into its near and far clusters of depth: the front, cut down to where it is drawn,
/// and the back, the whole part with the front's place filled in. None where its depth does not
/// divide, or one side would be empty.
pub fn split(part: &Part) -> Option<(Part, Part)> {
    let (w, h) = (part.pixels.width, part.pixels.height);
    let n = w * h;
    let alpha = |p: usize| part.pixels.data[p * 4 + 3];
    // What is drawn, without the specks a layer is scattered with all over the canvas: those are
    // far from the part and of another depth, and would be a cluster of their own -- a "far" half
    // of nothing but specks. They are left out of the clustering, and stay with the back.
    let drawn = Image::from_data(w, h, 1, (0..n).map(|p| alpha(p) > DRAWN).collect());
    let (labels, pieces) = connected_components(&drawn);
    let clustered: Vec<bool> = (0..n)
        .map(|p| drawn.data[p] && pieces[labels[p] as usize].area >= SPECK)
        .collect();

    let mut histogram = [0u64; 256];
    for p in (0..n).filter(|&p| clustered[p]) {
        histogram[part.depth.data[p] as usize] += 1;
    }
    let (_, _, bound) = two_means(&histogram)?;
    let front: Vec<bool> = (0..n)
        .map(|p| clustered[p] && (part.depth.data[p] as usize) <= bound)
        .collect();
    let back: Vec<bool> = (0..n).map(|p| clustered[p] && !front[p]).collect();
    // A side of next to nothing is not a stratum: the part is left whole.
    let least = (clustered.iter().filter(|&&c| c).count() / 20).max(SMALLEST);
    let count = |mask: &[bool]| mask.iter().filter(|&&m| m).count();
    if count(&front) < least || count(&back) < least {
        return None;
    }

    let front_mask = Image::from_data(w, h, 1, front.clone());
    let grown = dilate_ellipse(&front_mask);
    let soft = feather(&front_mask);

    // The front: the part's colour, its alpha under the feathered mask.
    let mut front_pixels = part.pixels.clone();
    for p in 0..n {
        let a = alpha(p) as f32 * soft.data[p] / 255.0;
        front_pixels.data[p * 4 + 3] = a.round().clamp(0.0, 255.0) as u8;
    }
    let front_part = cut(
        Part {
            tag: format!("{}-0", part.tag),
            left: part.left,
            top: part.top,
            pixels: front_pixels,
            depth: part.depth.clone(),
            depth_median: median_depth(part, &front)?,
        },
        DRAWN,
    )?;

    // The back: the whole part, its outline the part's own -- what the model drew of it, hidden
    // or not -- its colour filled in over the grown front from what is still its own, and its
    // depth there the median of what it keeps.
    let kept: Vec<bool> = (0..n)
        .map(|p| alpha(p) as i32 - soft.data[p].round() as i32 > KEPT)
        .collect();
    let colours: Vec<[f32; 3]> = (0..n)
        .map(|p| [0, 1, 2].map(|c| part.pixels.data[p * 4 + c] as f32))
        .collect();
    let colour_known: Vec<bool> = (0..n).map(|p| !grown.data[p] && kept[p]).collect();
    let colours = fill(&colours, &colour_known, w, h);

    let mut back_pixels = part.pixels.clone();
    let mut back_depth = part.depth.clone();
    let rest = median_depth(part, &kept).unwrap_or(part.depth_median);
    for p in 0..n {
        if grown.data[p] {
            for c in 0..3 {
                back_pixels.data[p * 4 + c] = colours[p][c].round() as u8;
            }
        }
        if soft.data[p] > 127.0 {
            back_depth.data[p] = (rest * 255.0).round() as u8;
        }
    }
    let back_part = Part {
        tag: format!("{}-1", part.tag),
        left: part.left,
        top: part.top,
        pixels: back_pixels,
        depth: back_depth,
        depth_median: median_depth(part, &back)?,
    };
    Some((front_part, back_part))
}

/// `part` cut down to where its alpha is above `floor`; None for nothing.
fn cut(part: Part, floor: u8) -> Option<Part> {
    let (w, h) = (part.pixels.width, part.pixels.height);
    let mask = Image::from_data(
        w,
        h,
        1,
        (0..w * h)
            .map(|p| part.pixels.data[p * 4 + 3] > floor)
            .collect(),
    );
    let (x, y, bw, bh) = bounding_rect(&mask)?;
    Some(Part {
        left: part.left + x,
        top: part.top + y,
        pixels: part.pixels.crop(x, y, x + bw, y + bh),
        depth: part.depth.crop(x, y, x + bw, y + bh),
        ..part
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A part 20 by 10 whose left half is at depth 40 and right half at 200, all of it drawn.
    fn two_steps() -> Part {
        let (w, h) = (20, 10);
        let mut pixels = Image::new(w, h, 4);
        let mut depth = Image::new(w, h, 1);
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                pixels.data[p * 4..p * 4 + 4].copy_from_slice(&[100, 150, 200, 255]);
                depth.data[p] = if x < 10 { 40 } else { 200 };
            }
        }
        Part {
            tag: "topwear".into(),
            left: 5,
            top: 7,
            pixels,
            depth,
            depth_median: 0.5,
        }
    }

    #[test]
    fn two_means_divide_between_the_clusters() {
        let mut histogram = [0u64; 256];
        histogram[10] = 5;
        histogram[12] = 5;
        histogram[200] = 3;
        let (near, far, bound) = two_means(&histogram).unwrap();
        assert_eq!(near, 11.0);
        assert_eq!(far, 200.0);
        assert!((12..200).contains(&bound), "{bound}");
        let mut one = [0u64; 256];
        one[7] = 9;
        assert!(two_means(&one).is_none());
    }

    #[test]
    fn a_part_is_cut_into_its_near_front_and_its_whole_back() {
        let part = two_steps();
        let (front, back) = split(&part).unwrap();
        assert_eq!(front.tag, "topwear-0");
        assert_eq!(back.tag, "topwear-1");
        // The front is the near half, cut down to it, a feathered pixel past it at most.
        assert_eq!((front.left, front.top), (5, 7));
        assert!(front.pixels.width <= 11, "{}", front.pixels.width);
        assert!((front.depth_median - 40.0 / 255.0).abs() < 1e-6);
        // The back is the whole part, filled in where the front was, at the far median.
        assert_eq!((back.pixels.width, back.pixels.height), (20, 10));
        assert!((back.depth_median - 200.0 / 255.0).abs() < 1e-6);
        assert!(
            back.pixels.data[3] > 200,
            "the front's place is filled, not cut out"
        );
        // Its depth there is the median of what it keeps: the far half.
        assert_eq!(back.depth.data[0], 200);
    }

    #[test]
    fn specks_far_off_are_not_a_far_half() {
        // One depth where the part is drawn, and specks all over the canvas at another: what a
        // model's layer looks like. Without leaving them out the part is cut into itself and specks.
        let mut part = two_steps();
        part.depth.data.iter_mut().for_each(|d| *d = 60);
        let (w, h) = (60, 40);
        let mut pixels = Image::new(w, h, 4);
        let mut depth = Image::new(w, h, 1);
        for y in 0..10 {
            for x in 0..20 {
                let p = y * w + x;
                pixels.data[p * 4..p * 4 + 4].copy_from_slice(&[100, 150, 200, 255]);
                depth.data[p] = 60;
            }
        }
        for (x, y) in [(50, 30), (55, 5), (40, 35), (30, 20), (58, 38)] {
            let p = y * w + x;
            pixels.data[p * 4 + 3] = 40;
            depth.data[p] = 240;
        }
        part.pixels = pixels;
        part.depth = depth;
        assert!(split(&part).is_none());
    }

    #[test]
    fn a_flat_part_is_left_whole_and_others_are_never_cut() {
        let mut flat = two_steps();
        flat.depth.data.iter_mut().for_each(|d| *d = 90);
        assert!(split(&flat).is_none());

        // The face at depth 120, between the top's halves, and the top over it in the document: the
        // top's front stays where the top was, and its back goes under the face.
        let as_layer = |part: Part| DepthLayer {
            name: part.tag,
            left: part.left as i32,
            top: part.top as i32,
            pixels: part.pixels,
            depth: part.depth,
        };
        let mut face = two_steps();
        face.tag = "face".into();
        face.depth.data.iter_mut().for_each(|d| *d = 120);
        let back_hair = {
            let mut hair = two_steps();
            hair.tag = "back hair".into();
            hair.depth.data.iter_mut().for_each(|d| *d = 250);
            hair
        };
        let layers = stratify(vec![
            as_layer(back_hair),
            as_layer(face),
            as_layer(two_steps()),
        ]);
        let tags: Vec<&str> = layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(tags, ["back hair", "topwear-1", "face", "topwear-0"]);
        // Where it was on the canvas, both halves.
        assert_eq!((layers[3].left, layers[3].top), (5, 7));
        assert_eq!((layers[1].left, layers[1].top), (5, 7));
    }
}
