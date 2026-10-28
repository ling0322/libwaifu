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

//! From whole-canvas layers and their depth to the parts a document is made of.
//!
//! Upstream's `further_extr` and `save_part`, written out. Each layer is cut down to where it is
//! drawn, the pairs that come in twos -- hands, eyes, ears -- may be split into a left and a right,
//! the nose and mouth take their colour from the picture itself, and every part gets one depth,
//! the median of its own. That depth is the drawing order: the farthest part is the bottom layer.
//!
//! Two things in upstream are left out because they never run for the model this reads: the
//! `eyes` and `hair` branches of `further_extr` look for parts by names the v3 model does not
//! draw, and a minimum part size whose box is cut away again before anything reads it.

use super::image::{
    bounding_rect, connected_components, dilate_cross, mask_above, median, Image, Rgba,
};

/// What an alpha has to be above for a pixel to belong to a part.
const PART_ALPHA: u8 = 10;
/// The alpha below which a pixel of a drawn layer is taken for noise and cleared.
pub const LAYER_ALPHA: u8 = 15;

/// The parts that may be split into a left and a right one.
pub const SPLITS: [&str; 6] = [
    "handwear", "eyewhite", "irides", "eyelash", "eyebrow", "ears",
];

/// One part of the illustration, cut down to where it is drawn.
#[derive(Clone, Debug)]
pub struct Part {
    pub tag: String,
    /// Where its top left corner is on the canvas.
    pub left: usize,
    pub top: usize,
    /// RGBA, not premultiplied, only as large as what is drawn.
    pub pixels: Rgba,
    /// Its depth, the same size, 0 nearest and 255 farthest.
    pub depth: Image<u8>,
    /// The median depth of what is drawn, in `[0, 1]`: the order parts are laid in.
    pub depth_median: f32,
}

/// A part on its way through: its crop, a float depth, and where on the canvas the crop is.
struct Piece {
    tag: String,
    left: usize,
    top: usize,
    pixels: Rgba,
    depth: Image<f32>,
}

/// A layer and its depth as the two models leave them, both the whole canvas.
pub struct Drawn<'a> {
    pub tag: &'a str,
    pub pixels: &'a Rgba,
    pub depth: &'a Image<u8>,
}

/// `load_part`: the layer cut to its bounding box, or None if too little of it is drawn.
///
/// Too little is four pixels or fewer, counted outside the bottom and right tenth of the canvas --
/// upstream's test, which reads `mask[:-p, :-p]`.
fn load(drawn: &Drawn) -> Option<Piece> {
    let mask = mask_above(drawn.pixels, 3, PART_ALPHA);
    let (w, h) = (mask.width, mask.height);
    let p = w.max(h) / 10;
    let mut count = 0;
    for y in 0..h.saturating_sub(p) {
        for x in 0..w.saturating_sub(p) {
            count += mask.data[y * w + x] as usize;
        }
    }
    if count <= 4 {
        return None;
    }
    let (x, y, bw, bh) = bounding_rect(&mask)?;
    let depth = drawn.depth.crop(x, y, x + bw, y + bh);
    Some(Piece {
        tag: drawn.tag.to_string(),
        left: x,
        top: y,
        pixels: drawn.pixels.crop(x, y, x + bw, y + bh),
        depth: Image::from_data(
            bw,
            bh,
            1,
            depth.data.iter().map(|&d| d as f32 / 255.0).collect(),
        ),
    })
}

/// `part_lr_split`: the two largest pieces of a part apart, the one further left in the picture
/// named `-r` -- the character's right -- and the other `-l`. A part in one piece is left whole.
///
/// Only the drawn pieces compete: label 0 is the background. Upstream sorts every label and drops
/// the largest, meaning the background, which in a tight crop holding two large pieces is not the
/// largest -- and then a real piece is dropped and the background split off in its place.
fn split(piece: Piece) -> Vec<Piece> {
    let mask = mask_above(&piece.pixels, 3, PART_ALPHA);
    let (labels, stats) = connected_components(&mask);
    if stats.len() <= 2 {
        return vec![piece];
    }
    let mut order: Vec<usize> = (1..stats.len()).collect();
    order.sort_by(|&a, &b| stats[b].area.cmp(&stats[a].area));
    let (a, b) = (order[0], order[1]);
    let centre = |i: usize| stats[i].x as f64 + stats[i].width as f64 / 2.0;
    let (right, left) = match centre(b) < centre(a) {
        true => (b, a),
        false => (a, b),
    };

    let cut = |label: usize, suffix: &str| -> Piece {
        let s = stats[label];
        let (x1, y1) = (s.x + s.width, s.y + s.height);
        let mut pixels = piece.pixels.crop(s.x, s.y, x1, y1);
        let mut depth = piece.depth.crop(s.x, s.y, x1, y1);
        let mut own = Image::new(s.width, s.height, 1);
        for y in 0..s.height {
            for x in 0..s.width {
                own.data[y * s.width + x] =
                    labels[(s.y + y) * mask.width + s.x + x] == label as u32;
            }
        }
        let own = dilate_cross(&own);
        for p in 0..s.width * s.height {
            if !own.data[p] {
                pixels.data[p * 4 + 3] = 0;
                depth.data[p] = 1.0;
            }
        }
        Piece {
            tag: format!("{}-{suffix}", piece.tag),
            left: piece.left + s.x,
            top: piece.top + s.y,
            pixels,
            depth,
        }
    };
    vec![cut(right, "r"), cut(left, "l")]
}

/// `save_part`: cut down again to where the alpha is above the threshold, with the median depth
/// of what is left. None where nothing is.
fn finish(piece: Piece) -> Option<Part> {
    let mask = mask_above(&piece.pixels, 3, PART_ALPHA);
    let mut drawn: Vec<f32> = piece
        .depth
        .data
        .iter()
        .zip(&mask.data)
        .filter(|(_, &m)| m)
        .map(|(&d, _)| d)
        .collect();
    let depth_median = median(&mut drawn)?;
    let (x, y, w, h) = bounding_rect(&mask)?;
    let depth = piece.depth.crop(x, y, x + w, y + h);
    Some(Part {
        tag: piece.tag,
        left: piece.left + x,
        top: piece.top + y,
        pixels: piece.pixels.crop(x, y, x + w, y + h),
        depth: Image::from_data(
            w,
            h,
            1,
            depth
                .data
                .iter()
                .map(|&d| (d.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect(),
        ),
        depth_median,
    })
}

/// Every drawn layer as the parts of a document, bottom first.
///
/// `drawn` is in the order the depth model left them in, which decides ties. `page` is the
/// picture the layers were drawn from, on the same canvas, which the nose and mouth take their
/// colour from: what the model drew for them is a smoothed guess, and they are small enough that
/// the original is always better.
pub fn assemble(drawn: &[Drawn], page: &Rgba, split_pairs: bool) -> Vec<Part> {
    let mut pieces: Vec<Piece> = drawn.iter().filter_map(load).collect();

    if split_pairs {
        for tag in SPLITS {
            if let Some(index) = pieces.iter().position(|p| p.tag == tag) {
                let piece = pieces.remove(index);
                pieces.extend(split(piece));
            }
        }
    }

    for piece in pieces.iter_mut() {
        if piece.tag == "nose" || piece.tag == "mouth" {
            let source = page.crop(
                piece.left,
                piece.top,
                piece.left + piece.pixels.width,
                piece.top + piece.pixels.height,
            );
            for p in 0..source.width * source.height {
                for c in 0..3 {
                    piece.pixels.data[p * 4 + c] = source.data[p * 4 + c];
                }
            }
        }
    }

    let mut parts: Vec<Part> = pieces.into_iter().filter_map(finish).collect();

    // Whatever the model thought, a nose and a mouth are in front of the face and the ears behind
    // it. Split ears are named `ears-r` and `ears-l`, which upstream's check misses, and so does
    // this one.
    if let Some(face) = parts
        .iter()
        .find(|p| p.tag == "face")
        .map(|p| p.depth_median)
    {
        for part in parts.iter_mut() {
            match part.tag.as_str() {
                "nose" | "mouth" | "eyes" if part.depth_median > face => {
                    part.depth_median = face - 0.001
                }
                "earr" | "earl" | "ears" => part.depth_median = face + 0.001,
                _ => {}
            }
        }
    }

    // Farthest first, ties in the order they came: a stable sort, as Python's is.
    parts.sort_by(|a, b| {
        b.depth_median
            .partial_cmp(&a.depth_median)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    parts
}

/// The depth of each part of a composite slot, from the one depth map the slot was given.
///
/// `hair` is drawn as a back and a front and `eyes` as four parts, but the depth model sees each
/// as one picture and says one depth for it. Each part takes that depth where it is drawn, and
/// where an earlier part -- one nearer the front, since they are taken front first -- already
/// covers it, the median of its own uncovered depth instead. Everywhere else is 255, the farthest.
///
/// `parts` is front first. What comes back is in the same order.
pub fn composite_depths(depth: &Image<u8>, parts: &[&Rgba]) -> Vec<Image<u8>> {
    let n = depth.width * depth.height;
    let mut covered = vec![false; n];
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        let own: Vec<bool> = (0..n).map(|p| part.data[p * 4 + 3] > LAYER_ALPHA).collect();
        let mut visible: Vec<f32> = (0..n)
            .filter(|&p| own[p] && !covered[p])
            .map(|p| depth.data[p] as f32)
            .collect();
        // An empty median is a NaN upstream, which numpy casts to 0.
        let hidden = median(&mut visible).map(|m| m as u8).unwrap_or(0);
        let mut local = Image::from_data(depth.width, depth.height, 1, vec![255u8; n]);
        for p in 0..n {
            if own[p] {
                local.data[p] = match covered[p] {
                    true => hidden,
                    false => depth.data[p],
                };
                covered[p] = true;
            }
        }
        out.push(local);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Rgba {
        let mut img = Image::new(w, h, 4);
        for y in y0..y1 {
            for x in x0..x1 {
                for c in 0..4 {
                    img.set(x, y, c, 200);
                }
            }
        }
        img
    }

    #[test]
    fn a_part_is_cut_to_where_it_is_drawn_and_ordered_by_depth() {
        let near = square(40, 40, 2, 2, 10, 10);
        let far = square(40, 40, 5, 5, 20, 20);
        let near_depth = Image::from_data(40, 40, 1, vec![30; 1600]);
        let far_depth = Image::from_data(40, 40, 1, vec![200; 1600]);
        let page = Image::new(40, 40, 4);
        let parts = assemble(
            &[
                Drawn {
                    tag: "face",
                    pixels: &near,
                    depth: &near_depth,
                },
                Drawn {
                    tag: "topwear",
                    pixels: &far,
                    depth: &far_depth,
                },
            ],
            &page,
            false,
        );
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].tag, "topwear");
        assert_eq!((parts[0].left, parts[0].top), (5, 5));
        assert_eq!((parts[0].pixels.width, parts[0].pixels.height), (15, 15));
        assert_eq!(parts[1].tag, "face");
    }

    #[test]
    fn a_pair_is_split_left_into_right() {
        let mut both = square(40, 40, 2, 2, 8, 8);
        both.paste(&square(10, 10, 0, 0, 6, 6), 20, 2);
        let depth = Image::from_data(40, 40, 1, vec![100; 1600]);
        let parts = assemble(
            &[Drawn {
                tag: "handwear",
                pixels: &both,
                depth: &depth,
            }],
            &Image::new(40, 40, 4),
            true,
        );
        let tags: Vec<&str> = parts.iter().map(|p| p.tag.as_str()).collect();
        assert_eq!(tags, vec!["handwear-r", "handwear-l"]);
        assert_eq!(parts[0].left, 2);
        assert_eq!(parts[1].left, 20);
    }

    #[test]
    fn two_pieces_larger_than_the_background_are_both_kept() {
        // A 61 by 10 crop: two 30 by 10 pieces a column apart. Each covers 300 pixels and the
        // background between them only 10.
        let mut both = square(61, 10, 0, 0, 30, 10);
        both.paste(&square(30, 10, 0, 0, 30, 10), 31, 0);
        let depth = Image::from_data(61, 10, 1, vec![100; 610]);
        let parts = assemble(
            &[Drawn {
                tag: "ears",
                pixels: &both,
                depth: &depth,
            }],
            &Image::new(61, 10, 4),
            true,
        );
        let tags: Vec<(&str, usize)> = parts.iter().map(|p| (p.tag.as_str(), p.left)).collect();
        assert_eq!(tags, vec![("ears-r", 0), ("ears-l", 31)]);
        assert!(parts.iter().all(|p| p.pixels.width == 30));
    }

    #[test]
    fn a_covered_part_takes_the_median_of_what_it_shows() {
        let front = square(4, 1, 0, 0, 2, 1);
        let back = square(4, 1, 0, 0, 4, 1);
        let depth = Image::from_data(4, 1, 1, vec![10, 20, 30, 50]);
        let out = composite_depths(&depth, &[&front, &back]);
        assert_eq!(out[0].data, vec![10, 20, 255, 255]);
        assert_eq!(out[1].data, vec![40, 40, 30, 50]);
    }
}
