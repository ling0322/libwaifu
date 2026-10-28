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

//! The picture handling See-through does on the host, between and around its two models.
//!
//! Upstream does all of this with OpenCV and numpy, and what is here is those calls written out:
//! `cv2.resize` as its area and linear filters, `torch.nn.functional.interpolate` as bilinear
//! with `align_corners=False`, `boundingRect`, `connectedComponentsWithStats` and a 3 by 3
//! dilation. Each says which call it stands for. None of them is bit for bit OpenCV -- OpenCV
//! resizes eight-bit pictures in fixed point -- and none needs to be: what comes out of them is
//! read by a model or thresholded, and a value one apart on 255 moves neither.

/// A picture `width` by `height`, `channels` values a pixel, row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct Image<T> {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<T>,
}

/// Eight-bit RGBA, not premultiplied: how a picture comes in and how a layer goes out.
pub type Rgba = Image<u8>;

impl<T: Copy + Default> Image<T> {
    pub fn new(width: usize, height: usize, channels: usize) -> Image<T> {
        Image {
            width,
            height,
            channels,
            data: vec![T::default(); width * height * channels],
        }
    }

    pub fn from_data(width: usize, height: usize, channels: usize, data: Vec<T>) -> Image<T> {
        assert_eq!(
            data.len(),
            width * height * channels,
            "a picture's size and its data"
        );
        Image {
            width,
            height,
            channels,
            data,
        }
    }

    pub fn at(&self, x: usize, y: usize, c: usize) -> T {
        self.data[(y * self.width + x) * self.channels + c]
    }

    pub fn set(&mut self, x: usize, y: usize, c: usize, value: T) {
        self.data[(y * self.width + x) * self.channels + c] = value;
    }

    /// The pixels in `x0..x1` by `y0..y1`, clipped to the picture the way a numpy slice is.
    pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Image<T> {
        let (x1, y1) = (x1.min(self.width), y1.min(self.height));
        let (x0, y0) = (x0.min(x1), y0.min(y1));
        let mut out = Image::new(x1 - x0, y1 - y0, self.channels);
        for y in y0..y1 {
            let from = (y * self.width + x0) * self.channels;
            let to = ((y - y0) * out.width) * self.channels;
            let n = (x1 - x0) * self.channels;
            out.data[to..to + n].copy_from_slice(&self.data[from..from + n]);
        }
        out
    }

    /// `other` written over this picture with its top left at `(x, y)`, whatever of it fits.
    pub fn paste(&mut self, other: &Image<T>, x: usize, y: usize) {
        assert_eq!(
            self.channels, other.channels,
            "pasting between channel counts"
        );
        for row in 0..other.height {
            if y + row >= self.height {
                break;
            }
            let n = other.width.min(self.width.saturating_sub(x));
            let to = ((y + row) * self.width + x) * self.channels;
            let from = row * other.width * other.channels;
            self.data[to..to + n * self.channels]
                .copy_from_slice(&other.data[from..from + n * self.channels]);
        }
    }

    /// Channel `c` alone.
    pub fn channel(&self, c: usize) -> Image<T> {
        Image::from_data(
            self.width,
            self.height,
            1,
            self.data
                .iter()
                .skip(c)
                .step_by(self.channels)
                .copied()
                .collect(),
        )
    }
}

impl Image<u8> {
    pub fn to_f32(&self) -> Image<f32> {
        Image::from_data(
            self.width,
            self.height,
            self.channels,
            self.data.iter().map(|&v| v as f32).collect(),
        )
    }
}

impl Image<f32> {
    /// Rounded to the nearest and saturated, which is what OpenCV writes a float result back to
    /// eight bits as.
    pub fn to_u8_rounded(&self) -> Image<u8> {
        Image::from_data(
            self.width,
            self.height,
            self.channels,
            self.data
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect(),
        )
    }

    /// Clipped and truncated, which is what numpy's `.astype(np.uint8)` of a clipped array does.
    pub fn to_u8_truncated(&self) -> Image<u8> {
        Image::from_data(
            self.width,
            self.height,
            self.channels,
            self.data
                .iter()
                .map(|&v| v.clamp(0.0, 255.0) as u8)
                .collect(),
        )
    }
}

/// `cv2.resize(..., INTER_AREA)` for a picture made smaller: every pixel out is the mean of the
/// pixels in, weighted by how much of each it covers.
pub fn resize_area(src: &Image<f32>, width: usize, height: usize) -> Image<f32> {
    let weights = |from: usize, to: usize| -> Vec<Vec<(usize, f32)>> {
        let scale = from as f64 / to as f64;
        (0..to)
            .map(|i| {
                let (lo, hi) = (i as f64 * scale, (i + 1) as f64 * scale);
                let mut out = Vec::new();
                let mut s = lo.floor() as usize;
                while (s as f64) < hi && s < from {
                    let cover = (hi.min(s as f64 + 1.0) - lo.max(s as f64)) / scale;
                    if cover > 1e-12 {
                        out.push((s, cover as f32));
                    }
                    s += 1;
                }
                out
            })
            .collect()
    };
    separable(
        src,
        width,
        height,
        &weights(src.width, width),
        &weights(src.height, height),
    )
}

/// `cv2.resize(..., INTER_LINEAR)`: half pixel centres, the edge pixel repeated beyond the edge.
pub fn resize_linear(src: &Image<f32>, width: usize, height: usize) -> Image<f32> {
    separable(
        src,
        width,
        height,
        &linear_weights(src.width, width),
        &linear_weights(src.height, height),
    )
}

/// The two taps of a linear filter with half pixel centres, for one axis.
///
/// The same arithmetic serves `cv2.resize` and PyTorch's bilinear interpolate with
/// `align_corners=False` and no antialiasing: both place output pixel `i` at
/// `(i + 0.5) * from / to - 0.5` in the input and clamp below at zero.
fn linear_weights(from: usize, to: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = from as f64 / to as f64;
    (0..to)
        .map(|i| {
            let at = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
            let lo = (at.floor() as usize).min(from - 1);
            let hi = (lo + 1).min(from - 1);
            let t = (at - lo as f64) as f32;
            match lo == hi {
                true => vec![(lo, 1.0)],
                false => vec![(lo, 1.0 - t), (hi, t)],
            }
        })
        .collect()
}

/// A separable filter: every row through `horizontal`, then every column through `vertical`.
fn separable(
    src: &Image<f32>,
    width: usize,
    height: usize,
    horizontal: &[Vec<(usize, f32)>],
    vertical: &[Vec<(usize, f32)>],
) -> Image<f32> {
    let c = src.channels;
    let mut rows = Image::new(width, src.height, c);
    for y in 0..src.height {
        for (x, taps) in horizontal.iter().enumerate() {
            for k in 0..c {
                let mut sum = 0.0;
                for &(s, w) in taps {
                    sum += w * src.data[(y * src.width + s) * c + k];
                }
                rows.data[(y * width + x) * c + k] = sum;
            }
        }
    }
    let mut out = Image::new(width, height, c);
    for (y, taps) in vertical.iter().enumerate() {
        for x in 0..width {
            for k in 0..c {
                let mut sum = 0.0;
                for &(s, w) in taps {
                    sum += w * rows.data[(s * width + x) * c + k];
                }
                out.data[(y * width + x) * c + k] = sum;
            }
        }
    }
    out
}

/// See-through's `smart_resize`: area to shrink, linear to grow, and nothing for the same size.
pub fn smart_resize(src: &Image<f32>, width: usize, height: usize) -> Image<f32> {
    if width == src.width && height == src.height {
        return src.clone();
    }
    match width * height < src.width * src.height {
        true => resize_area(src, width, height),
        false => resize_linear(src, width, height),
    }
}

/// [`smart_resize`] of an eight-bit picture, rounded back to eight bits.
pub fn smart_resize_u8(src: &Image<u8>, width: usize, height: usize) -> Image<u8> {
    if width == src.width && height == src.height {
        return src.clone();
    }
    smart_resize(&src.to_f32(), width, height).to_u8_rounded()
}

/// What `center_square_pad_resize` did to a picture, so that what is found in the square can be
/// put back where it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Square {
    /// The side of the padded square, before it was scaled.
    pub side: usize,
    /// Where the picture sits in it.
    pub x: usize,
    pub y: usize,
}

/// The picture centred in a square of transparent black, then scaled to `target` on a side.
pub fn center_square_pad_resize(src: &Rgba, target: usize) -> (Rgba, Square) {
    let side = src.width.max(src.height);
    let (x, y) = ((side - src.width) / 2, (side - src.height) / 2);
    let padded = match src.width == src.height {
        true => src.clone(),
        false => {
            let mut padded = Image::new(side, side, src.channels);
            padded.paste(src, x, y);
            padded
        }
    };
    let (x, y) = match src.width == src.height {
        true => (0, 0),
        false => (x, y),
    };
    (
        smart_resize_u8(&padded, target, target),
        Square { side, x, y },
    )
}

/// `cv2.boundingRect(cv2.findNonZero(mask))` as `(x, y, width, height)`, or None for an empty
/// mask -- where OpenCV would be handed None and fail.
pub fn bounding_rect(mask: &Image<bool>) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..mask.height {
        for x in 0..mask.width {
            if mask.data[y * mask.width + x] {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    match x0 == usize::MAX {
        true => None,
        false => Some((x0, y0, x1 - x0, y1 - y0)),
    }
}

/// Where channel `c` of `image` is above `threshold`.
pub fn mask_above(image: &Rgba, c: usize, threshold: u8) -> Image<bool> {
    Image::from_data(
        image.width,
        image.height,
        1,
        image
            .data
            .iter()
            .skip(c)
            .step_by(image.channels)
            .map(|&v| v > threshold)
            .collect(),
    )
}

/// One label of [`connected_components`]: its bounding box and how many pixels it has.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Component {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub area: usize,
}

/// `cv2.connectedComponentsWithStats(mask, connectivity=8)`: a label per pixel, and the stats of
/// each label, the background's first.
///
/// Labels are numbered in the order a raster scan first meets them, which is OpenCV's order too.
pub fn connected_components(mask: &Image<bool>) -> (Vec<u32>, Vec<Component>) {
    let (w, h) = (mask.width, mask.height);
    let mut labels = vec![0u32; w * h];
    let mut stats = vec![Component {
        x: 0,
        y: 0,
        width: w,
        height: h,
        area: 0,
    }];
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask.data[start] || labels[start] != 0 {
            continue;
        }
        let label = stats.len() as u32;
        let (mut x0, mut y0, mut x1, mut y1, mut area) = (w, h, 0, 0, 0);
        labels[start] = label;
        stack.push(start);
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            area += 1;
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let q = ny as usize * w + nx as usize;
                    if mask.data[q] && labels[q] == 0 {
                        labels[q] = label;
                        stack.push(q);
                    }
                }
            }
        }
        stats.push(Component {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
            area,
        });
    }
    let background = labels.iter().filter(|&&l| l == 0).count();
    stats[0].area = background;
    (labels, stats)
}

/// `cv2.dilate` by `getStructuringElement(MORPH_ELLIPSE, (3, 3))`, which is the 3 by 3 cross.
pub fn dilate_cross(mask: &Image<bool>) -> Image<bool> {
    let (w, h) = (mask.width, mask.height);
    let mut out = mask.clone();
    for y in 0..h {
        for x in 0..w {
            if mask.data[y * w + x] {
                continue;
            }
            let near = (x > 0 && mask.data[y * w + x - 1])
                || (x + 1 < w && mask.data[y * w + x + 1])
                || (y > 0 && mask.data[(y - 1) * w + x])
                || (y + 1 < h && mask.data[(y + 1) * w + x]);
            out.data[y * w + x] = near;
        }
    }
    out
}

/// `np.median`: the middle value, or the mean of the two middle values. None for no values.
pub fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    Some(match n % 2 {
        1 => values[n / 2],
        _ => (values[n / 2 - 1] + values[n / 2]) / 2.0,
    })
}

/// `img_alpha_blending(..., premultiplied=False)` with no depth: each picture laid over the ones
/// before it, the colour accumulated premultiplied and left that way, the alpha summed and clipped.
///
/// The colour that comes out is not divided back by its alpha. That is upstream's, and what its
/// depth model was trained on reading, so it is kept.
pub fn blend_over(pictures: &[&Rgba]) -> Rgba {
    let (w, h) = (pictures[0].width, pictures[0].height);
    let mut rgb = vec![0f32; w * h * 3];
    let mut alpha = vec![0f32; w * h];
    for picture in pictures {
        for p in 0..w * h {
            let a = picture.data[p * 4 + 3] as f32 / 255.0;
            alpha[p] = (alpha[p] + a).min(1.0);
            for c in 0..3 {
                rgb[p * 3 + c] = rgb[p * 3 + c] * (1.0 - a) + picture.data[p * 4 + c] as f32 * a;
            }
        }
    }
    let mut out = Image::new(w, h, 4);
    for p in 0..w * h {
        for c in 0..3 {
            out.data[p * 4 + c] = rgb[p * 3 + c].clamp(0.0, 255.0) as u8;
        }
        out.data[p * 4 + 3] = (alpha[p] * 255.0).clamp(0.0, 255.0) as u8;
    }
    out
}

/// The colour of `argb` with its transparent parts filled in from what surrounds them: Lvmin's
/// `pad_rgb_torch`, in `[0, 1]`.
///
/// A pyramid of the premultiplied picture, each level a sixth smaller than the last, is walked
/// from the single pixel at its top back down, and at each level the colour there is laid over
/// the blurred colour from above. What is opaque keeps its own colour and what is transparent takes
/// its surroundings', which is something an autoencoder can encode without inventing an edge.
///
/// The colour coming out is the premultiplied composite, as upstream's is. `rgba` is RGBA in
/// `[0, 1]`, not premultiplied.
pub fn pad_rgb(rgba: &Image<f32>) -> Image<f32> {
    let (w, h) = (rgba.width, rgba.height);
    // Premultiplied, alpha first, as upstream holds it.
    let mut level = Image::new(w, h, 4);
    for p in 0..w * h {
        let a = rgba.data[p * 4 + 3];
        level.data[p * 4] = a;
        for c in 0..3 {
            level.data[p * 4 + 1 + c] = rgba.data[p * 4 + c] * a;
        }
    }

    let mut pyramid = vec![level];
    loop {
        let last = pyramid.last().expect("never empty");
        if last.width.min(last.height) == 1 {
            break;
        }
        // `int(W / 1.2)`: truncated. Upstream hands interpolate the width as the height and the
        // height as the width, which for the square pictures it is given is the same thing.
        let (nw, nh) = (
            ((last.width as f64) / 1.2) as usize,
            ((last.height as f64) / 1.2) as usize,
        );
        let next = resize_linear(last, nw.max(1), nh.max(1));
        pyramid.push(next);
    }
    pyramid.reverse();

    // The mean colour of the whole picture, weighted by alpha.
    let top = &pyramid[0];
    let total_alpha: f32 = (0..top.width * top.height).map(|p| top.data[p * 4]).sum();
    let mut fg = Image::new(1, 1, 3);
    for c in 0..3 {
        let sum: f32 = (0..top.width * top.height)
            .map(|p| top.data[p * 4 + 1 + c])
            .sum();
        fg.data[c] = sum / total_alpha.max(1e-8);
    }

    for level in &pyramid {
        let up = resize_linear(&fg, level.width, level.height);
        let mut next = Image::new(level.width, level.height, 3);
        for p in 0..level.width * level.height {
            let a = level.data[p * 4];
            for c in 0..3 {
                next.data[p * 3 + c] = level.data[p * 4 + 1 + c] + up.data[p * 3 + c] * (1.0 - a);
            }
        }
        fg = next;
    }
    fg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_shrinks_by_averaging_whole_blocks() {
        let src = Image::from_data(4, 2, 1, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
        let out = resize_area(&src, 2, 1);
        assert_eq!(out.data, vec![5.0, 9.0]);
    }

    #[test]
    fn area_weighs_a_partly_covered_pixel_by_its_share() {
        // Three pixels into two: each output covers one and a half inputs.
        let src = Image::from_data(3, 1, 1, vec![0.0, 3.0, 6.0]);
        let out = resize_area(&src, 2, 1);
        assert!((out.data[0] - 1.0).abs() < 1e-6);
        assert!((out.data[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn linear_doubles_with_half_pixel_centres() {
        let src = Image::from_data(2, 1, 1, vec![0.0, 4.0]);
        let out = resize_linear(&src, 4, 1);
        // Centres at -0.25 (clamped to 0), 0.25, 0.75, 1.25 (past the last).
        assert_eq!(out.data, vec![0.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn a_wide_picture_is_centred_in_its_square() {
        let src = Image::from_data(4, 2, 4, vec![255; 4 * 2 * 4]);
        let (out, square) = center_square_pad_resize(&src, 4);
        assert_eq!(
            square,
            Square {
                side: 4,
                x: 0,
                y: 1
            }
        );
        assert_eq!(out.at(0, 0, 3), 0);
        assert_eq!(out.at(0, 1, 3), 255);
        assert_eq!(out.at(0, 3, 3), 0);
    }

    #[test]
    fn components_are_eight_connected_and_counted_in_raster_order() {
        #[rustfmt::skip]
        let mask = Image::from_data(5, 3, 1, vec![
            true,  false, false, false, true,
            false, true,  false, false, true,
            false, false, false, false, false,
        ]);
        let (labels, stats) = connected_components(&mask);
        assert_eq!(stats.len(), 3);
        assert_eq!(labels[0], 1);
        assert_eq!(labels[6], 1);
        assert_eq!(labels[4], 2);
        assert_eq!(stats[1].area, 2);
        assert_eq!(
            (stats[2].x, stats[2].y, stats[2].width, stats[2].height),
            (4, 0, 1, 2)
        );
        assert_eq!(stats[0].area, 11);
    }

    #[test]
    fn the_cross_dilates_without_its_corners() {
        let mut mask = Image::new(3, 3, 1);
        mask.data[4] = true;
        let out = dilate_cross(&mask);
        assert_eq!(
            out.data,
            vec![false, true, false, true, true, true, false, true, false]
        );
    }

    #[test]
    fn an_opaque_picture_keeps_its_own_colour_when_padded() {
        let rgba = Image::from_data(3, 3, 4, [0.25, 0.5, 0.75, 1.0].repeat(9));
        let out = pad_rgb(&rgba);
        for p in 0..9 {
            assert!((out.data[p * 3] - 0.25).abs() < 1e-6);
            assert!((out.data[p * 3 + 2] - 0.75).abs() < 1e-6);
        }
    }

    #[test]
    fn a_transparent_pixel_takes_its_neighbours_colour() {
        let mut data = [1.0, 0.0, 0.0, 1.0].repeat(16);
        data[5 * 4 + 3] = 0.0;
        let out = pad_rgb(&Image::from_data(4, 4, 4, data));
        assert!(out.data[5 * 3] > 0.9, "{}", out.data[5 * 3]);
    }

    #[test]
    fn the_median_of_an_even_count_is_the_mean_of_the_middle_two() {
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut []), None);
    }
}
