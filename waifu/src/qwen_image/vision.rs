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

//! Qwen3-VL's vision tower: a picture in, the rows its slots in the prompt read out.
//!
//! A ViT of twenty-seven blocks over sixteen-pixel patches, read off `Qwen3VLVisionModel` and the
//! processor in front of it. What comes out is one row per 2 x 2 of patches -- the merger folds
//! four neighbours into one and projects them to the encoder's width -- and three more sets of
//! rows taken from blocks 8, 16 and 24 the same way, which the encoder adds to its first three
//! layers (DeepStack).
//!
//! # What is easy to get wrong here
//!
//! *The patches are in merge order, not raster order.* The processor lays them out a 2 x 2 block
//! at a time -- block row, block column, then the four inside -- so that four neighbours are four
//! consecutive rows and merging them is a reshape. The positions, the rotary table and the
//! interpolated position embeddings all follow the same order.
//!
//! *A picture is a clip of two frames.* The tower also reads video, so its patch embedding is a
//! 3-D convolution two frames deep, and a picture is handed to it twice. Two identical frames
//! through it are one frame through the sum of its two slices, which is what runs here.
//!
//! *The position embeddings are interpolated.* The tower learned a 48 x 48 table, and a picture's
//! grid is whatever size it is: each patch reads four entries of the table, bilinearly, with the
//! corners aligned. The taps and their weights are worked out on the host.
//!
//! *The rotation is two dimensional* and pairs a head's halves the way the language model does:
//! the first quarter of the angles is the row's and the second the column's.
//!
//! *The two mergers normalize at different widths.* The last one normalizes each patch and then
//! folds four together; the DeepStack ones fold first and normalize the fold.
//!
//! The residual stream is kept in float32 and the sublayers run narrow, as the denoiser's are.

use std::fmt;
use std::rc::Rc;

use super::config::VisionConfig;
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Device, Extent, Graph, Ir, RunContext, Tensor, Value, Weights,
};
use crate::krea2::dit::gelu_tanh;
use crate::krea2::text_encoder::rotate;
use crate::layers::{Embedding, LayerNorm, Linear};

/// Where each patch of a `rows` x `columns` grid sits, in the order the processor lays them out:
/// a 2 x 2 block at a time.
fn merge_order(rows: i32, columns: i32) -> Vec<(i32, i32)> {
    let mut order = Vec::with_capacity((rows * columns) as usize);
    for block_row in 0..rows / 2 {
        for block_column in 0..columns / 2 {
            for inner_row in 0..2 {
                for inner_column in 0..2 {
                    order.push((2 * block_row + inner_row, 2 * block_column + inner_column));
                }
            }
        }
    }
    order
}

/// The two entries of a `side`-long axis of the table that position `index` of a `size`-long axis
/// reads, and their weights: `F.interpolate(mode="bilinear", align_corners=True)`, as
/// `_interpolation_axis_taps_weights` writes it out. In float32, as the reference works it.
fn taps(index: i32, size: i32, side: i32) -> ([i32; 2], [f32; 2]) {
    let source = index as f32 * (side - 1) as f32 / (size - 1).max(1) as f32;
    let floor = source.floor();
    let fraction = source - floor;
    let first = (floor as i32).clamp(0, side - 1);
    let second = (floor as i32 + 1).clamp(0, side - 1);
    ([first, second], [1.0 - fraction, fraction])
}

/// `-1..=1` RGB, `(3, H, W)` as a flat slice, to the processor's patches: one row of `3 * P * P`
/// per patch, channel, then row, then column inside it, the patches in [`merge_order`].
fn patches(pixels: &[f32], height: i32, width: i32, patch: i32) -> Vec<f32> {
    let (rows, columns) = (height / patch, width / patch);
    let plane = (height * width) as usize;
    let mut out = Vec::with_capacity(3 * plane);
    for (row, column) in merge_order(rows, columns) {
        for channel in 0..3usize {
            for y in 0..patch {
                let start = channel * plane + ((row * patch + y) * width + column * patch) as usize;
                out.extend_from_slice(&pixels[start..start + patch as usize]);
            }
        }
    }
    out
}

/// Normalization over the last dimension, computed wide: the stream it reads is float32 and the
/// package's weights are half.
fn layer_norm(g: &Graph, x: Value, width: i32) -> Value {
    let weight = g.cast(g.load(LayerNorm::WEIGHT, &[width]), DType::Float);
    let bias = g.cast(g.load(LayerNorm::BIAS, &[width]), DType::Float);
    g.layer_norm(x, Some(weight), Some(bias), VisionConfig::NORM_EPS)
}

/// One block. `x` is `(1, N, D)` in float32, and so is what comes back.
fn block(
    config: &VisionConfig,
    g: &Graph,
    x: Value,
    cos: Value,
    sin: Value,
    narrow: DType,
) -> Value {
    let d = config.hidden_size;
    let (count, head_dim) = (config.num_heads, config.head_dim());
    let length = Extent::of(x, 1);

    let h = g.cast(layer_norm(&g.subgraph("norm1"), x, d), narrow);
    let qkv = Linear::graph(&g.subgraph("attn.qkv"), h, d, 3 * d, true);
    let split = |index: i32| {
        let part = g.contiguous(g.slice(qkv, 2, index * d, (index + 1) * d));
        g.view(part, [Extent::At(1), length, count.into(), head_dim.into()])
    };

    // Rotated in float32, as `apply_rotary_pos_emb_vision` does.
    let turn = |value: Value| {
        let wide = rotate(g, g.cast(value, DType::Float), cos, sin, head_dim / 2);
        g.contiguous(g.transpose(g.cast(wide, narrow), 1, 2))
    };
    let q = turn(split(0));
    let k = turn(split(1));
    let v = g.contiguous(g.transpose(split(2), 1, 2));

    let out = g.attention(q, k, v, false);
    let out = g.contiguous(g.transpose(out, 1, 2));
    let out = g.view(out, [Extent::At(1), length, d.into()]);
    let out = Linear::graph(&g.subgraph("attn.proj"), out, d, d, true);
    let x = g.add(x, g.cast(out, DType::Float));

    let h = g.cast(layer_norm(&g.subgraph("norm2"), x, d), narrow);
    let h = Linear::graph(&g.subgraph("mlp.linear_fc1"), h, d, config.mlp_size, true);
    let h = Linear::graph(
        &g.subgraph("mlp.linear_fc2"),
        gelu_tanh(g, h),
        config.mlp_size,
        d,
        true,
    );
    g.add(x, g.cast(h, DType::Float))
}

/// Four neighbouring patches into one row of the encoder's width. `x` is `(1, N, D)` in float32;
/// `merged` is how many rows that makes, `N / 4`. `after` is DeepStack's: fold, then normalize the
/// fold, where the last merger normalizes each patch first.
fn merger(
    config: &VisionConfig,
    g: &Graph,
    x: Value,
    merged: Extent,
    after: bool,
    narrow: DType,
) -> Value {
    let d = config.hidden_size;
    let folded = 4 * d;
    let fold = |value: Value| g.view(value, [merged, folded.into()]);

    let x = match after {
        true => layer_norm(&g.subgraph("norm"), fold(x), folded),
        false => fold(layer_norm(&g.subgraph("norm"), x, d)),
    };
    let h = Linear::graph(
        &g.subgraph("linear_fc1"),
        g.cast(x, narrow),
        folded,
        folded,
        true,
    );
    // `nn.GELU()`: the exact one, unlike the blocks'.
    Linear::graph(
        &g.subgraph("linear_fc2"),
        g.gelu(h),
        folded,
        config.out_hidden_size,
        true,
    )
}

/// The tower, written into `g`.
///
/// `patches` is `(N / 4, 4, 3 * P * P)`, already in merge order, so that both the number of
/// patches and the number of rows they merge to are lengths the graph can name. `pos_at{k}` and
/// `pos_weight{k}` are the four bilinear taps into the learned table, `(N,)` each.
fn write(config: &VisionConfig, float_type: DType, g: &Graph) {
    let d = config.hidden_size;
    let patch = config.patch_size;
    let width = 3 * patch * patch;

    let input = g.input("patches");
    let merged = Extent::of(input, 0);
    let length = Extent::Prod {
        value: input,
        from: 0,
        to: 2,
    };
    let pixels = g.cast(
        g.view(input, [Extent::At(1), length, width.into()]),
        float_type,
    );

    // The convolution's slices along time, summed: both frames of a picture are the same frame.
    let embed = g.subgraph("patch_embed.proj");
    let kernel = embed.load(
        Linear::WEIGHT,
        &[d, 3, config.temporal_patch_size, patch, patch],
    );
    let folded = (0..config.temporal_patch_size)
        .map(|frame| {
            g.cast(
                g.contiguous(g.slice(kernel, 2, frame, frame + 1)),
                DType::Float,
            )
        })
        .reduce(|left, right| g.add(left, right))
        .expect("a tower reads at least one frame");
    let folded = g.cast(g.view(folded, [d, width]), float_type);
    let x = g.add(
        g.matmul(pixels, g.transpose(folded, 0, 1)),
        embed.load(Linear::BIAS, &[d]),
    );

    // Four lookups into the table and their weights, summed in float32 as the reference does.
    // Each comes out `(N, D)`; the weight is `(N,)`, which broadcasts over a leading dimension
    // only, so the rows are turned to columns for the product and back afterwards.
    let table = g.subgraph("pos_embed");
    let table = table.load(Embedding::WEIGHT, &[config.num_position_embeddings, d]);
    let position = (0..4)
        .map(|tap| {
            let rows = g.cast(
                g.lookup(table, g.input(&format!("pos_at{tap}"))),
                DType::Float,
            );
            g.mul(
                g.transpose(rows, 0, 1),
                g.input(&format!("pos_weight{tap}")),
            )
        })
        .reduce(|left, right| g.add(left, right))
        .expect("four taps");
    let position = g.contiguous(g.transpose(position, 0, 1));
    let mut x = g.add(g.cast(x, DType::Float), position);

    let cos = g.input("rope_cos");
    let sin = g.input("rope_sin");

    for index in 0..config.depth {
        x = block(
            config,
            &g.subgraph(&format!("blocks.{index}")),
            x,
            cos,
            sin,
            float_type,
        );
        if let Some(at) = config
            .deepstack_layers
            .iter()
            .position(|&layer| layer == index)
        {
            let sub = g.subgraph(&format!("deepstack_merger_list.{at}"));
            let rows = merger(config, &sub, x, merged, true, float_type);
            g.output(&format!("deepstack{at}"), rows);
        }
    }

    let rows = merger(config, &g.subgraph("merger"), x, merged, false, float_type);
    g.output("embeds", rows);
}

/// What the tower made of one picture.
pub struct SeenPicture {
    /// The grid of the prompt's slots: a quarter of the patches each way.
    pub rows: i32,
    pub columns: i32,
    /// `(rows * columns, D)`, one per slot, row by row.
    pub embeds: Tensor,
    /// `(rows * columns, D)` each, one per DeepStack layer.
    pub deepstack: Vec<Tensor>,
}

pub struct VisionTower {
    config: VisionConfig,
    ir: Ir,
    weights: Rc<Weights>,
}

impl fmt::Debug for VisionTower {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VisionTower({} instructions)", self.ir.len())
    }
}

impl VisionTower {
    /// `name` is the namespace the package holds the tower under. Its weights are half whatever
    /// the rest of the package is: the exporter never quantizes them.
    pub fn build(
        config: VisionConfig,
        name: &str,
        weights: &Rc<Weights>,
        float_type: DType,
    ) -> Result<VisionTower> {
        let graph = Graph::new();
        write(&config, float_type, &graph.subgraph(name));
        check_parameters(&graph, weights.as_ref())?;

        Ok(VisionTower {
            ir: Ir::compile(&graph),
            weights: Rc::clone(weights),
            config,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// `picture` is `(1, 3, H, W)` in `-1..=1` -- the processor's `x / 127.5 - 1` -- with both
    /// sides multiples of [`VisionConfig::alignment`]. It is read at that size: the processor
    /// would resize it to the multiple nearest, and it already is one.
    pub fn forward(&self, picture: &Tensor) -> Result<SeenPicture> {
        let shape = picture.shape();
        let alignment = self.config.alignment();
        let [1, 3, height, width] = shape[..] else {
            return Err(Error::model(format!(
                "the vision tower reads a picture as (1, 3, H, W), got {shape:?}"
            )));
        };
        if height % alignment != 0 || width % alignment != 0 {
            return Err(Error::model(format!(
                "the vision tower reads a picture whose sides are multiples of {alignment}, got \
                 {width} x {height}"
            )));
        }

        let device = picture.device();
        let patch = self.config.patch_size;
        let (rows, columns) = (height / patch, width / patch);
        let count = rows * columns;

        let pixels = picture
            .contiguous()?
            .to_device(Device::Cpu)?
            .cast(DType::Float)?
            .to_vec_f32()?;
        let width3 = 3 * patch * patch;
        let laid = patches(&pixels, height, width, patch);
        let laid = Tensor::from_f32(&[count / 4, 4, width3], &laid)?.to_device(device)?;

        let order = merge_order(rows, columns);
        let side = self.config.grid_side();
        let mut at = vec![Vec::with_capacity(count as usize); 4];
        let mut weight = vec![Vec::with_capacity(count as usize); 4];
        for &(row, column) in &order {
            let (row_taps, row_weights) = taps(row, rows, side);
            let (column_taps, column_weights) = taps(column, columns, side);
            for i in 0..2 {
                for j in 0..2 {
                    at[2 * i + j].push((row_taps[i] * side + column_taps[j]) as i64);
                    weight[2 * i + j].push(row_weights[i] * column_weights[j]);
                }
            }
        }

        // `Qwen3VLVisionRotaryEmbedding`: a quarter of the head's angles turn with the row and
        // a quarter with the column, and the rotation pairs the head's two halves.
        let heads = self.config.num_heads as usize;
        let quarter = (self.config.head_dim() / 4) as usize;
        let mut cos = Vec::with_capacity(count as usize * heads * 2 * quarter);
        let mut sin = Vec::with_capacity(count as usize * heads * 2 * quarter);
        let frequencies = (0..quarter)
            .map(|index| {
                let exponent = (2 * index) as f32 / (2 * quarter) as f32;
                1.0 / (VisionConfig::ROPE_THETA as f32).powf(exponent)
            })
            .collect::<Vec<_>>();
        for &(row, column) in &order {
            let angles = frequencies
                .iter()
                .map(|f| row as f32 * f)
                .chain(frequencies.iter().map(|f| column as f32 * f))
                .collect::<Vec<_>>();
            for _ in 0..heads {
                cos.extend(angles.iter().map(|angle| angle.cos()));
                sin.extend(angles.iter().map(|angle| angle.sin()));
            }
        }
        let table = [count, self.config.num_heads, 2 * quarter as i32];
        let cos = Tensor::from_f32(&table, &cos)?.to_device(device)?;
        let sin = Tensor::from_f32(&table, &sin)?.to_device(device)?;

        let at = at
            .iter()
            .map(|indices| Ok(Tensor::from_i64(&[count], indices)?.to_device(device)?))
            .collect::<Result<Vec<_>>>()?;
        let weight = weight
            .iter()
            .map(|weights| Ok(Tensor::from_f32(&[count], weights)?.to_device(device)?))
            .collect::<Result<Vec<_>>>()?;
        let names = (0..4)
            .map(|tap| (format!("pos_at{tap}"), format!("pos_weight{tap}")))
            .collect::<Vec<_>>();

        let mut run = RunContext::new(&*self.weights)
            .input("patches", &laid)
            .input("rope_cos", &cos)
            .input("rope_sin", &sin);
        for (tap, (at_name, weight_name)) in names.iter().enumerate() {
            run = run
                .input(at_name, &at[tap])
                .input(weight_name, &weight[tap]);
        }

        let mut outputs = self.ir.run(&run)?;
        let mut take = |name: &str| {
            outputs
                .iter()
                .position(|(output, _)| output == name)
                .map(|index| outputs.swap_remove(index).1)
                .ok_or_else(|| Error::model(format!("the vision tower produced no {name}")))
        };
        let embeds = take("embeds")?;
        let deepstack = (0..self.config.deepstack_layers.len())
            .map(|index| take(&format!("deepstack{index}")))
            .collect::<Result<Vec<_>>>()?;

        Ok(SeenPicture {
            rows: rows / 2,
            columns: columns / 2,
            embeds,
            deepstack,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lays_the_patches_out_a_block_at_a_time() {
        // `get_vision_position_ids` for a 4 x 4 grid: block (0, 0), then block (0, 1), each
        // block's four row by row.
        let order = merge_order(4, 4);
        assert_eq!(&order[..4], &[(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(&order[4..8], &[(0, 2), (0, 3), (1, 2), (1, 3)]);
        assert_eq!(order[8], (2, 0));
        assert_eq!(order.len(), 16);
    }

    #[test]
    fn interpolates_with_the_corners_aligned() {
        // The ends of the picture read the ends of the table, whole.
        assert_eq!(taps(0, 64, 48), ([0, 1], [1.0, 0.0]));
        assert_eq!(taps(63, 64, 48), ([47, 47], [1.0, 0.0]));
        // A grid of one reads the first entry, as `clamp(size - 1, min=1)` makes it.
        assert_eq!(taps(0, 1, 48), ([0, 1], [1.0, 0.0]));
        // Halfway along a grid of 95 is entry 23.5 of 48.
        let ([first, second], [near, far]) = taps(47, 95, 48);
        assert_eq!((first, second), (23, 24));
        assert!((near - 0.5).abs() < 1e-6 && (far - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cuts_patches_the_way_the_processor_does() {
        // A 2 x 4 grid of 1-pixel patches over three channels: the value is channel * 100 + the
        // pixel's raster index, so the order the patches come out in is readable.
        let (height, width) = (2, 4);
        let pixels = (0..3)
            .flat_map(|channel| (0..height * width).map(move |at| (channel * 100 + at) as f32))
            .collect::<Vec<_>>();
        let laid = patches(&pixels, height, width, 1);
        // The first block is pixels 0, 1, 4, 5; each patch is its three channels.
        assert_eq!(&laid[..6], &[0.0, 100.0, 200.0, 1.0, 101.0, 201.0]);
        assert_eq!(&laid[6..12], &[4.0, 104.0, 204.0, 5.0, 105.0, 205.0]);
        assert_eq!(&laid[12..15], &[2.0, 102.0, 202.0]);
    }
}
