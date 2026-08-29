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

//! The layers every model here is built from.
//!
//! Each one is built from a [`VarBuilder`] pointing at its own namespace, which is what ties a
//! layer to the weights stored for it, and each checks the shape of what it reads so that a
//! mismatched model package is caught while it loads rather than part way through a forward pass.

use crate::flint::{functional as F, Tensor};

use crate::error::{Error, Result};
use crate::var_builder::VarBuilder;

/// Normalization over the last dimension: subtract the mean, divide by the standard deviation,
/// scale and shift.
#[derive(Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub const WEIGHT: &'static str = "weight";
    pub const BIAS: &'static str = "bias";

    pub fn build(d_model: i32, eps: f32, vb: &VarBuilder) -> Result<LayerNorm> {
        Ok(LayerNorm {
            weight: vb.get(Self::WEIGHT, &[d_model])?,
            bias: vb.get(Self::BIAS, &[d_model])?,
            eps,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(F::layer_norm(
            input,
            Some(&self.weight),
            Some(&self.bias),
            self.eps,
        )?)
    }
}

/// A table of word embeddings, read by token id.
#[derive(Debug)]
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    pub const WEIGHT: &'static str = "weight";

    pub fn build(d_model: i32, vocab_size: i32, vb: &VarBuilder) -> Result<Embedding> {
        Ok(Embedding {
            weight: vb.get(Self::WEIGHT, &[vocab_size, d_model])?,
        })
    }

    /// The embeddings of `input` `<long>(L)`, as `<float>(L, D)`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dim = input.dim()?;
        if dim != 1 {
            return Err(Error::model(format!(
                "embedding takes packed token ids as <long>(L), got a {dim}-D tensor"
            )));
        }

        Ok(F::lookup(&self.weight, input)?)
    }
}

/// A two dimensional convolution, with a bias, as every convolution in a diffusion model has one.
///
/// Only what those models ask for: a square kernel, one group, and no dilation.
#[derive(Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Tensor,
    stride: i32,
    padding: i32,
}

impl Conv2d {
    pub fn build(
        in_channels: i32,
        out_channels: i32,
        kernel: i32,
        stride: i32,
        padding: i32,
        vb: &VarBuilder,
    ) -> Result<Conv2d> {
        Ok(Conv2d {
            weight: vb.get("weight", &[out_channels, in_channels, kernel, kernel])?,
            bias: vb.get("bias", &[out_channels])?,
            stride,
            padding,
        })
    }

    /// `input` is `(N, C, H, W)`, and so is what comes back.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(F::conv2d(
            input,
            &self.weight,
            Some(&self.bias),
            self.stride,
            self.padding,
            1,
            1,
        )?)
    }
}

/// Normalization over a group of channels and all of the space they cover.
///
/// This is what a diffusion model normalizes with. A batch of one image says nothing about its
/// own statistics, which is why the mean and variance are taken this way rather than over the
/// batch.
#[derive(Debug)]
pub struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    groups: i32,
    eps: f32,
}

impl GroupNorm {
    pub fn build(channels: i32, groups: i32, eps: f32, vb: &VarBuilder) -> Result<GroupNorm> {
        if groups <= 0 || channels % groups != 0 {
            return Err(Error::model(format!(
                "module {:?}: {channels} channels do not divide into {groups} groups",
                vb.name()
            )));
        }

        Ok(GroupNorm {
            weight: vb.get("weight", &[channels])?,
            bias: vb.get("bias", &[channels])?,
            groups,
            eps,
        })
    }

    /// `input` is `(N, C, H, W)`, and so is what comes back.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(F::group_norm(
            input,
            Some(&self.weight),
            Some(&self.bias),
            self.groups,
            self.eps,
        )?)
    }
}

/// A fully connected layer, with an optional bias.
#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub const WEIGHT: &'static str = "weight";
    pub const BIAS: &'static str = "bias";

    pub fn build(in_dim: i32, out_dim: i32, has_bias: bool, vb: &VarBuilder) -> Result<Linear> {
        if in_dim <= 0 || out_dim <= 0 {
            return Err(Error::model(format!(
                "linear layer {:?} has an invalid shape ({in_dim}, {out_dim})",
                vb.name()
            )));
        }

        let weight = vb.get(Self::WEIGHT, &[out_dim, in_dim])?;
        let bias = if has_bias {
            Some(vb.get(Self::BIAS, &[out_dim])?)
        } else {
            // A bias in the package that the model does not expect means the two disagree about
            // what this layer is, which is worth saying now rather than quietly ignoring.
            if vb.has(Self::BIAS) {
                return Err(Error::model(format!(
                    "module {:?}: has_bias is false but the model holds a bias",
                    vb.name()
                )));
            }
            None
        };

        Ok(Linear { weight, bias })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dim = input.dim()?;
        if dim < 2 {
            return Err(Error::model(format!(
                "linear layer takes at least a 2-D input, got a {dim}-D tensor"
            )));
        }

        let x = F::matmul(input, &self.weight.transpose(0, 1)?)?;
        match &self.bias {
            Some(bias) => Ok(F::add(&x, bias)?),
            None => Ok(x),
        }
    }
}
