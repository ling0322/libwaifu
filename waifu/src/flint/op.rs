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

//! What a node of a [`Graph`](super::Graph) is: an operation, the values it reads, and the
//! settings that are not values.
//!
//! One variant per operation of [`functional`](super::functional), except that the
//! families whose members differ in nothing but which kernel runs -- the pointwise unary
//! operations, the pointwise binary ones, the reductions -- are one variant carrying a kind,
//! since anything walking a graph treats all of them the same way and only the executor cares
//! which is which.

use std::fmt;

use super::{Bound, DType, Device, Tensor};

/// One value in a graph: a symbol, and nothing more.
///
/// What it says is whether two values are the same value. It does not say what produced it, where
/// that is kept, or when it runs. Those are things a graph answers *about* a value rather than
/// things the value carries, and [`Graph::op`](super::Graph::op),
/// [`Graph::site`](super::Graph::site) and [`Graph::nodes`](super::Graph::nodes) are where to ask
/// them. `%3` is its name, not its position.
///
/// Deliberately not [`Ord`]: comparing two symbols could only be a way of asking which of them
/// runs first, and that is the question this does not answer. It happens to be true today that
/// the graph numbers its symbols in the order it runs them, which is exactly why the comparison
/// is not offered -- code that leant on it would keep working right up until a pass moved a node,
/// and then be wrong silently. Walk [`Graph::nodes`](super::Graph::nodes) instead, which is the
/// run order because it says it is.
///
/// Only a [`Graph`](super::Graph) makes one, and it is only good against the graph that made it:
/// one from somewhere else names a node here by accident or names nothing at all.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Value(u32);

impl Value {
    /// Where the graph that made this keeps the node behind it.
    ///
    /// The one place a symbol is read as a position, and it stays inside `nn` for that reason: a
    /// graph handing out the slot a node went into as that node's symbol is the graph's own
    /// business, and the day it stops doing that -- a pass that moves a node, an operation with
    /// two results -- nothing outside will have come to depend on it, because nothing outside
    /// could ask.
    pub(super) fn slot(self) -> usize {
        self.0 as usize
    }

    /// The symbol for the node kept in `slot`.
    pub(super) fn at(slot: usize) -> Value {
        Value(slot as u32)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// A size a node was written with: one the graph knew, or one it will read off a value when it
/// runs.
///
/// The one thing in a graph that is a number rather than a tensor. It is here because a graph is
/// written before anything it will be handed exists, and the alternative -- a number the builder
/// had to know -- is a pass that is written for one prompt length or one picture size and has to
/// be written again for the next. An [`Extent::Of`] says *however long that turns out to be*, and
/// is resolved against the tensor in that slot when the instruction runs.
///
/// A value an extent names is a value the node reads, and [`Op::operands`] counts it as one:
/// reading how long something is is as much a reason to keep it alive as reading its elements.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Extent {
    /// A size the graph knew, which has to be a real one: the C API refuses a negative size for
    /// every call that takes a shape (`capi.cc`, `toShape`), so the `-1` that would have a
    /// reshape work a dimension out for itself is not reachable from here. A length the graph
    /// does not know is an [`Extent::Of`], which is better anyway -- it says *which* value the
    /// length comes from, where `-1` only says *the one left over*.
    At(i32),
    /// However long dimension `dim` of `value` is when the graph runs, `dim` negative to count
    /// from the back.
    Of { value: Value, dim: i32 },
    /// However many elements dimensions `from..to` of `value` cover between them, which is those
    /// sizes multiplied together.
    ///
    /// What a reshape that folds two dimensions into one needs, and the one piece of arithmetic
    /// an extent does: a transformer reading an image as a sequence wants `(N, C, H * W)`, and
    /// `H * W` is not a number either the graph or [`Extent::Of`] can produce. Dimensions count
    /// from the front here and the range is half open, the way a slice's is; a graph does not
    /// know its own rank, so counting a range from the back would name different dimensions for
    /// different inputs.
    Prod { value: Value, from: i32, to: i32 },
    /// The far end of the dimension, which only means anything as a slice bound: it is what
    /// [`Bound::End`] says, and it is here so that both ends of a slice are the one type.
    End,
}

impl Extent {
    /// However long dimension `dim` of `value` turns out to be.
    pub fn of(value: Value, dim: i32) -> Extent {
        Extent::Of { value, dim }
    }

    /// However many elements dimensions `from..to` of `value` turn out to cover.
    pub fn prod(value: Value, from: i32, to: i32) -> Extent {
        Extent::Prod { value, from, to }
    }

    /// The value this reads the length of, for a walker collecting the edges of a graph.
    pub fn value(self) -> Option<Value> {
        match self {
            Extent::Of { value, .. } | Extent::Prod { value, .. } => Some(value),
            Extent::At(_) | Extent::End => None,
        }
    }
}

impl From<i32> for Extent {
    fn from(size: i32) -> Extent {
        Extent::At(size)
    }
}

impl From<Bound> for Extent {
    fn from(bound: Bound) -> Extent {
        match bound {
            Bound::At(index) => Extent::At(index),
            Bound::End => Extent::End,
        }
    }
}

impl fmt::Display for Extent {
    /// An open end is written as nothing at all, the way a Rust range leaves it out.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Extent::At(size) => write!(f, "{size}"),
            Extent::Of { value, dim } => write!(f, "dim({value}, {dim})"),
            Extent::Prod { value, from, to } => write!(f, "dims({value}, {from}..{to})"),
            Extent::End => Ok(()),
        }
    }
}

/// A list of them, as a printed node writes it.
fn shape_of(extents: &[Extent]) -> String {
    let written: Vec<String> = extents.iter().map(Extent::to_string).collect();
    format!("[{}]", written.join(", "))
}

/// A pointwise operation of one operand, which keeps the shape it was handed.
///
/// [`Unary::Softmax`] is here despite reading a whole row, and the two gated units despite
/// halving the last dimension: what these have in common is being an activation applied to one
/// value with nothing to configure, which is what a graph needs to know about them.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Unary {
    Neg,
    Abs,
    Exp,
    Sqrt,
    Rsqrt,
    Square,
    Sigmoid,
    Tanh,
    Relu,
    Gelu,
    /// `x * sigmoid(1.702 * x)`, which OpenAI's CLIP uses in place of GELU.
    QuickGelu,
    Silu,
    /// Over the last dimension.
    Softmax,
    /// Halves the last dimension: `swish(x[..D / 2]) * x[D / 2..]`.
    Swiglu,
    /// The same gating with a GELU.
    Geglu,
}

impl Unary {
    /// What this is called in [`functional`](super::functional), which is also what a
    /// printed graph says.
    pub fn name(self) -> &'static str {
        match self {
            Unary::Neg => "neg",
            Unary::Abs => "abs",
            Unary::Exp => "exp",
            Unary::Sqrt => "sqrt",
            Unary::Rsqrt => "rsqrt",
            Unary::Square => "square",
            Unary::Sigmoid => "sigmoid",
            Unary::Tanh => "tanh",
            Unary::Relu => "relu",
            Unary::Gelu => "gelu",
            Unary::QuickGelu => "quick_gelu",
            Unary::Silu => "silu",
            Unary::Softmax => "softmax",
            Unary::Swiglu => "swiglu",
            Unary::Geglu => "geglu",
        }
    }
}

/// A pointwise operation of two operands, the second broadcast over the leading dimensions of the
/// first.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    /// Element-wise equality, which produces [`DType::Bool`] rather than what it was handed.
    Eq,
}

impl Binary {
    pub fn name(self) -> &'static str {
        match self {
            Binary::Add => "add",
            Binary::Sub => "sub",
            Binary::Mul => "mul",
            Binary::Div => "div",
            Binary::Eq => "eq",
        }
    }
}

/// A pointwise operation of one operand and one number the graph already knows.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Scalar {
    Mul,
    Div,
}

impl Scalar {
    pub fn name(self) -> &'static str {
        match self {
            Scalar::Mul => "mul_scalar",
            Scalar::Div => "div_scalar",
        }
    }
}

/// A reduction along one dimension, which the result drops.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Reduce {
    Sum,
    Max,
    Min,
}

impl Reduce {
    pub fn name(self) -> &'static str {
        match self {
            Reduce::Sum => "sum",
            Reduce::Max => "max",
            Reduce::Min => "min",
        }
    }
}

/// What one node of a graph computes, and what it computes it from.
///
/// The operands are in the variant rather than in a list beside it, so that an optional one --
/// the bias a convolution may not have -- is an [`Option`] and not a length to be interpreted.
/// [`Op::operands`] is the list, for a walker that wants the edges and not the arithmetic.
///
/// # What is not here
///
/// The operations that write into a tensor the caller already has: `copy`, `fill`,
/// `rotary_embedding`, `store_kv_cache`, `repetition_penalty`, and the paged attention that reads
/// what they wrote. A value in a graph is what a node produced and nothing else, and an operation
/// whose result is a change to somebody else's tensor has no way to say that here. They stay
/// eager calls.
#[derive(Clone, Debug)]
pub enum Op {
    /// A tensor the caller hands over when the graph is run.
    Input {
        name: String,
    },
    /// A weight, read out of the package the model was written to.
    ///
    /// `name` is the whole name the file knows it by -- what
    /// [`Graph::name_of`](super::Graph::name_of) made of it, so a layer that was built into a
    /// [`subgraph`](super::Graph::subgraph) says `"to_q.weight"` and this holds
    /// `"down1.attn.to_q.weight"`. `shape` is what the layer expects it to be, which is checked
    /// against what the file turns out to hold.
    ///
    /// What a load hands over is what was written: the element type the exporter chose, on the
    /// host the bytes were read into. Putting it where the pass runs is [`Op::ToDevice`] and
    /// [`Op::Cast`], which are nodes like any other and belong to whoever built the graph.
    Load {
        name: String,
        shape: Vec<i32>,
    },
    /// A tensor that was already made, built into the graph.
    ///
    /// The escape hatch, for a value that is neither an input nor a weight and that no operation
    /// here produces. It costs what the tensor costs -- a graph holding one keeps it alive, and
    /// stays on the thread and the device it was made for -- so a value the graph can compute is
    /// better computed.
    Constant {
        tensor: Tensor,
    },

    Zeros {
        shape: Vec<Extent>,
        dtype: DType,
        device: Device,
    },
    Arange {
        begin: i64,
        end: i64,
        step: i64,
        device: Device,
    },
    /// Uniform in `[0, 1)`.
    Rand {
        shape: Vec<Extent>,
        dtype: DType,
        device: Device,
    },
    /// Normal, mean 0 and variance 1.
    Randn {
        shape: Vec<Extent>,
        device: Device,
    },
    /// `-inf` where a position may not attend and `0` where it may.
    CausalMask {
        max_len: Extent,
        device: Device,
    },

    Unary {
        kind: Unary,
        input: Value,
    },
    Binary {
        kind: Binary,
        lhs: Value,
        rhs: Value,
    },
    Scalar {
        kind: Scalar,
        input: Value,
        other: f32,
    },
    /// `input % other`, for a [`DType::Long`] value. Apart from the operands being integers this
    /// belongs with [`Op::Scalar`].
    ModScalar {
        input: Value,
        other: i64,
    },
    Reduce {
        kind: Reduce,
        input: Value,
        dim: i32,
    },

    /// Zero mean and unit variance over the last dimension, then scale and shift.
    LayerNorm {
        input: Value,
        weight: Option<Value>,
        bias: Option<Value>,
        eps: f32,
    },
    /// The same without subtracting the mean, and with no shift.
    RmsNorm {
        input: Value,
        weight: Value,
        eps: f32,
    },
    /// Over each group of channels and the space it covers, for `input` `(N, C, H, W)`.
    GroupNorm {
        input: Value,
        weight: Option<Value>,
        bias: Option<Value>,
        groups: i32,
        eps: f32,
    },

    /// Matrix multiplication, batched over the leading dimensions.
    Matmul {
        lhs: Value,
        rhs: Value,
    },
    /// `lhs` times the transpose of a weight held in FP8, which is two values rather than one:
    /// `weight` is `<fp8e4m3>(rows, k)` and `channel_scale` is `<float>(rows)`, row `r` of the
    /// weight meaning `weight[r] * channel_scale[r]`.
    ///
    /// Two operands and not one composite, because two is what the package holds and two is what
    /// the operator takes. The gain is that a scale is then an ordinary weight under an ordinary
    /// name, reached by an ordinary [`Op::Load`]: everything that walks loads -- what
    /// [`resident`](super::resident) reads, the liveness a `free` comes from,
    /// [`Residency::LowVram`](super::Residency::LowVram) putting one weight at a time across the
    /// bus -- reaches the scale without being taught that it exists.
    ///
    /// No transpose comes before this the way one comes before a [`Matmul`](Op::Matmul): this
    /// reads the weight in the `(out, in)` a package stores it in.
    Fp8Matmul {
        lhs: Value,
        weight: Value,
        channel_scale: Value,
    },
    /// The rows of `table` named by `indices`.
    Lookup {
        table: Value,
        indices: Value,
    },
    Conv2d {
        input: Value,
        weight: Value,
        bias: Option<Value>,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    },
    /// Repeat each pixel `scale` times along both spatial axes.
    UpsampleNearest2d {
        input: Value,
        scale: i32,
    },
    /// Scaled dot product attention over `(N, nHead, L, D)` queries.
    Attention {
        query: Value,
        key: Value,
        value: Value,
        causal: bool,
    },
    /// Join two values along `dim`, which is the only dimension they may disagree on.
    Cat {
        lhs: Value,
        rhs: Value,
        dim: i32,
    },

    /// The same elements under a new shape.
    View {
        input: Value,
        shape: Vec<Extent>,
    },
    Transpose {
        input: Value,
        dim0: i32,
        dim1: i32,
    },
    /// The half-open range `[begin, end)` of `dim`.
    Slice {
        input: Value,
        dim: i32,
        begin: Extent,
        end: Extent,
    },
    /// One entry of the first dimension, dropping that dimension.
    Subtensor {
        input: Value,
        index: i32,
    },
    Unsqueeze {
        input: Value,
        dim: i32,
    },
    Squeeze {
        input: Value,
        dim: i32,
    },
    /// The same elements laid out next to each other, copying only if they are not already.
    Contiguous {
        input: Value,
    },
    Cast {
        input: Value,
        dtype: DType,
    },
    ToDevice {
        input: Value,
        device: Device,
    },
}

impl Op {
    /// What this is called, which is the name of the function in
    /// [`functional`](super::functional) that runs it.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Input { .. } => "input",
            Op::Load { .. } => "load",
            Op::Constant { .. } => "constant",
            Op::Zeros { .. } => "zeros",
            Op::Arange { .. } => "arange",
            Op::Rand { .. } => "rand",
            Op::Randn { .. } => "randn",
            Op::CausalMask { .. } => "causal_mask",
            Op::Unary { kind, .. } => kind.name(),
            Op::Binary { kind, .. } => kind.name(),
            Op::Scalar { kind, .. } => kind.name(),
            Op::ModScalar { .. } => "mod_scalar",
            Op::Reduce { kind, .. } => kind.name(),
            Op::LayerNorm { .. } => "layer_norm",
            Op::RmsNorm { .. } => "rms_norm",
            Op::GroupNorm { .. } => "group_norm",
            Op::Matmul { .. } => "matmul",
            Op::Fp8Matmul { .. } => "fp8_matmul",
            Op::Lookup { .. } => "lookup",
            Op::Conv2d { .. } => "conv2d",
            Op::UpsampleNearest2d { .. } => "upsample_nearest2d",
            Op::Attention { .. } => "attention",
            Op::Cat { .. } => "cat",
            Op::View { .. } => "view",
            Op::Transpose { .. } => "transpose",
            Op::Slice { .. } => "slice",
            Op::Subtensor { .. } => "subtensor",
            Op::Unsqueeze { .. } => "unsqueeze",
            Op::Squeeze { .. } => "squeeze",
            Op::Contiguous { .. } => "contiguous",
            Op::Cast { .. } => "cast",
            Op::ToDevice { .. } => "to_device",
        }
    }

    /// The values this node reads: the ones it takes as data, and then the ones it reads only
    /// the length of.
    ///
    /// These are the edges of the graph -- everything a walker needs to know what has to be
    /// computed before this node can be, without knowing what any of it computes -- which is why
    /// an [`Extent::Of`] is in here. A node that reshapes to however long something else is
    /// cannot run before that something else, and cannot run after it has been let go of.
    pub fn operands(&self) -> Vec<Value> {
        let mut values = self.data_operands();
        values.extend(self.extents().into_iter().filter_map(Extent::value));

        values
    }

    /// The sizes this node was written with, which may be sizes of other values.
    pub fn extents(&self) -> Vec<Extent> {
        match self {
            Op::Zeros { shape, .. } | Op::Rand { shape, .. } | Op::Randn { shape, .. } => {
                shape.clone()
            }
            Op::View { shape, .. } => shape.clone(),
            Op::CausalMask { max_len, .. } => vec![*max_len],
            Op::Slice { begin, end, .. } => vec![*begin, *end],
            _ => Vec::new(),
        }
    }

    /// The values the operation takes as data, in the order it takes them, with an optional
    /// operand that is not there simply left out. What a printed node writes between its
    /// brackets, before the settings.
    fn data_operands(&self) -> Vec<Value> {
        match self {
            Op::Input { .. }
            | Op::Load { .. }
            | Op::Constant { .. }
            | Op::Zeros { .. }
            | Op::Arange { .. }
            | Op::Rand { .. }
            | Op::Randn { .. }
            | Op::CausalMask { .. } => Vec::new(),

            Op::Unary { input, .. }
            | Op::Scalar { input, .. }
            | Op::ModScalar { input, .. }
            | Op::Reduce { input, .. }
            | Op::UpsampleNearest2d { input, .. }
            | Op::View { input, .. }
            | Op::Transpose { input, .. }
            | Op::Slice { input, .. }
            | Op::Subtensor { input, .. }
            | Op::Unsqueeze { input, .. }
            | Op::Squeeze { input, .. }
            | Op::Contiguous { input }
            | Op::Cast { input, .. }
            | Op::ToDevice { input, .. } => vec![*input],

            Op::Binary { lhs, rhs, .. } | Op::Matmul { lhs, rhs } | Op::Cat { lhs, rhs, .. } => {
                vec![*lhs, *rhs]
            }
            Op::Lookup { table, indices } => vec![*table, *indices],
            Op::Fp8Matmul {
                lhs,
                weight,
                channel_scale,
            } => vec![*lhs, *weight, *channel_scale],
            Op::RmsNorm { input, weight, .. } => vec![*input, *weight],

            Op::LayerNorm {
                input,
                weight,
                bias,
                ..
            }
            | Op::GroupNorm {
                input,
                weight,
                bias,
                ..
            } => {
                let mut operands = vec![*input];
                operands.extend(weight.iter().chain(bias.iter()));
                operands
            }

            Op::Conv2d {
                input,
                weight,
                bias,
                ..
            } => {
                let mut operands = vec![*input, *weight];
                operands.extend(bias.iter());
                operands
            }

            Op::Attention {
                query, key, value, ..
            } => vec![*query, *key, *value],
        }
    }
}

impl fmt::Display for Op {
    /// The operation as a call: its name, the values it reads, and then the settings that are not
    /// values, so that a printed graph reads as the source it stands for.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut call = Call::new(f, self.name())?;
        for operand in self.data_operands() {
            call.arg(operand)?;
        }

        match self {
            Op::Input { name } => call.arg(format_args!("{name:?}"))?,
            Op::Load { name, shape } => {
                call.arg(format_args!("{name:?}"))?;
                call.arg(format_args!("{shape:?}"))?;
            }
            Op::Constant { tensor } => {
                call.arg(format_args!("{:?}{:?}", tensor.dtype(), tensor.shape()))?
            }

            Op::Zeros {
                shape,
                dtype,
                device,
            }
            | Op::Rand {
                shape,
                dtype,
                device,
            } => {
                call.arg(shape_of(shape))?;
                call.arg(format_args!("{dtype:?}"))?;
                call.arg(device.name())?;
            }
            Op::Randn { shape, device } => {
                call.arg(shape_of(shape))?;
                call.arg(device.name())?;
            }
            Op::Arange {
                begin,
                end,
                step,
                device,
            } => {
                call.arg(format_args!("{begin}..{end} by {step}"))?;
                call.arg(device.name())?;
            }
            Op::CausalMask { max_len, device } => {
                call.arg(max_len)?;
                call.arg(device.name())?;
            }

            Op::Scalar { other, .. } => call.arg(other)?,
            Op::ModScalar { other, .. } => call.arg(other)?,
            Op::Reduce { dim, .. } | Op::Cat { dim, .. } => call.arg(format_args!("dim={dim}"))?,

            Op::LayerNorm { eps, .. } | Op::RmsNorm { eps, .. } => {
                call.arg(format_args!("eps={eps}"))?
            }
            Op::GroupNorm { groups, eps, .. } => {
                call.arg(format_args!("groups={groups}"))?;
                call.arg(format_args!("eps={eps}"))?;
            }

            Op::Conv2d {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => {
                call.arg(format_args!("stride={stride}"))?;
                call.arg(format_args!("padding={padding}"))?;
                call.arg(format_args!("dilation={dilation}"))?;
                call.arg(format_args!("groups={groups}"))?;
            }
            Op::UpsampleNearest2d { scale, .. } => call.arg(format_args!("scale={scale}"))?,
            Op::Attention { causal, .. } => call.arg(format_args!("causal={causal}"))?,

            Op::View { shape, .. } => call.arg(shape_of(shape))?,
            Op::Transpose { dim0, dim1, .. } => {
                call.arg(dim0)?;
                call.arg(dim1)?;
            }
            Op::Slice {
                dim, begin, end, ..
            } => {
                call.arg(format_args!("dim={dim}"))?;
                call.arg(format_args!("{begin}..{end}"))?;
            }
            Op::Subtensor { index, .. } => call.arg(index)?,
            Op::Unsqueeze { dim, .. } | Op::Squeeze { dim, .. } => call.arg(dim)?,
            Op::Cast { dtype, .. } => call.arg(format_args!("{dtype:?}"))?,
            Op::ToDevice { device, .. } => call.arg(device.name())?,

            // Everything the operands already said in full.
            Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Matmul { .. }
            | Op::Fp8Matmul { .. }
            | Op::Lookup { .. }
            | Op::Contiguous { .. } => (),
        }

        call.finish()
    }
}

/// Writes `name(a, b, c)` while the caller is still deciding what the arguments are.
///
/// Which is what printing an operation needs: the operands come from one place and the settings
/// from another, and neither knows whether it is writing the first argument or the fourth.
struct Call<'a, 'b> {
    out: &'a mut fmt::Formatter<'b>,
    written: bool,
}

impl<'a, 'b> Call<'a, 'b> {
    fn new(out: &'a mut fmt::Formatter<'b>, name: &str) -> Result<Call<'a, 'b>, fmt::Error> {
        write!(out, "{name}(")?;
        Ok(Call {
            out,
            written: false,
        })
    }

    fn arg(&mut self, value: impl fmt::Display) -> fmt::Result {
        if self.written {
            self.out.write_str(", ")?;
        }
        self.written = true;

        write!(self.out, "{value}")
    }

    fn finish(self) -> fmt::Result {
        self.out.write_str(")")
    }
}
