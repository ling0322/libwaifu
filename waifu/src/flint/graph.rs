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

//! The graph itself, and the one function per operation that puts a node into it.

use std::cell::RefCell;
use std::fmt;
use std::panic::Location;
use std::rc::Rc;

use super::{DType, Device, Tensor};

use super::op::{Binary, Extent, Op, Reduce, Scalar, Unary, Value};

/// A forward pass written down rather than run.
///
/// Every method that names an operation adds a node and hands back the [`Value`] it produces, so
/// building a graph looks like the arithmetic it stands for and nothing is computed while it is
/// being written:
///
/// ```
/// use waifu::flint::Graph;
///
/// let g = Graph::new();
/// let x = g.input("hidden");
/// let weight = g.input("weight");
///
/// let transposed = g.transpose(weight, 0, 1);
/// let logits = g.matmul(x, transposed);
/// let logits = g.silu(logits);
/// g.output("logits", logits);
///
/// assert_eq!(g.len(), 5);
/// ```
///
/// # What it holds
///
/// Nodes, in the order they were added. An operand is a value that already exists -- there is no
/// way to name one that does not -- so a graph cannot hold a cycle, and walking the nodes from
/// the front is a topological order without anything having to sort them.
///
/// A leaf that comes from outside is one of three things. [`Graph::input`] is a tensor the caller
/// hands over when the graph runs, named so that it can; [`Graph::load`] is a weight, named as the
/// model's package knows it; and [`Graph::constant`] is a tensor built into the graph, for the
/// value that is neither. The rest of the leaves --
/// [`Graph::zeros`], [`Graph::arange`], [`Graph::randn`] and the two beside them -- are made out
/// of nothing when the graph runs, and are nodes like any other.
///
/// # Namespaces
///
/// A weight is named the way the package names it, which is a path: `down1.attn.to_q.weight`. No
/// layer says the whole of that -- an attention block knows it wants `to_q.weight` and does not
/// know it is the first attention of the first down block. So a graph carries a namespace, and
/// [`Graph::subgraph`] goes one level into it for as long as what it hands back is held:
///
/// ```
/// use waifu::flint::Graph;
///
/// let g = Graph::new();
/// let attn = g.subgraph("down1").subgraph("attn");
///
/// let query = attn.load("to_q.weight", &[640, 640]);
/// assert_eq!(attn.name_of("to_q.weight"), "down1.attn.to_q.weight");
///
/// // The one graph, whichever handle put the node into it.
/// assert_eq!(g.len(), 1);
/// assert_eq!(g.name(), "");
/// ```
///
/// What [`Graph::subgraph`] hands back is a `Graph`: the same nodes, the same values, the same
/// everything, with one name in front of what [`Graph::load`] is asked for. Which is the whole of
/// how a weight comes to be named: a layer asks for `"weight"`, and where it was written into the
/// graph decides that this is `down1.attn.to_q.weight`.
///
/// The namespace says where a weight is read from and nothing else. [`Graph::input`] and
/// [`Graph::output`] are what the caller of the whole graph hands over and takes back, so they
/// are named where the graph is run and not where the node happens to be built.
///
/// # Holding one
///
/// A `Graph` is a handle on the nodes rather than the nodes themselves, so every method that adds
/// one takes `&self`: a handle is not exclusive and could not promise to be, since
/// [`Graph::subgraph`] hands out another one over the same nodes. [`Graph::clone`] makes a third,
/// and what it costs is a pointer and a name.
///
/// Which is why reading a graph back hands over what it read rather than a reference into it.
/// [`Graph::op`] gives an [`Op`] and [`Graph::nodes`] a list, both copies: a borrow would have to
/// keep the nodes borrowed for as long as the caller held it, and the next node added underneath
/// would find them already borrowed. What it costs is a copy of one node, which is a handful of
/// numbers; what it buys is that a graph can be read while it is still being written.
///
/// # Where a node came from
///
/// Every method that adds a node is `#[track_caller]`, so a node also records where it came
/// from: the line that built it, and the namespace of the handle it was built through.
/// [`Graph::site`] hands back both as a [`Site`], and `{:#}` prints it beside the node.
///
/// This is what a graph costs to debug, bought back. An operator that is handed something it
/// cannot work with does not return an error: it prints and aborts, which is
/// [`functional`](super::functional)'s first warning about itself. In an eager pass the
/// stack trace that comes with it points at the line that made the mistake. Here it points at
/// whatever loop is running the graph, which is the same line for all ten thousand nodes, and
/// the code that built the node ran long ago and is not on the stack at all. The site is what
/// is left to point back at it, and it costs nothing at run time.
///
/// The namespace is the other half of the same question. A line is where the code is, and one
/// line of a layer builds every copy of that layer in the model: `Linear::graph`'s matmul is one
/// line and one node in each of the model's attentions. The namespace is which of them this is
/// -- `down1.attn.to_q` -- which is what names the weights beside it and what a reader of the
/// model is looking for.
///
/// # What it does not do
///
/// It does not compute, and it does not check shapes. Nothing here holds a shape at all: an
/// operand is a node, not a tensor, and what any of it is the shape of is not known until
/// something runs it against real inputs. So a graph that is put together wrongly is found out
/// where it is run, by the same fatal check that finds an eager call put together wrongly. What
/// this does guarantee is the structure -- every operand exists, every node comes after the ones
/// it reads -- which is what a runner needs to not have to check.
#[derive(Clone, Default)]
pub struct Graph {
    /// The nodes, shared by every handle on this graph. See the type documentation.
    body: Rc<RefCell<Body>>,
    /// What [`Graph::load`] puts in front of the name it is given, which is the one thing a
    /// handle has of its own. Empty at the root and one level deeper in every
    /// [`Graph::subgraph`].
    namespace: String,
    /// How the package holds the matrices this graph multiplies by, which decides what a
    /// projection is built out of. See [`WeightFormat`].
    weight_format: WeightFormat,
}

/// How a package stored the matrices a model multiplies by.
///
/// Not a property of the model: a projection is the same projection either way, the weight has the
/// same name and the same shape, and the pass computes the same thing to within the quantization
/// error. What changes is which nodes read it -- so this is carried on the [`Graph`] rather than
/// passed to every layer, reaching each of them the same way the graph itself does.
///
/// It comes from the model's configuration, because that is where a package says what is in it. A
/// graph could instead go looking for the scales beside a weight and decide from that, and it
/// deliberately does not: then what a model *is* would depend on what a reader found, two packages
/// with the same configuration could compile to different passes, and a package that lost half its
/// scales would quietly build a mixture rather than fail.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WeightFormat {
    /// As the exporter wrote them, which is what every package so far holds: float16 for most
    /// models, float32 where one needs the width.
    #[default]
    Float,
    /// Quantized to E4M3 with one `<float>` scale per output channel, stored as two tensors --
    /// `"…weight"` and `"…weight.scale"`. Half the bytes on the device, about 2.6e-2 of relative
    /// error, and `docs/fp8.md` for the whole of it.
    Fp8,
}

impl WeightFormat {
    /// What a model's configuration calls this, or `None` for a name no configuration may use.
    pub fn from_name(name: &str) -> Option<WeightFormat> {
        match name {
            "float" => Some(WeightFormat::Float),
            "fp8" => Some(WeightFormat::Fp8),
            _ => None,
        }
    }

    /// The name a configuration writes it under.
    pub fn name(&self) -> &'static str {
        match self {
            WeightFormat::Float => "float",
            WeightFormat::Fp8 => "fp8",
        }
    }
}

/// What a graph actually holds, which no handle holds a piece of on its own.
#[derive(Debug, Default)]
struct Body {
    nodes: Vec<Op>,
    /// Where each node was built, in step with `nodes`. See [`Graph`] for why a graph keeps this.
    sites: Vec<Site>,
    inputs: Vec<(String, Value)>,
    outputs: Vec<(String, Value)>,
}

/// Where a node came from: the line that built it, and the namespace it was built in.
///
/// Two halves of one question, and neither answers it alone. The line says which code asked for
/// the node, but one line of a layer builds that layer everywhere it appears in the model; the
/// namespace says which appearance this is, but a namespace is a whole block's worth of nodes.
/// Together they are the node.
///
/// [`Graph::site`] gives one out rather than lending it, the way everything else read back out of
/// a graph is given out. See [`Graph`] for why a graph keeps this at all.
#[derive(Clone, Debug)]
pub struct Site {
    location: &'static Location<'static>,
    namespace: String,
}

impl Site {
    /// The line that built the node.
    pub fn location(&self) -> &'static Location<'static> {
        self.location
    }

    /// The namespace the node was built in, which is the one weights under it are named from.
    /// Empty for a node built through the root handle.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }
}

impl fmt::Display for Site {
    /// `waifu/src/sdxl/unet.rs:120:9 in down1.attn`, and the location alone at the root, where
    /// there is no namespace to name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.location)?;
        if !self.namespace.is_empty() {
            write!(f, " in {}", self.namespace)?;
        }

        Ok(())
    }
}

impl Graph {
    /// A graph written from a configuration alone, which is what almost all of one is.
    pub fn new() -> Graph {
        Graph::default()
    }

    /// A graph written for a package that holds its matrices in `weight_format`.
    ///
    /// The model's configuration is where that comes from; see [`WeightFormat`] for why it is
    /// carried here rather than handed to every layer, and why it is read rather than guessed.
    pub fn with_weights(weight_format: WeightFormat) -> Graph {
        Graph {
            weight_format,
            ..Graph::default()
        }
    }

    /// How the package this graph is being written for holds its matrices.
    pub fn weight_format(&self) -> WeightFormat {
        self.weight_format
    }

    /// The same graph, with weights read out of the `name` sub-namespace of this one.
    ///
    /// What comes back is another handle on these nodes, not a graph of its own: a node added
    /// through it is added here, and a value it hands out is one of this graph's. The only
    /// difference is the name [`Graph::load`] makes of what it is asked for, which is what a
    /// layer hands the layers underneath it.
    ///
    /// Nothing has to be given back afterwards. This handle's namespace is unchanged -- the two
    /// are separate handles, and a layer holding one cannot have its name moved by a layer
    /// holding another.
    pub fn subgraph(&self, name: &str) -> Graph {
        Graph {
            body: Rc::clone(&self.body),
            namespace: self.name_of(name),
            weight_format: self.weight_format,
        }
    }

    /// The whole name of `name` in this namespace, which is what the package knows it by.
    ///
    /// An empty `name` is this namespace itself, so `subgraph("")` is the level it was taken
    /// from. Which is what a caller narrowing a graph to a namespace it was handed needs, since
    /// the namespace it was handed may be the root.
    pub fn name_of(&self, name: &str) -> String {
        match (self.namespace.as_str(), name) {
            ("", _) => name.to_string(),
            (namespace, "") => namespace.to_string(),
            (namespace, name) => format!("{namespace}.{name}"),
        }
    }

    /// The whole name of the namespace weights are being read from, which is what an error about
    /// it should say.
    pub fn name(&self) -> &str {
        &self.namespace
    }

    /// Put `op` at the end of the graph and hand back the value it produces.
    ///
    /// Every method below is one of these; this is the one to reach for when the operation is
    /// already an [`Op`] in hand, and it is the only place a node is made.
    ///
    /// # Panics
    ///
    /// If `op` reads a value this graph did not make. A [`Value`] belongs to one graph, and one
    /// from somewhere else names a node here by accident or names nothing at all. Neither is a
    /// thing to carry on with.
    #[track_caller]
    pub fn push(&self, op: Op) -> Value {
        let mut body = self.body.borrow_mut();
        for operand in op.operands() {
            assert!(
                operand.slot() < body.nodes.len(),
                "{operand} is not a value of this graph, which has {} of them",
                body.nodes.len()
            );
        }

        body.nodes.push(op);
        body.sites.push(Site {
            location: Location::caller(),
            namespace: self.namespace.clone(),
        });

        Value::at(body.nodes.len() - 1)
    }

    /// A tensor the caller hands over when the graph is run, under `name`.
    ///
    /// Named as the caller of the whole graph knows it, not as this namespace would have it. See
    /// the type documentation.
    #[track_caller]
    pub fn input(&self, name: &str) -> Value {
        let value = self.push(Op::Input {
            name: name.to_string(),
        });
        self.body
            .borrow_mut()
            .inputs
            .push((name.to_string(), value));

        value
    }

    /// A weight of `shape`, read from the model's package under `name` in this namespace.
    #[track_caller]
    pub fn load(&self, name: &str, shape: &[i32]) -> Value {
        self.push(Op::Load {
            name: self.name_of(name),
            shape: shape.to_vec(),
        })
    }

    /// A tensor built into the graph. See [`Op::Constant`] for what it costs.
    #[track_caller]
    pub fn constant(&self, tensor: Tensor) -> Value {
        self.push(Op::Constant { tensor })
    }

    /// Say that `value` is one of the things this graph is for, under `name`.
    ///
    /// A graph may have several: a text encoder produces a hidden state and a pooled vector, and
    /// naming them is how whoever runs it tells the two apart.
    #[track_caller]
    pub fn output(&self, name: &str, value: Value) {
        self.expect(value);
        self.body
            .borrow_mut()
            .outputs
            .push((name.to_string(), value));
    }

    // Leaves that are computed rather than handed over.

    /// A tensor of zeros.
    ///
    /// `shape` takes plain numbers -- `g.zeros([1, 4, 64, 64], ..)` -- or [`Extent`]s where a
    /// size is not one the graph knows. See [`Extent`] for what the second kind is for.
    #[track_caller]
    pub fn zeros(
        &self,
        shape: impl IntoIterator<Item = impl Into<Extent>>,
        dtype: DType,
        device: Device,
    ) -> Value {
        self.push(Op::Zeros {
            shape: extents(shape),
            dtype,
            device,
        })
    }

    #[track_caller]
    pub fn arange(&self, begin: i64, end: i64, step: i64, device: Device) -> Value {
        self.push(Op::Arange {
            begin,
            end,
            step,
            device,
        })
    }

    #[track_caller]
    pub fn rand(
        &self,
        shape: impl IntoIterator<Item = impl Into<Extent>>,
        dtype: DType,
        device: Device,
    ) -> Value {
        self.push(Op::Rand {
            shape: extents(shape),
            dtype,
            device,
        })
    }

    #[track_caller]
    pub fn randn(
        &self,
        shape: impl IntoIterator<Item = impl Into<Extent>>,
        device: Device,
    ) -> Value {
        self.push(Op::Randn {
            shape: extents(shape),
            device,
        })
    }

    #[track_caller]
    pub fn causal_mask(&self, max_len: impl Into<Extent>, device: Device) -> Value {
        self.push(Op::CausalMask {
            max_len: max_len.into(),
            device,
        })
    }

    // The pointwise families. One method each, since `g.silu(x)` is what the eager code says and
    // what a reader of the graph is looking for.

    #[track_caller]
    pub fn unary(&self, kind: Unary, input: Value) -> Value {
        self.push(Op::Unary { kind, input })
    }

    #[track_caller]
    pub fn binary(&self, kind: Binary, lhs: Value, rhs: Value) -> Value {
        self.push(Op::Binary { kind, lhs, rhs })
    }

    #[track_caller]
    pub fn add(&self, lhs: Value, rhs: Value) -> Value {
        self.binary(Binary::Add, lhs, rhs)
    }

    #[track_caller]
    pub fn sub(&self, lhs: Value, rhs: Value) -> Value {
        self.binary(Binary::Sub, lhs, rhs)
    }

    #[track_caller]
    pub fn mul(&self, lhs: Value, rhs: Value) -> Value {
        self.binary(Binary::Mul, lhs, rhs)
    }

    #[track_caller]
    pub fn div(&self, lhs: Value, rhs: Value) -> Value {
        self.binary(Binary::Div, lhs, rhs)
    }

    #[track_caller]
    pub fn eq(&self, lhs: Value, rhs: Value) -> Value {
        self.binary(Binary::Eq, lhs, rhs)
    }

    #[track_caller]
    pub fn neg(&self, input: Value) -> Value {
        self.unary(Unary::Neg, input)
    }

    #[track_caller]
    pub fn abs(&self, input: Value) -> Value {
        self.unary(Unary::Abs, input)
    }

    #[track_caller]
    pub fn exp(&self, input: Value) -> Value {
        self.unary(Unary::Exp, input)
    }

    #[track_caller]
    pub fn sqrt(&self, input: Value) -> Value {
        self.unary(Unary::Sqrt, input)
    }

    #[track_caller]
    pub fn rsqrt(&self, input: Value) -> Value {
        self.unary(Unary::Rsqrt, input)
    }

    #[track_caller]
    pub fn square(&self, input: Value) -> Value {
        self.unary(Unary::Square, input)
    }

    #[track_caller]
    pub fn sigmoid(&self, input: Value) -> Value {
        self.unary(Unary::Sigmoid, input)
    }

    #[track_caller]
    pub fn tanh(&self, input: Value) -> Value {
        self.unary(Unary::Tanh, input)
    }

    #[track_caller]
    pub fn relu(&self, input: Value) -> Value {
        self.unary(Unary::Relu, input)
    }

    #[track_caller]
    pub fn gelu(&self, input: Value) -> Value {
        self.unary(Unary::Gelu, input)
    }

    #[track_caller]
    pub fn quick_gelu(&self, input: Value) -> Value {
        self.unary(Unary::QuickGelu, input)
    }

    #[track_caller]
    pub fn silu(&self, input: Value) -> Value {
        self.unary(Unary::Silu, input)
    }

    #[track_caller]
    pub fn softmax(&self, input: Value) -> Value {
        self.unary(Unary::Softmax, input)
    }

    #[track_caller]
    pub fn swiglu(&self, input: Value) -> Value {
        self.unary(Unary::Swiglu, input)
    }

    #[track_caller]
    pub fn geglu(&self, input: Value) -> Value {
        self.unary(Unary::Geglu, input)
    }

    #[track_caller]
    pub fn mul_scalar(&self, input: Value, other: f32) -> Value {
        self.push(Op::Scalar {
            kind: Scalar::Mul,
            input,
            other,
        })
    }

    #[track_caller]
    pub fn div_scalar(&self, input: Value, other: f32) -> Value {
        self.push(Op::Scalar {
            kind: Scalar::Div,
            input,
            other,
        })
    }

    #[track_caller]
    pub fn mod_scalar(&self, input: Value, other: i64) -> Value {
        self.push(Op::ModScalar { input, other })
    }

    // Reductions. `dim` may be negative to count from the back, and the result drops it.

    #[track_caller]
    pub fn reduce(&self, kind: Reduce, input: Value, dim: i32) -> Value {
        self.push(Op::Reduce { kind, input, dim })
    }

    #[track_caller]
    pub fn sum(&self, input: Value, dim: i32) -> Value {
        self.reduce(Reduce::Sum, input, dim)
    }

    #[track_caller]
    pub fn max(&self, input: Value, dim: i32) -> Value {
        self.reduce(Reduce::Max, input, dim)
    }

    #[track_caller]
    pub fn min(&self, input: Value, dim: i32) -> Value {
        self.reduce(Reduce::Min, input, dim)
    }

    // Normalizations.

    #[track_caller]
    pub fn layer_norm(
        &self,
        input: Value,
        weight: Option<Value>,
        bias: Option<Value>,
        eps: f32,
    ) -> Value {
        self.push(Op::LayerNorm {
            input,
            weight,
            bias,
            eps,
        })
    }

    #[track_caller]
    pub fn rms_norm(&self, input: Value, weight: Value, eps: f32) -> Value {
        self.push(Op::RmsNorm { input, weight, eps })
    }

    #[track_caller]
    pub fn group_norm(
        &self,
        input: Value,
        weight: Option<Value>,
        bias: Option<Value>,
        groups: i32,
        eps: f32,
    ) -> Value {
        self.push(Op::GroupNorm {
            input,
            weight,
            bias,
            groups,
            eps,
        })
    }

    // The operations that do the work.

    #[track_caller]
    pub fn matmul(&self, lhs: Value, rhs: Value) -> Value {
        self.push(Op::Matmul { lhs, rhs })
    }

    /// `lhs` times the transpose of an FP8 weight, which is the two values it is stored as: the
    /// `<fp8e4m3>(rows, k)` elements and the `<float>(rows)` scale. See [`Op::Fp8Matmul`].
    #[track_caller]
    pub fn fp8_matmul(&self, lhs: Value, weight: Value, channel_scale: Value) -> Value {
        self.push(Op::Fp8Matmul {
            lhs,
            weight,
            channel_scale,
        })
    }

    #[track_caller]
    pub fn lookup(&self, table: Value, indices: Value) -> Value {
        self.push(Op::Lookup { table, indices })
    }

    #[track_caller]
    pub fn conv2d(
        &self,
        input: Value,
        weight: Value,
        bias: Option<Value>,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Value {
        self.push(Op::Conv2d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        })
    }

    #[track_caller]
    pub fn upsample_nearest2d(&self, input: Value, scale: i32) -> Value {
        self.push(Op::UpsampleNearest2d { input, scale })
    }

    #[track_caller]
    pub fn attention(&self, query: Value, key: Value, value: Value, causal: bool) -> Value {
        self.push(Op::Attention {
            query,
            key,
            value,
            causal,
        })
    }

    #[track_caller]
    pub fn cat(&self, lhs: Value, rhs: Value, dim: i32) -> Value {
        self.push(Op::Cat { lhs, rhs, dim })
    }

    // Shapes. These are views in an eager pass and nodes here: a graph says what is read, and
    // whether reading it copies anything is the runner's business.

    /// The same elements under a new shape, which may be one the graph does not know all of.
    #[track_caller]
    pub fn view(&self, input: Value, shape: impl IntoIterator<Item = impl Into<Extent>>) -> Value {
        self.push(Op::View {
            input,
            shape: extents(shape),
        })
    }

    #[track_caller]
    pub fn transpose(&self, input: Value, dim0: i32, dim1: i32) -> Value {
        self.push(Op::Transpose { input, dim0, dim1 })
    }

    /// The half-open range `[begin, end)` of `dim`.
    ///
    /// Both bounds take a plain `i32`, negative to count from the back, [`Bound::End`](super::Bound::End) to leave
    /// that side alone, or an [`Extent`] where the graph does not know where the range stops --
    /// `g.slice(table, 0, 0, Extent::of(ids, 0))` is the first as-many-rows-as-there-are-ids of
    /// a table.
    #[track_caller]
    pub fn slice(
        &self,
        input: Value,
        dim: i32,
        begin: impl Into<Extent>,
        end: impl Into<Extent>,
    ) -> Value {
        self.push(Op::Slice {
            input,
            dim,
            begin: begin.into(),
            end: end.into(),
        })
    }

    #[track_caller]
    pub fn subtensor(&self, input: Value, index: i32) -> Value {
        self.push(Op::Subtensor { input, index })
    }

    #[track_caller]
    pub fn unsqueeze(&self, input: Value, dim: i32) -> Value {
        self.push(Op::Unsqueeze { input, dim })
    }

    #[track_caller]
    pub fn squeeze(&self, input: Value, dim: i32) -> Value {
        self.push(Op::Squeeze { input, dim })
    }

    #[track_caller]
    pub fn contiguous(&self, input: Value) -> Value {
        self.push(Op::Contiguous { input })
    }

    #[track_caller]
    pub fn cast(&self, input: Value, dtype: DType) -> Value {
        self.push(Op::Cast { input, dtype })
    }

    #[track_caller]
    pub fn to_device(&self, input: Value, device: Device) -> Value {
        self.push(Op::ToDevice { input, device })
    }

    // Reading a graph back. Every one of these hands over what it read rather than a reference
    // into the nodes, for the reason the type documentation gives.

    /// How many nodes are here.
    pub fn len(&self) -> usize {
        self.body.borrow().nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What produces `value`.
    ///
    /// # Panics
    ///
    /// If `value` is not one of this graph's, for the reason [`Graph::push`] gives. Use
    /// [`Graph::get`] where that is a question rather than a mistake.
    #[track_caller]
    pub fn op(&self, value: Value) -> Op {
        self.expect(value);
        self.body.borrow().nodes[value.slot()].clone()
    }

    /// The same, for a caller that is asking rather than reading.
    pub fn get(&self, value: Value) -> Option<Op> {
        self.body.borrow().nodes.get(value.slot()).cloned()
    }

    /// Where the node producing `value` came from.
    ///
    /// What an error about a node should say, since the node itself only says what it computes
    /// and every node of a model says much the same thing. See the type documentation for why a
    /// graph is worth this and an eager pass is not.
    ///
    /// # Panics
    ///
    /// If `value` is not one of this graph's, the way [`Graph::op`] does.
    #[track_caller]
    pub fn site(&self, value: Value) -> Site {
        self.expect(value);
        self.body.borrow().sites[value.slot()].clone()
    }

    /// Every node, paired with the value it produces, in an order where a node comes after
    /// everything it reads. Which is to say: the order to run them in.
    ///
    /// The whole of the graph at once, which is what something compiling one wants and what it
    /// would otherwise have to ask for a node at a time.
    pub fn nodes(&self) -> Vec<(Value, Op)> {
        self.body
            .borrow()
            .nodes
            .iter()
            .enumerate()
            .map(|(index, op)| (Value::at(index), op.clone()))
            .collect()
    }

    /// What the caller has to hand over, in the order it was asked for.
    pub fn inputs(&self) -> Vec<(String, Value)> {
        self.body.borrow().inputs.clone()
    }

    /// What the graph is for, in the order it was named.
    pub fn outputs(&self) -> Vec<(String, Value)> {
        self.body.borrow().outputs.clone()
    }

    /// The input called `name`, if the graph asked for one.
    pub fn input_of(&self, name: &str) -> Option<Value> {
        find(&self.body.borrow().inputs, name)
    }

    /// The output called `name`, if the graph produces one.
    pub fn output_of(&self, name: &str) -> Option<Value> {
        find(&self.body.borrow().outputs, name)
    }

    /// Every weight the graph reads, by whole name, in the order it first reads one and each
    /// named once.
    ///
    /// What a graph asks of a package, which is worth being able to ask before a run: a name that
    /// is not there is otherwise found in the middle of a pass, one node at a time.
    pub fn parameters(&self) -> Vec<String> {
        let body = self.body.borrow();

        let mut names: Vec<String> = Vec::new();
        for op in &body.nodes {
            if let Op::Load { name, .. } = op {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
        }

        names
    }

    /// How many times each value is read, in step with [`Graph::nodes`] and counting each naming
    /// as an output as a read.
    ///
    /// What a runner counts down to know when a tensor is finished with: a value nothing has left
    /// to read is one whose memory can go back. A value at zero was computed for nothing, which is
    /// worth knowing too.
    pub fn use_counts(&self) -> Vec<usize> {
        let body = self.body.borrow();

        let mut counts = vec![0; body.nodes.len()];
        for op in &body.nodes {
            for operand in op.operands() {
                counts[operand.slot()] += 1;
            }
        }
        for (_, value) in &body.outputs {
            counts[value.slot()] += 1;
        }

        counts
    }

    /// Panics unless `value` is one of this graph's.
    #[track_caller]
    fn expect(&self, value: Value) {
        let held = self.body.borrow().nodes.len();
        assert!(
            value.slot() < held,
            "{value} is not a value of this graph, which has {held} of them"
        );
    }
}

/// A written-down shape as the graph holds it.
fn extents(shape: impl IntoIterator<Item = impl Into<Extent>>) -> Vec<Extent> {
    shape.into_iter().map(Into::into).collect()
}

/// The value a list of named ones calls `name`.
fn find(named: &[(String, Value)], name: &str) -> Option<Value> {
    named
        .iter()
        .find(|(other, _)| other == name)
        .map(|(_, value)| *value)
}

impl fmt::Debug for Graph {
    /// How much is here and which namespace this handle names, rather than what is here: a graph
    /// is thousands of nodes, and [`Display`](fmt::Display) is what writes them out.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Graph({:?}, {} nodes)", self.namespace, self.len())
    }
}

impl fmt::Display for Graph {
    /// The graph as the pass it stands for, one node per line, which is the only way to read one
    /// that has not been run.
    ///
    /// The alternate form, `{:#}`, writes where each node was built after it. That is the form to
    /// print when something has gone wrong and the question is which code is responsible; the
    /// plain one is for reading the arithmetic, which a second file and line on every line of it
    /// does not help with.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let body = self.body.borrow();

        writeln!(f, "graph {{")?;
        for (slot, op) in body.nodes.iter().enumerate() {
            write!(f, "  {} = {op}", Value::at(slot))?;
            if f.alternate() {
                write!(f, "  ; {}", body.sites[slot])?;
            }
            writeln!(f)?;
        }

        for (name, value) in &body.outputs {
            writeln!(f, "  -> {name} = {value}")?;
        }

        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One transformer feed forward, which is enough of a pass to have every kind of leaf in it.
    fn feed_forward() -> Graph {
        let g = Graph::new();
        let x = g.input("hidden");
        let fc1 = g.load("fc1.weight", &[8, 4]);
        let fc2 = g.load("fc2.weight", &[4, 8]);

        let up_weight = g.transpose(fc1, 0, 1);
        let up = g.matmul(x, up_weight);
        let up = g.gelu(up);

        let down_weight = g.transpose(fc2, 0, 1);
        let down = g.matmul(up, down_weight);
        let out = g.add(x, down);
        g.output("hidden", out);

        g
    }

    #[test]
    fn builds_nodes_in_the_order_they_are_added() {
        let g = feed_forward();

        assert_eq!(g.len(), 9);
        assert_eq!(g.inputs().len(), 1);
        assert_eq!(g.input_of("hidden"), Some(Value::at(0)));
        assert_eq!(g.output_of("hidden"), Some(Value::at(8)));
        assert_eq!(g.output_of("pooled"), None);

        // Every operand is a node that is already there, so walking the nodes is a topological
        // order. Asked of the walk rather than of the symbols, which say nothing about order.
        let mut defined = std::collections::HashSet::new();
        for (value, op) in g.nodes() {
            for operand in op.operands() {
                assert!(
                    defined.contains(&operand),
                    "{operand} is read before it is defined"
                );
            }
            defined.insert(value);
        }
    }

    #[test]
    fn reads_back_what_each_node_computes() {
        let g = feed_forward();

        assert_eq!(g.op(Value::at(4)).name(), "matmul");
        assert_eq!(
            g.op(Value::at(4)).operands(),
            vec![Value::at(0), Value::at(3)]
        );
        assert_eq!(g.op(Value::at(3)).name(), "transpose");
        assert_eq!(g.parameters(), vec!["fc1.weight", "fc2.weight"]);
        assert!(g.get(Value::at(99)).is_none());
    }

    #[test]
    fn counts_what_is_left_to_read_a_value() {
        let counts = feed_forward().use_counts();

        // The input is read twice: once by the projection and once by the residual.
        assert_eq!(counts[0], 2);
        // The last node is read by nothing, and named as an output.
        assert_eq!(counts[8], 1);
    }

    #[test]
    fn an_optional_operand_that_is_not_there_is_not_an_edge() {
        let g = Graph::new();
        let x = g.input("x");
        let weight = g.input("weight");

        let with_bias = g.layer_norm(x, Some(weight), Some(weight), 1e-5);
        let bare = g.layer_norm(x, None, None, 1e-5);

        assert_eq!(g.op(with_bias).operands().len(), 3);
        assert_eq!(g.op(bare).operands(), vec![x]);
    }

    #[test]
    fn prints_the_pass_it_stands_for() {
        let g = Graph::new();
        let x = g.input("latent");
        let weight = g.load("conv.weight", &[320, 4, 3, 3]);

        let x = g.conv2d(x, weight, None, 1, 1, 1, 1);
        let x = g.slice(x, 1, 0, 4);
        let x = g.silu(x);
        g.output("sample", x);

        assert_eq!(
            g.to_string(),
            concat!(
                "graph {\n",
                "  %0 = input(\"latent\")\n",
                "  %1 = load(\"conv.weight\", [320, 4, 3, 3])\n",
                "  %2 = conv2d(%0, %1, stride=1, padding=1, dilation=1, groups=1)\n",
                "  %3 = slice(%2, dim=1, 0..4)\n",
                "  %4 = silu(%3)\n",
                "  -> sample = %4\n",
                "}"
            )
        );
    }

    #[test]
    fn a_node_remembers_where_it_was_built() {
        let g = Graph::new();
        let x = g.input("x");

        let line = line!() + 1;
        let y = g.silu(x);

        // Through `silu` and `unary` to `push`: every step of that chain forwards its caller, so
        // what is recorded is this line and not one in this file's builders.
        assert_eq!(g.site(y).location().line(), line);
        assert!(g.site(y).location().file().ends_with("graph.rs"));
        assert_ne!(g.site(x).location().line(), g.site(y).location().line());
    }

    #[test]
    fn a_node_remembers_which_namespace_built_it() {
        let g = Graph::new();
        let attn = g.subgraph("down1").subgraph("attn");

        let x = g.input("x");
        let weight = attn.load("to_q.weight", &[8, 8]);

        // The handle that added the node, not the graph it went into: both of these are nodes of
        // `g`, and only one of them was built through `attn`.
        assert_eq!(g.site(weight).namespace(), "down1.attn");
        assert_eq!(g.site(x).namespace(), "");

        // And it is what the printed site says, after the line, for the node that has one.
        assert!(g.site(weight).to_string().ends_with(" in down1.attn"));
        assert_eq!(g.site(x).to_string(), g.site(x).location().to_string());
    }

    #[test]
    fn prints_where_a_node_was_built_only_when_asked() {
        let g = Graph::new();
        let x = g.input("latent");
        let x = g.silu(x);
        g.output("sample", x);

        assert!(!g.to_string().contains("graph.rs:"));

        let annotated = format!("{g:#}");
        assert!(annotated.contains("%1 = silu(%0)  ; "));
        assert!(annotated.contains("graph.rs:"));
    }

    #[test]
    fn names_a_weight_as_the_namespace_it_was_built_in() {
        let g = Graph::new();
        assert_eq!(g.name(), "");
        assert_eq!(g.name_of("weight"), "weight");

        let block = g.subgraph("down1");
        assert_eq!(block.name(), "down1");

        let attn = block.subgraph("attn");
        assert_eq!(attn.name(), "down1.attn");

        // Three handles, all live at once, each still naming its own level. Nothing had to be
        // given back for the one above to go on being what it was.
        attn.load("to_q.weight", &[4, 4]);
        block.load("norm.weight", &[4]);
        g.load("out.bias", &[4]);

        assert_eq!(attn.name(), "down1.attn");
        assert_eq!(block.name(), "down1");
        assert_eq!(g.name(), "");

        // And every one of those went into the one graph, under the name the level it was built
        // at made of it.
        assert_eq!(g.len(), 3);
        assert_eq!(
            g.parameters(),
            vec!["down1.attn.to_q.weight", "down1.norm.weight", "out.bias"]
        );
    }

    #[test]
    fn a_level_with_no_name_is_the_level_it_was_taken_from() {
        let g = Graph::new();

        assert_eq!(g.subgraph("").name(), "");
        assert_eq!(g.subgraph("down1").subgraph("").name(), "down1");
        assert_eq!(g.subgraph("").subgraph("down1").name(), "down1");
    }

    #[test]
    fn a_subgraph_is_the_graph_it_came_from() {
        let g = Graph::new();
        let x = g.input("hidden");

        let scaled = {
            let block = g.subgraph("down1");
            let weight = block.load("weight", &[4, 4]);

            // A value from outside the subgraph is a value of the same graph, so reading one
            // here is not the mistake `push` refuses.
            block.matmul(x, weight)
        };

        // The subgraph is gone and everything it added is here, because it was never anywhere
        // else. The value it handed back names a node of this graph.
        g.output("hidden", scaled);
        assert_eq!(g.len(), 3);
        assert_eq!(g.output_of("hidden"), Some(scaled));
    }

    #[test]
    fn cloning_a_graph_makes_another_handle_on_the_same_nodes() {
        let g = Graph::new();
        let other = g.clone();

        let x = other.input("hidden");
        g.output("hidden", g.silu(x));

        assert_eq!(g.len(), 2);
        assert_eq!(other.len(), 2);
        assert_eq!(other.output_of("hidden"), Some(Value::at(1)));
    }

    #[test]
    fn says_how_much_is_here_and_which_namespace_it_names() {
        let g = Graph::new();
        g.input("hidden");

        assert_eq!(format!("{g:?}"), "Graph(\"\", 1 nodes)");
        assert_eq!(
            format!("{:?}", g.subgraph("down1").subgraph("attn")),
            "Graph(\"down1.attn\", 1 nodes)"
        );
    }

    #[test]
    #[should_panic(expected = "is not a value of this graph")]
    fn refuses_a_value_from_another_graph() {
        let other = Graph::new();
        let stray = other.input("x");

        Graph::new().silu(stray);
    }
}
