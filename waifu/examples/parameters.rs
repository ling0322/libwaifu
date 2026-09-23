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

//! Every tensor a model's graph loads, with the shape it expects, as one name per line.
//!
//! ```text
//! cargo run --release --example parameters -- s2mel
//! ```
//!
//! This is the list an exporter has to satisfy, and it is worth having as a program rather than
//! as a careful reading of the module: a graph builds its names out of nested subgraphs and a
//! loop counter, so the only authority on what it asks for is the graph itself. An exporter
//! written against this list and then checked against it cannot quietly disagree with the
//! runtime about a name -- which is the failure that produces a model that loads and speaks
//! nonsense.
//!
//! The shapes are what the graph declares, not what the checkpoint holds. Where the two disagree
//! it is the exporter's job to say so; `tools/indextts_gpt_exporter.py` is the worked example.

use waifu::flint::{DType, Device, Graph, Op};
use waifu::indextts::{emotion, s2mel, semantic_codec, w2v_bert};

const F32: DType = DType::Float;
const CPU: Device = Device::Cpu;

/// A length that is long enough for every extent in these graphs to be a real one. Nothing here
/// depends on it, because a load's shape never does.
const FRAMES: i32 = 64;

fn w2v_bert_graph(g: &Graph) {
    let config = w2v_bert::Config::w2v_bert_2();

    w2v_bert::graph(
        g,
        g.input("x"),
        &config,
        FRAMES,
        w2v_bert::Config::USED_LAYERS,
        g.input("distances"),
        F32,
        CPU,
    )
    .expect("w2v-bert builds");
}

/// The encoder, which IndexTTS-2.5 does not run -- see the module note. Here so that the half
/// that defines the alphabet can still be asked what it loads.
fn semantic_codec_graph(g: &Graph) {
    let config = semantic_codec::Config::indextts();

    semantic_codec::similarity(g, g.input("x"), &config, FRAMES, F32, CPU)
        .expect("the codec builds");
}

/// The decoder, which is the half the pipeline runs.
fn semantic_codec_decode_graph(g: &Graph) {
    let config = semantic_codec::Config::indextts();

    semantic_codec::decode(g, g.input("codes"), &config, FRAMES, F32, CPU)
        .expect("the codec decodes");
}

fn s2mel_graph(g: &Graph) {
    let config = s2mel::Config::indextts();

    s2mel::graph(
        g,
        g.input("x"),
        g.input("prompt_x"),
        g.input("cond"),
        g.input("style"),
        g.input("time"),
        g.input("time2"),
        &config,
        FRAMES,
        g.input("cos"),
        g.input("sin"),
        F32,
        CPU,
    )
    .expect("s2mel builds");

    s2mel::length_regulator(
        &g.subgraph("length_regulator"),
        g.input("tokens"),
        g.input("selection"),
        &s2mel::RegulatorConfig::indextts(),
        FRAMES,
        F32,
        CPU,
    )
    .expect("the regulator builds");
}

fn emotion_graph(g: &Graph) {
    let config = emotion::Config::indextts();

    emotion::graph(
        g,
        g.input("x"),
        &config,
        FRAMES,
        g.input("positions"),
        F32,
        CPU,
    )
    .expect("the emotion path builds");
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();

    let build: fn(&Graph) = match which.as_str() {
        "w2v_bert" => w2v_bert_graph,
        "semantic_codec" => semantic_codec_graph,
        "semantic_codec_decode" => semantic_codec_decode_graph,
        "s2mel" => s2mel_graph,
        "indextts_emotion" => emotion_graph,
        _ => {
            eprintln!(
                "usage: parameters <w2v_bert|semantic_codec|semantic_codec_decode|s2mel|indextts_emotion>\n\
                 \n\
                 Prints one `name\\tshape` per tensor the graph loads."
            );
            std::process::exit(2);
        }
    };

    let g = Graph::new();
    build(&g);

    let mut seen: Vec<(String, Vec<i32>)> = Vec::new();
    for (_, op) in g.nodes() {
        if let Op::Load { name, shape } = op {
            if !seen.iter().any(|(held, _)| *held == name) {
                seen.push((name, shape));
            }
        }
    }

    for (name, shape) in &seen {
        let dims: Vec<String> = shape.iter().map(|size| size.to_string()).collect();
        println!("{name}\t{}", dims.join(","));
    }

    eprintln!("{} tensors", seen.len());
}
