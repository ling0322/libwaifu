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

//! What the tools in src/bin share.

use waifu::GenerationOptions;

/// The request a tool draws with for the model whose manifest is at `path`: what a screen would
/// start its boxes at -- the kind of model's defaults with the manifest's `suggested:` block over
/// them, as libwaifu_modelmanager works it out -- at a fixed seed, so two runs draw one picture.
pub fn settings(path: &str) -> Result<GenerationOptions, waifu::Error> {
    let said = waifu_modelmanager::defaults::defaults_json(path)
        .ok_or_else(|| waifu::Error::Model(format!("there is no manifest at {path}")))?;
    let number = |key: &str| said[key].as_f64().unwrap_or_default();
    Ok(GenerationOptions {
        width: number("width") as i32,
        height: number("height") as i32,
        num_steps: number("steps") as i32,
        guidance_scale: number("guidance") as f32,
        // The author's own, where the card gave one; nothing where it did not.
        negative_prompt: said["avoid"].as_str().unwrap_or_default().to_string(),
        seed: Some(7),
        ..GenerationOptions::default()
    })
}
