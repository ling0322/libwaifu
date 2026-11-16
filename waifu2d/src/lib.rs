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

//! waifu2d: a drawing in layers -- a PSD, as img2layers or See-through writes one -- rigged to
//! move, the way Anime2.5DRig (https://github.com/852wa/Anime2.5DRig, MIT, Copyright (c) 2026
//! hakoniwa) rigs one, and drawn by whichever GPU API the platform has.
//!
//! Everything but the window is here, once, for every app: reading the document ([`psd`]),
//! rigging it ([`rig`]), moving it ([`motion`]), what an app's page does with the pointer and
//! the clock ([`stage`]), and drawing it ([`render`]) through a [`render::Backend`] per API --
//! Metal, Direct3D 11, Vulkan, and a CPU reference the others are held to. Apps reach it through
//! the C API in [`c_api`], `waifu2d.h`. docs/waifu2d.md is the design.

pub mod blinker;
pub mod c_api;
pub mod closed_parts;
pub mod error;
pub mod geom;
pub mod motion;
pub mod mouths;
pub mod psd;
pub mod render;
pub mod rig;
pub mod rng;
pub mod speech;
pub mod stage;

pub use error::Error;
pub use geom::{Point, Rect, RgbaImage, Size};
pub use motion::{Controls, Frame, Motion, Pose, Switches};
pub use rig::{PartId, Rig};
pub use stage::Stage;
