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

//! Drawing a rig, behind one trait with a backend per API.
//!
//! A backend decides nothing. Which white masks which iris, how faint a part is, where the canvas
//! sits in the target -- all of that is in the [`DrawList`] the shared code hands it, made by
//! [`draw_list`]. A backend uploads the [`Assets`], rasterizes as docs/waifu2d.md's "backend
//! contract" says, and presents. [`cpu`] is that contract's executable form, which the GPU
//! backends are tested against.

pub mod cpu;
#[cfg(target_os = "macos")]
pub mod metal;

use std::ffi::c_void;
use std::ptr::NonNull;

use crate::rig::{MaskId, PartId, Rig};
use crate::{Error, Frame, Rect, RgbaImage, Stage};

/// A rig's pictures and grids as every backend uploads them, made once per rig by the shared
/// code: premultiplied, with their mip chain, so that no backend's own mipmapping makes it differ
/// from another's.
#[derive(Clone, Debug, PartialEq)]
pub struct Assets {
    /// The [`Rig::id`] it was made from.
    pub rig_id: u64,
    pub canvas_size: [u32; 2],
    /// Part by part, as [`Rig::parts`].
    pub parts: Vec<PartAssets>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PartAssets {
    /// Level 0 first, premultiplied RGBA8, each half the last (a 2x2 box, rounded down, at least
    /// 1), down to 1 by 1. Never empty: a part with no pixels has one transparent texel.
    pub mips: Vec<RgbaImage>,
    /// 0 to 1, vertex by vertex.
    pub uv: Vec<[f32; 2]>,
    /// Triangles.
    pub indices: Vec<u32>,
}

impl Assets {
    /// Every part of `rig` premultiplied and mipped, with its grid's uv and triangles.
    pub fn new(rig: &Rig) -> Assets {
        Assets {
            rig_id: rig.id,
            canvas_size: [rig.width, rig.height],
            parts: rig
                .parts
                .iter()
                .map(|part| {
                    let base = premultiplied(&part.pixels);
                    PartAssets {
                        mips: mip_chain(base),
                        uv: part.mesh.uv.clone(),
                        indices: part.mesh.indices.clone(),
                    }
                })
                .collect(),
        }
    }
}

/// A part's picture premultiplied as RigMetal.swift's `load` did it, `(c * a + 127) / 255` a
/// channel: at least 1 by 1, and transparent where its bytes are not `width * height * 4`.
fn premultiplied(pixels: &RgbaImage) -> RgbaImage {
    let (w, h) = (pixels.width.max(1), pixels.height.max(1));
    if pixels.rgba.len() != w * h * 4 {
        return RgbaImage::transparent(w, h);
    }
    let mut rgba = pixels.rgba.clone();
    for texel in rgba.chunks_exact_mut(4) {
        let a = texel[3] as u16;
        for c in &mut texel[..3] {
            *c = ((*c as u16 * a + 127) / 255) as u8;
        }
    }
    RgbaImage::new(w, h, rgba)
}

/// `base` and every level under it down to 1 by 1, as a GPU sizes them: each half the one above,
/// rounded down, at least 1. A texel of a level is the box of the level above it that it covers,
/// 2 by 2, its sum rounded to the nearest, `(sum + n / 2) / n`. Where the level above is odd along
/// an edge, the last texel along it takes the leftover row or column as well -- a 3-wide box --
/// so that nothing of the picture is dropped; where it is 1, the box is the one there is.
fn mip_chain(base: RgbaImage) -> Vec<RgbaImage> {
    let mut mips = vec![base];
    loop {
        let above = mips.last().unwrap();
        if above.width == 1 && above.height == 1 {
            return mips;
        }
        let (w, h) = ((above.width / 2).max(1), (above.height / 2).max(1));
        let span = |i: usize, n: usize, of: usize| -> (usize, usize) {
            let start = (i * 2).min(of - 1);
            let end = if i == n - 1 { of } else { i * 2 + 2 };
            (start, end)
        };
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            let (y0, y1) = span(y, h, above.height);
            for x in 0..w {
                let (x0, x1) = span(x, w, above.width);
                let mut sum = [0u32; 4];
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        let at = (sy * above.width + sx) * 4;
                        for c in 0..4 {
                            sum[c] += above.rgba[at + c] as u32;
                        }
                    }
                }
                let n = ((y1 - y0) * (x1 - x0)) as u32;
                for c in 0..4 {
                    rgba[(y * w + x) * 4 + c] = ((sum[c] + n / 2) / n) as u8;
                }
            }
        }
        mips.push(RgbaImage::new(w, h, rgba));
    }
}

/// What a backend has loaded, as far as a [`DrawList`] must agree with it: which rig, and each
/// part's vertex count and the least count its triangles need.
#[derive(Clone, Debug, Default)]
pub(crate) struct Loaded {
    rig_id: u64,
    /// Part by part: how many uv it has, and one past the highest index of its triangles.
    parts: Vec<(usize, usize)>,
}

impl Loaded {
    pub(crate) fn of(assets: &Assets) -> Loaded {
        Loaded {
            rig_id: assets.rig_id,
            parts: assets
                .parts
                .iter()
                .map(|p| {
                    (
                        p.uv.len(),
                        p.indices.iter().map(|&i| i as usize + 1).max().unwrap_or(0),
                    )
                })
                .collect(),
        }
    }

    /// Whether `list` is a frame of this rig, every part it draws with the vertices its triangles
    /// read: what keeps a backend from reading past a buffer.
    pub(crate) fn check(&self, list: &DrawList) -> Result<(), Error> {
        if list.rig_id != self.rig_id {
            return Err(Error::InvalidArgument(format!(
                "a frame of rig {} with rig {} loaded",
                list.rig_id, self.rig_id
            )));
        }
        let count = self.parts.len();
        let frame = &list.frame;
        if frame.offsets.len() != count + 1
            || frame.alphas.len() != count
            || frame.offsets.windows(2).any(|w| w[0] > w[1])
            || frame.offsets[count] as usize > frame.positions.len()
        {
            return Err(Error::InvalidArgument(format!(
                "a frame of {} parts for a rig of {count}",
                frame.alphas.len()
            )));
        }
        let used = list
            .masks
            .iter()
            .flat_map(|m| m.parts.iter())
            .chain(list.draws.iter().map(|d| &d.part));
        for &part in used {
            let Some(&(uv, needs)) = self.parts.get(part.index()) else {
                return Err(Error::InvalidArgument(format!(
                    "{part:?} of a rig of {count}"
                )));
            };
            let have = frame.vertices(part).len();
            if have != uv || needs > have {
                return Err(Error::InvalidArgument(format!(
                    "{part:?} has {have} vertices this frame for {uv} uv"
                )));
            }
        }
        for draw in &list.draws {
            if let Some(id) = draw.mask {
                if !list.masks.iter().any(|m| m.id == id) {
                    return Err(Error::InvalidArgument(format!(
                        "{:?} is cut by {id:?}, which is not listed",
                        draw.part
                    )));
                }
            }
        }
        Ok(())
    }
}

/// A shape that clips other parts this frame: how much of these parts, as they are now, covers
/// each pixel, 0 to 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub id: MaskId,
    pub parts: Vec<PartId>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Draw {
    pub part: PartId,
    pub alpha: f32,
    /// This part shows only where that mask covers. Always one of this frame's masks.
    pub mask: Option<MaskId>,
}

/// One frame, decided. The rules it keeps: no id twice in `masks`; every `Draw::mask` is in
/// `masks`; ids need not be contiguous or start at 0.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawList {
    /// The [`Rig::id`] it is a frame of.
    pub rig_id: u64,
    pub canvas_size: [u32; 2],
    /// Pixels.
    pub target: [u32; 2],
    /// The canvas fitted into the target, in its pixels.
    pub viewport: Rect,
    /// Every part's vertices and alpha this frame; a backend uploads `frame.positions` whole.
    pub frame: Frame,
    /// Drawn first, each into a coverage image of its own. Only the masks some draw reads are
    /// here; empty, there is no mask pass this frame.
    pub masks: Vec<Mask>,
    /// Then into the target, back to front. Parts that do not show are left out.
    pub draws: Vec<Draw>,
}

/// The least alpha a part is drawn at: below it, it is left out of [`DrawList::draws`].
pub const MIN_ALPHA: f32 = 0.004;

/// `frame` of `rig` as a backend draws it into `target` pixels: RigMetal.swift's `encode`, with
/// what it decided taken out of the backend.
///
/// A mask is listed when one of the parts it clips shows this frame, with all of its own parts
/// drawn into it whether they show or not -- as RigMetal drew the whites -- and every part that
/// shows is drawn, in order, cut by the first listed mask that clips it.
pub fn draw_list(rig: &Rig, frame: Frame, target: [u32; 2]) -> DrawList {
    let shows = |part: PartId| {
        frame
            .alphas
            .get(part.index())
            .is_some_and(|&a| a >= MIN_ALPHA)
    };
    let masks: Vec<&crate::rig::MaskDef> = rig
        .masks
        .iter()
        .filter(|mask| mask.clips.iter().any(|&p| shows(p)))
        .collect();
    let draws = (0..rig.parts.len() as u32)
        .map(PartId)
        .filter(|&part| shows(part))
        .map(|part| Draw {
            part,
            alpha: frame.alpha(part),
            mask: masks
                .iter()
                .find(|mask| mask.clips.contains(&part))
                .map(|mask| mask.id),
        })
        .collect();
    let canvas_size = [rig.width, rig.height];
    DrawList {
        rig_id: rig.id,
        canvas_size,
        target,
        viewport: fit(
            [canvas_size[0] as f64, canvas_size[1] as f64],
            [target[0] as f64, target[1] as f64],
        ),
        masks: masks
            .iter()
            .map(|mask| Mask {
                id: mask.id,
                parts: mask.parts.clone(),
            })
            .collect(),
        draws,
        frame,
    }
}

/// The canvas fitted into a box, in its middle: the one copy of what RigRenderer.fit was.
pub fn fit(canvas: [f64; 2], into: [f64; 2]) -> Rect {
    let scale = (into[0] / canvas[0].max(1.0)).min(into[1] / canvas[1].max(1.0));
    let (w, h) = (canvas[0] * scale, canvas[1] * scale);
    Rect::new((into[0] - w) / 2.0, (into[1] - h) / 2.0, w, h)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Api {
    Cpu,
    Metal,
    D3D11,
    Vulkan,
}

/// What a backend draws into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// Kept by the backend and read back with [`Backend::read_rgba`]: tests, thumbnails, and any
    /// host that composes a bitmap itself.
    Offscreen { width: u32, height: u32 },
    /// A CAMetalLayer the host owns (Metal; Vulkan through VK_EXT_metal_surface).
    MetalLayer { layer: NonNull<c_void> },
    /// A window's client area (D3D11 swap chain; Vulkan through VK_KHR_win32_surface).
    Win32Window {
        hwnd: NonNull<c_void>,
        hinstance: NonNull<c_void>,
    },
    /// A shared D3D11 texture a WPF D3DImage shows (D3D11 only).
    D3D11Shared { width: u32, height: u32 },
}

/// One API drawing into one surface. Not `Send`: it is made, drawn with and dropped on one thread,
/// the one the host's surface belongs to.
pub trait Backend {
    fn api(&self) -> Api;
    /// Uploads a rig's pictures and grids, replacing whatever was loaded.
    fn load(&mut self, assets: &Assets) -> Result<(), Error>;
    /// The size of what is drawn into, in pixels: a swap chain's buffers, a drawable's size, the
    /// offscreen texture. Called before the first draw and whenever it changes.
    fn resize(&mut self, width: u32, height: u32) -> Result<(), Error>;
    /// Draws `list` over `clear` (premultiplied RGBA). A window or a layer is presented; an
    /// offscreen texture is kept for read_rgba. `list.target` is what resize was last told.
    fn draw(&mut self, list: &DrawList, clear: [f32; 4]) -> Result<(), Error>;
    /// The last draw, premultiplied RGBA8, top row first, `width * height * 4` bytes. Offscreen
    /// only; [`Error::Unsupported`] elsewhere.
    fn read_rgba(&mut self, out: &mut [u8]) -> Result<(), Error>;
    /// For [`Surface::D3D11Shared`]: the shared handle of the texture last drawn into.
    fn shared_handle(&self) -> Option<NonNull<c_void>> {
        None
    }
}

/// Metal on macOS, Direct3D 11 on Windows, Vulkan elsewhere.
pub fn default_api() -> Api {
    if cfg!(target_os = "macos") {
        Api::Metal
    } else if cfg!(windows) {
        Api::D3D11
    } else {
        Api::Vulkan
    }
}

/// Whether this build has `api` and the machine can run it.
pub fn is_available(api: Api) -> bool {
    match api {
        Api::Cpu => true,
        #[cfg(target_os = "macos")]
        Api::Metal => metal::is_available(),
        _ => false,
    }
}

pub fn create(api: Api, surface: Surface) -> Result<Box<dyn Backend>, Error> {
    match api {
        Api::Cpu => Ok(Box::new(cpu::CpuBackend::new(surface)?)),
        #[cfg(target_os = "macos")]
        Api::Metal => Ok(Box::new(metal::MetalBackend::new(surface)?)),
        other => Err(Error::Unsupported(format!(
            "{other:?} is not in this build"
        ))),
    }
}

/// A backend and the rig it holds: loads a stage's rig the first time it is drawn, and again
/// whenever the stage's rig is another one.
pub struct Renderer {
    backend: Box<dyn Backend>,
    loaded: Option<u64>,
    size: [u32; 2],
}

impl Renderer {
    pub fn new(api: Api, surface: Surface) -> Result<Renderer, Error> {
        let size = match surface {
            Surface::Offscreen { width, height } | Surface::D3D11Shared { width, height } => {
                [width, height]
            }
            _ => [0, 0],
        };
        Ok(Renderer {
            backend: create(api, surface)?,
            loaded: None,
            size,
        })
    }

    pub fn api(&self) -> Api {
        self.backend.api()
    }

    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// Cheap when the size is the one it has.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        if [width, height] == self.size {
            return Ok(());
        }
        if width == 0 || height == 0 {
            return Err(Error::InvalidArgument(format!(
                "a {width} by {height} surface"
            )));
        }
        self.backend.resize(width, height)?;
        self.size = [width, height];
        Ok(())
    }

    /// Draws the stage as it is now over `clear` and presents it; loads the stage's rig first if
    /// it is not the one loaded.
    pub fn draw(&mut self, stage: &Stage, clear: [f32; 4]) -> Result<(), Error> {
        if self.size[0] == 0 || self.size[1] == 0 {
            return Err(Error::InvalidArgument(
                "drawn before it was given a size".into(),
            ));
        }
        let rig = stage.rig();
        if self.loaded != Some(rig.id) {
            self.loaded = None;
            self.backend.load(&Assets::new(rig))?;
            self.loaded = Some(rig.id);
        }
        self.backend
            .draw(&stage.draw_list(self.size[0], self.size[1]), clear)
    }

    pub fn read_rgba(&mut self, out: &mut [u8]) -> Result<(), Error> {
        let need = self.size[0] as usize * self.size[1] as usize * 4;
        if out.len() != need {
            return Err(Error::InvalidArgument(format!(
                "{} bytes for {need}",
                out.len()
            )));
        }
        self.backend.read_rgba(out)
    }

    pub fn shared_handle(&self) -> Option<NonNull<c_void>> {
        self.backend.shared_handle()
    }

    pub fn backend_mut(&mut self) -> &mut dyn Backend {
        self.backend.as_mut()
    }
}
