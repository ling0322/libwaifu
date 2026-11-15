# waifu2d: one rig, many renderers

**Status: reviewed. Implemented: the core, the C API, the CPU reference and Metal, and the Mac app
moved onto them. Still to come: D3D11 and Vulkan, and the Windows app.**

waifu2d moves a drawing in layers (a PSD, as img2layers or See-through writes one): the eyes
blink, the head turns, the hair swings. Today it is written twice -- `waifu-app/swiftui`
(`Rig.swift`, `RigMotion.swift`, `ClosedParts.swift`, `PSD.swift`, drawn by `RigMetal.swift`)
and `waifu-app/wpf` (a C# port of the same, drawn by SkiaSharp on the CPU) -- and the two have
drifted: the Windows port grew a mouth, wind, winks, a jump and dragging that the Mac does not
have.

This draft puts everything but the window into one Rust crate, `waifu2d`, behind one C API,
and puts the GPU behind one trait with a backend per API: Metal, Direct3D 11, Vulkan, and a CPU
reference the others are tested against.

```
                 waifu-app/swiftui            waifu-app/wpf             (any other host)
                 NSView + CAMetalLayer        UserControl + D3DImage
                 settings UI, pointer         settings UI, pointer, mic
                        │                            │
                        └────────── waifu2d.h ───────┘           C API, its own ABI version
                                       │
   ┌───────────────────────────────────┴───────────────────────────────────┐
   │ waifu2d (Rust)                                                        │
   │   psd     PSD reader                     ─┐                           │
   │   rig     Rig::build: slots, eyes, grids, │ shared, platform-free     │
   │           strands, stock closed eyes      │                           │
   │   motion  Motion: pose, blinker, springs  │                           │
   │   stage   Stage: controls, pointer, drag, │                           │
   │           fit, step(dt) -> DrawList      ─┘                           │
   │   render  trait Backend                                               │
   │             ├─ cpu     reference, tests, no GPU                       │
   │             ├─ metal   objc2-metal         (macOS)                    │
   │             ├─ d3d11   windows crate       (Windows)                  │
   │             └─ vulkan  ash                 (Windows, Linux; macOS via │
   │                                             MoltenVK, not shipped)    │
   └───────────────────────────────────────────────────────────────────────┘
```

The rule: **a backend decides nothing.** Which white masks which iris, how faint a part is,
where the canvas sits in the window -- all of that is in the `DrawList` the shared code hands
it. A backend uploads, rasterizes as the [contract](#the-backend-contract) says, and presents.

## The crate

`waifu2d/` joins the workspace: `crate-type = ["cdylib", "staticlib", "rlib"]`, built by CMake
into `build/` beside `libwaifu` and `libwaifu_modelmanager`, as those are. Its names:

| | |
|---|---|
| directory, package | `waifu2d/`, `waifu2d` |
| library (`[lib] name = "waifu2d"`) | `libwaifu2d.dylib`, `waifu2d.dll`, `libwaifu2d.so` |
| header | `waifu2d/include/waifu2d.h` |
| C functions, types, constants | `waifu2d_*`, `Waifu2d*`, `WAIFU2D_*` |

One name throughout -- the page's own -- for the library, the header and the prefix. It does not link libwaifu or flint: nothing in it runs a model. New dependencies, all
MIT/Apache-2.0: `objc2`, `objc2-metal`, `objc2-quartz-core` (macOS), `windows` (already in
`Cargo.lock`), `ash` (behind a `vulkan` feature).

```
waifu2d/src/
  lib.rs
  psd.rs            PsdDocument::read          (from PSD.swift)
  rig.rs            Rig, RigPart, RigMesh, ... (from Rig.swift; C# where it is a superset)
  closed_parts.rs   stock closed eyes          (from ClosedParts.swift)
  blinker.rs        Blinker                    (from Rig.swift)
  motion.rs         Motion, Controls, Pose     (C# RigMotion.cs: Swift's plus the Windows inputs)
  stage.rs          Stage                      (what RigView.swift / Waifu2dView.xaml.cs do today)
  render/mod.rs     Backend, Assets, DrawList, Surface, Renderer
  render/cpu.rs     render/metal.rs  render/d3d11.rs  render/vulkan.rs
  render/shaders/   rig.metal (MSL)  rig.hlsl  rig.vert rig.frag (GLSL, made SPIR-V at build time)
  c_api.rs          -> include/waifu2d.h (cbindgen, as waifu.h is)
waifu2d/tests/      ported RigTests.swift; backend conformance
```

## The Rust interface

### Rig -- read once, then never changed

```rust
/// A part, by its place in `Rig::parts` -- which is also its place in `Assets::parts` and
/// `Frame::offsets`. A rig never changes, so neither does a part's id. Its own type so that it
/// is never mistaken for a vertex's index or a mask's id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PartId(pub u32);

pub struct Rig {
    pub width: u32,
    pub height: u32,
    /// Back to front: the order they are drawn in. `PartId(i)` is `parts[i]`.
    pub parts: Vec<RigPart>,
    /// By `Side as usize`; `None` for an eye that does not blink.
    pub eyes: [Option<EyeAnchor>; 2],
    pub anchors: Anchors,
    pub face_scale: f64,            // f64 where the Swift app had CGFloat, f32 where it had Float
    /// What did not go as hoped, in words to show: "No eye whites, so the eyes do not blink."
    pub notes: Vec<String>,
    /// The shapes that clip other parts: each eye's white, for now. See "Render".
    pub masks: Vec<MaskDef>,
}

pub struct RigPart {
    pub name: String,
    pub slot: String,
    pub role: Role,                 // EyeWhite, Irides, EyeLash, EyeClose, Eyebrow, Face, Other
    pub side: Option<Side>,         // Left is the character's right, as the picture has it
    pub rect: Rect,                 // on the canvas
    pub pixels: RgbaImage,          // the size of `rect`, not premultiplied
    pub opacity: f64,
    pub depth: f32,
    pub group: Group,               // Head, Body
    pub mesh: RigMesh,              // rest positions, uv 0..1, triangle indices
    pub swing: Option<Swing>,
}

impl Rig {
    pub fn build(document: &PsdDocument) -> Rig;
    pub fn read_psd(bytes: &[u8]) -> Result<Rig, Error>;   // PsdDocument::read + build
}
```

`Rig` is `Send + Sync` and lives in an `Arc`: it is built on a worker thread, as the Mac app
does now, and shared by the stage and the renderer without a copy.

### Motion -- the pose from one frame to the next

```rust
#[derive(Clone)]
pub struct Switches {
    pub follow_pointer: bool,   // true
    pub idle: bool,             // true: sway
    pub look_around: bool,      // true
    pub physics: bool,          // true: hair swings
    pub blink: bool,            // true: on its own
    pub wind: bool,             // true. The Mac has no switch of its own: it sets it to `idle`
    pub auto_talk: bool,        // false: babble, the mouth moving with no sound
    pub gusts: bool,            // false: the Windows app's gusty wind, which swings a skirt's hem
}

/// Everything the host's settings say. Every field's default is what the Mac does today; the
/// ones after `body_follows` are the Windows port's and do nothing at their defaults.
#[derive(Clone)]
pub struct Controls {
    pub switches: Switches,
    pub eyes_open: f32,        // 1, 0..1
    pub closes_at: f32,        // 0.3, 0..1
    pub nod: f32,              // 1, 0..2
    pub body_follows: f32,     // 1, 0..2
    pub talk_level: f32,       // 0, 0..1: the microphone's loudness, which the host measures
    pub mouth_open: f32,       // 0, 0..1
    pub smile: f32,            // 0, -1..1
    pub brows: f32,            // 0, -1..1
    pub twist: f32,            // 0, -1..1
    pub wink_left: f32,        // 0, 0..1
    pub wink_right: f32,       // 0, 0..1
    pub wind_strength: f32,    // 1, 0..3
}

pub struct Motion { /* pose, blinker, strand and bust springs, jump, shake, nod, grab */ }

impl Motion {
    /// `seed` drives look-around and the blinker, so a test sees the same frames every run.
    pub fn new(rig: &Rig, seed: u64) -> Motion;
    /// `pointer` is -1..1 across and down the canvas, clamped to ±1.5; `dt` is clamped to 0.05.
    pub fn step(&mut self, rig: &Rig, controls: &Controls, pointer: Option<[f32; 2]>, dt: f64);
    pub fn blink_now(&mut self);
    pub fn jump(&mut self);
    pub fn shake(&mut self);
    pub fn nod_once(&mut self);
    /// The part pulled `pull` canvas pixels from where it was grabbed; `None` lets go.
    pub fn grab(&mut self, part: Option<PartId>, pull: [f32; 2]);
    pub fn pose(&self) -> &Pose;
    pub fn frame(&self, rig: &Rig, physics: bool) -> Frame;
}

/// Every part's vertices where they are now, on the canvas, and how much of each shows.
pub struct Frame {
    pub positions: Vec<[f32; 2]>,   // all parts, one after another
    pub offsets: Vec<u32>,          // parts.len() + 1: part i is positions[offsets[i]..offsets[i+1]]
    pub alphas: Vec<f32>,
}

impl Frame {
    /// Part `part`'s vertices: `positions[offsets[part]..offsets[part + 1]]`.
    pub fn vertices(&self, part: PartId) -> &[[f32; 2]];
    pub fn alpha(&self, part: PartId) -> f32;
}
```

### Stage -- the control code both apps share

What `RigView.Driver` and `Waifu2dView.xaml.cs` each do by hand today: fitting the canvas into
the view, turning the pointer into -1..1, picking the part under a press, dragging it, the
babble, the clock.

```rust
pub struct Stage { /* Arc<Rig>, Motion, Controls, view size, pointer, drag */ }

impl Stage {
    pub fn new(rig: Arc<Rig>, seed: u64) -> Stage;
    pub fn rig(&self) -> &Arc<Rig>;
    pub fn controls(&self) -> &Controls;
    pub fn controls_mut(&mut self) -> &mut Controls;

    /// The view's size, in whatever unit the pointer is given in (points, DIPs). Not pixels:
    /// the renderer is told those.
    pub fn set_view_size(&mut self, width: f32, height: f32);
    /// The pointer in the view: origin top left, y down. The host flips AppKit's y.
    pub fn pointer_moved(&mut self, x: f32, y: f32);
    pub fn pointer_left(&mut self);
    /// Grabs the part under (x, y), a swinging one first; false where there is none.
    pub fn pointer_down(&mut self, x: f32, y: f32) -> bool;
    pub fn pointer_up(&mut self);

    pub fn blink_now(&mut self);
    pub fn jump(&mut self);
    pub fn shake(&mut self);
    pub fn nod_once(&mut self);

    /// Moves on by `dt` seconds: the drag, the babble, then Motion::step.
    pub fn step(&mut self, dt: f64);
    pub fn frame(&self) -> Frame;
    /// This frame as a backend draws it into `width` by `height` pixels.
    pub fn draw_list(&self, width: u32, height: u32) -> DrawList;
}

/// The canvas fitted into a box, in its middle. The one copy of RigRenderer.fit.
pub fn fit(canvas: [f32; 2], into: [f32; 2]) -> Rect;
```

### Render -- what a backend is handed

```rust
/// A rig's pictures and grids as every backend uploads them, made once per rig by the shared
/// code: premultiplied, with their mip chain, so that no backend's own mipmapping makes it
/// differ from another's.
pub struct Assets {
    pub rig_id: u64,                // which Arc<Rig> it was made from
    pub canvas_size: [u32; 2],
    pub parts: Vec<PartAssets>,
}
pub struct PartAssets {
    pub mips: Vec<RgbaImage>,       // level 0 first, premultiplied RGBA8, each half the last (2x2 box)
    pub uv: Vec<[f32; 2]>,          // 0..1
    pub indices: Vec<u32>,          // triangles
}
impl Assets { pub fn new(rig: &Rig) -> Assets; }

/// One frame, decided.
pub struct DrawList {
    pub rig_id: u64,
    pub canvas_size: [u32; 2],
    pub target: [u32; 2],           // pixels
    pub viewport: Rect,             // the canvas fitted into the target, in its pixels
    /// Every part's vertices and alpha this frame; a backend uploads `frame.positions` whole.
    pub frame: Frame,
    /// Drawn first, each into a coverage image of its own. Only masks some draw reads are
    /// listed; empty, no mask pass this frame.
    pub masks: Vec<Mask>,
    /// Then into the target, back to front. Parts that do not show are left out.
    pub draws: Vec<Draw>,
}

/// A mask's name for as long as its rig lives: the same mask has the same id every frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MaskId(pub u32);

/// A shape that clips other parts: how much of these parts, as they are this frame, covers
/// each pixel, 0 to 1.
pub struct Mask {
    pub id: MaskId,
    pub parts: Vec<PartId>,
}

pub struct Draw {
    pub part: PartId,
    pub alpha: f32,
    /// This part shows only where that mask covers. Always one of this frame's `masks`.
    pub mask: Option<MaskId>,
}
```

Which masks a rig has is decided once, by `Rig::build`, and kept on the rig:

```rust
pub struct MaskDef {
    pub id: MaskId,
    pub name: String,          // for logs and tests: "eyewhite-left", "eyewhite-right"
    pub parts: Vec<PartId>,    // what it is made of
    pub clips: Vec<PartId>,    // the parts it cuts
}
// Rig { ..., pub masks: Vec<MaskDef> }
```

Each frame `Stage::draw_list` lists the ones some visible part reads, ids unchanged:

```rust
masks: [Mask { id: MaskId(0), parts: [left_white] },
        Mask { id: MaskId(1), parts: [right_white] }],
draws: [..., Draw { part: left_iris,  alpha: 1.0, mask: Some(MaskId(0)) },
             Draw { part: right_iris, alpha: 1.0, mask: Some(MaskId(1)) }, ...],
```

The rules a `DrawList` keeps: no id twice in `masks`; every `Draw::mask` is in `masks`; ids
need not be contiguous or start at 0. A mouth's inside would be one more `MaskDef`, with no
limit on how many. How a backend keeps its masks -- an image each, an array texture with a
layer each -- is its own business and not part of this interface; it may keep one per id
from frame to frame and reuse it.

### The trait

```rust
pub enum Api { Cpu, Metal, D3D11, Vulkan }

/// What a backend draws into.
pub enum Surface {
    /// Kept by the backend and read back with read_rgba: tests, thumbnails, and any host that
    /// composes a bitmap itself.
    Offscreen { width: u32, height: u32 },
    /// A CAMetalLayer the host owns (Metal; Vulkan through VK_EXT_metal_surface).
    MetalLayer { layer: NonNull<c_void> },
    /// A window's client area (D3D11 swap chain; Vulkan through VK_KHR_win32_surface).
    Win32Window { hwnd: NonNull<c_void>, hinstance: NonNull<c_void> },
    /// A shared D3D11 texture a WPF D3DImage shows (D3D11 only); see "Open questions".
    D3D11Shared { width: u32, height: u32 },
}

pub trait Backend {
    fn api(&self) -> Api;
    /// Uploads a rig's pictures and grids, replacing whatever was loaded.
    fn load(&mut self, assets: &Assets) -> Result<(), Error>;
    /// The size of what is drawn into, in pixels: a swap chain's buffers, a drawable's size,
    /// the offscreen texture. Called before the first draw and whenever it changes.
    fn resize(&mut self, width: u32, height: u32) -> Result<(), Error>;
    /// Draws `list` over `clear` (premultiplied RGBA). A window or a layer is presented; an
    /// offscreen texture is kept for read_rgba. `list.target` is what resize was last told.
    fn draw(&mut self, list: &DrawList, clear: [f32; 4]) -> Result<(), Error>;
    /// The last draw, premultiplied RGBA8, top row first. Offscreen only; Unsupported elsewhere.
    fn read_rgba(&mut self, out: &mut [u8]) -> Result<(), Error>;
    /// For D3D11Shared: the shared handle of the texture last drawn into. None elsewhere.
    fn shared_handle(&self) -> Option<NonNull<c_void>> { None }
}

pub fn is_available(api: Api) -> bool;
pub fn default_api() -> Api;        // Metal on macOS, D3D11 on Windows, Vulkan elsewhere
pub fn create(api: Api, surface: Surface) -> Result<Box<dyn Backend>, Error>;

/// A backend and the rig it holds: loads a stage's rig the first time it is drawn, and again
/// whenever the stage's rig is another one.
pub struct Renderer { /* Box<dyn Backend>, loaded rig_id, size */ }
impl Renderer {
    pub fn new(api: Api, surface: Surface) -> Result<Renderer, Error>;
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Error>;
    pub fn draw(&mut self, stage: &Stage, clear: [f32; 4]) -> Result<(), Error>;
    pub fn read_rgba(&mut self, out: &mut [u8]) -> Result<(), Error>;
}
```

A backend is not `Send`: it is made, drawn with and dropped on one thread, the one the host's
surface belongs to.

## The backend contract

What every backend draws, to the pixel; `render/cpu.rs` is its executable form. This is
`RigMetal.swift` as it is today, written down.

- **Textures**: RGBA8 UNORM, not sRGB. `PartAssets::mips` uploaded level by level, not
  generated: each level half the last, rounded down, a 2x2 box over premultiplied texels, the last
  texel along an odd edge taking the row or column left over (a box 3 wide). One sampler: linear
  min, mag and mip filtering, clamp to edge on both axes.
- **Level of detail**: what the GPU picks. On Apple's GPUs that is the longer axis of a pixel's
  footprint in the picture -- not the textbook `max(|d/dx|, |d/dy|)` -- so the reference takes
  `log2` of the larger singular value of the derivatives of `uv * size`, one a triangle (the
  mapping is affine). Open: another vendor's GPU may pick otherwise; if D3D11 or Vulkan cannot
  meet the tolerance below, the shaders will compute the level themselves and sample at it, one
  formula for every backend.
- **Vertices**: `list.frame.vertices(part)`, canvas pixels, and the part's `uv`;
  triangles from its `indices` (u32). From canvas to clip space: `c = p / canvas_size * 2 - 1`, then
  `(c.x, -c.y)` where clip-space y is up (Metal, D3D) and `(c.x, c.y)` where it is down
  (Vulkan). The viewport is `list.viewport`, depth 0..1; no depth test, no culling.
- **Blending**, every pass: source factor One, destination factor OneMinusSourceAlpha, add, the
  same for colour and alpha. Everything is premultiplied.
- **Mask pass**, when `list.masks` is not empty: each mask is a single-channel coverage image
  (R8 UNORM or better) the size of the target, cleared to 0. Its parts are drawn into it in
  order, each writing its texel's alpha, with the blending below -- so where two of its parts
  overlap it covers as much as both together. The reference (`cpu`) and the GPU backends keep
  one array texture with a layer a mask, grown when a frame asks for more layers than it has.
- **Main pass**: the target cleared to `clear`. Each `Draw` writes `texel * alpha * m`, where
  `m` is 1 without a mask and otherwise the coverage of the mask with that id, the texel at this
  fragment's pixel (a load, not a sample).
- **Target**: BGRA8 UNORM (RGBA8 offscreen), premultiplied; the host composes it as such
  (CAMetalLayer `isOpaque = false`, D3DImage with Pbgra32).
- **Conformance**: `tests/render.rs` draws a fixture rig -- overlapping faint parts, a bent grid,
  a picture drawn at a twelfth of its size, two masks, a part past the canvas into the letterbox
  -- through `Cpu` and every backend available on the machine, offscreen, at 64x48, 333x500 and
  1024x768, and requires 99.5% of channel values within 2/255 of the reference and none beyond
  8/255. Metal on Apple silicon: 100% within 2/255 at all three.

## The C API: `waifu2d.h`

Its own library and its own ABI version, apart from `waifu.h` (whose contract is unchanged).
Conventions are ABI 6's: opaque handles, NULL or false on failure with `waifu2d_last_error`
saying why on the failing thread, knobs set by number so that a new one is a minor bump, and
every `_free` fine with NULL.

```c
/* The MIT License (MIT). Copyright (c) 2026 Xiaoyang Chen. See LICENSE. */
#ifndef LIBWAIFU_WAIFU2D_H
#define LIBWAIFU_WAIFU2D_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define WAIFU2D_ABI_VERSION_MAJOR 1
#define WAIFU2D_ABI_VERSION_MINOR 0
#define WAIFU2D_ABI_VERSION ((WAIFU2D_ABI_VERSION_MAJOR << 16) | WAIFU2D_ABI_VERSION_MINOR)

uint32_t waifu2d_abi_version(void);
/* Why the last call on this thread failed; valid until the next failing call on it. */
const char *waifu2d_last_error(void);

/* ---- the rig: read once, never changed, any thread ------------------------------------------ */

typedef struct Waifu2dRig Waifu2dRig;

typedef int32_t Waifu2dRigAttribute;
#define WAIFU2D_RIG_WIDTH       0   /* int: canvas pixels */
#define WAIFU2D_RIG_HEIGHT      1   /* int */
#define WAIFU2D_RIG_PART_COUNT  2   /* int */
#define WAIFU2D_RIG_EYE_COUNT   3   /* int: eyes that blink, 0 to 2 */
#define WAIFU2D_RIG_NOTE_COUNT  4   /* int */

/* Slow: reads and rigs the whole document. Call it off the UI thread. */
Waifu2dRig *waifu2d_rig_open(const char *psd_path);
Waifu2dRig *waifu2d_rig_read(const uint8_t *bytes, size_t length);
/* A stage or a renderer that has it keeps it alive; the caller's handle may go at once. */
void waifu2d_rig_free(Waifu2dRig *rig);
bool waifu2d_rig_get_int_attribute(const Waifu2dRig *rig, Waifu2dRigAttribute attribute, int64_t *value);
/* A note in words to show, valid while the rig is. NULL past the last. */
const char *waifu2d_rig_note(const Waifu2dRig *rig, size_t index);

/* ---- the stage: the rig moving; one thread at a time ---------------------------------------- */

typedef struct Waifu2dStage Waifu2dStage;

typedef int32_t Waifu2dSwitch;       /* bool; defaults in brackets */
#define WAIFU2D_SWITCH_FOLLOW_POINTER 0   /* [on] */
#define WAIFU2D_SWITCH_IDLE           1   /* [on]  sway */
#define WAIFU2D_SWITCH_LOOK_AROUND    2   /* [on] */
#define WAIFU2D_SWITCH_PHYSICS        3   /* [on]  hair swings */
#define WAIFU2D_SWITCH_BLINK          4   /* [on]  on its own */
#define WAIFU2D_SWITCH_WIND           5   /* [on] */
#define WAIFU2D_SWITCH_AUTO_TALK      6   /* [off] */
#define WAIFU2D_SWITCH_GUSTS          7   /* [off]  the Windows app's wind */

typedef int32_t Waifu2dControl;      /* float; defaults and ranges in brackets */
#define WAIFU2D_CONTROL_EYES_OPEN     0   /* [1]   0..1 */
#define WAIFU2D_CONTROL_CLOSES_AT     1   /* [0.3] 0..1 */
#define WAIFU2D_CONTROL_NOD           2   /* [1]   0..2 */
#define WAIFU2D_CONTROL_BODY_FOLLOWS  3   /* [1]   0..2 */
#define WAIFU2D_CONTROL_TALK_LEVEL    4   /* [0]   0..1, the host's microphone */
#define WAIFU2D_CONTROL_MOUTH_OPEN    5   /* [0]   0..1 */
#define WAIFU2D_CONTROL_SMILE         6   /* [0]  -1..1 */
#define WAIFU2D_CONTROL_BROWS         7   /* [0]  -1..1 */
#define WAIFU2D_CONTROL_TWIST         8   /* [0]  -1..1 */
#define WAIFU2D_CONTROL_WINK_LEFT     9   /* [0]   0..1 */
#define WAIFU2D_CONTROL_WINK_RIGHT   10   /* [0]   0..1 */
#define WAIFU2D_CONTROL_WIND_STRENGTH 11  /* [1]   0..3 */

typedef int32_t Waifu2dAction;
#define WAIFU2D_ACTION_BLINK 0
#define WAIFU2D_ACTION_JUMP  1
#define WAIFU2D_ACTION_SHAKE 2
#define WAIFU2D_ACTION_NOD   3

/* seed: look-around and blinks; any number. */
Waifu2dStage *waifu2d_stage_new(const Waifu2dRig *rig, uint64_t seed);
void waifu2d_stage_free(Waifu2dStage *stage);
/* False, and nothing changed, for a number this library does not know or a value out of range. */
bool waifu2d_stage_set_switch(Waifu2dStage *stage, Waifu2dSwitch which, bool on);
bool waifu2d_stage_set_control(Waifu2dStage *stage, Waifu2dControl which, float value);
bool waifu2d_stage_act(Waifu2dStage *stage, Waifu2dAction action);

/* The view, in the unit the pointer comes in (points, DIPs): origin top left, y down. */
void waifu2d_stage_set_view_size(Waifu2dStage *stage, float width, float height);
void waifu2d_stage_pointer_moved(Waifu2dStage *stage, float x, float y);
void waifu2d_stage_pointer_left(Waifu2dStage *stage);
/* True where it grabbed a part, which follows the pointer until waifu2d_stage_pointer_up. */
bool waifu2d_stage_pointer_down(Waifu2dStage *stage, float x, float y);
void waifu2d_stage_pointer_up(Waifu2dStage *stage);

/* Moves on by dt seconds, clamped to 0.05; 0 for the first frame. */
void waifu2d_stage_step(Waifu2dStage *stage, double dt);

/* ---- the renderer: one backend on one surface, on the surface's thread ---------------------- */

typedef struct Waifu2dRenderer Waifu2dRenderer;

typedef int32_t Waifu2dApi;
#define WAIFU2D_API_DEFAULT 0   /* Metal on macOS, D3D11 on Windows, Vulkan elsewhere */
#define WAIFU2D_API_CPU     1
#define WAIFU2D_API_METAL   2
#define WAIFU2D_API_D3D11   3
#define WAIFU2D_API_VULKAN  4

typedef int32_t Waifu2dSurfaceKind;
#define WAIFU2D_SURFACE_OFFSCREEN     0   /* read back with waifu2d_renderer_read_rgba */
#define WAIFU2D_SURFACE_METAL_LAYER   1   /* handle: CAMetalLayer* */
#define WAIFU2D_SURFACE_WIN32_WINDOW  2   /* handle: HWND, handle2: HINSTANCE */
#define WAIFU2D_SURFACE_D3D11_SHARED  3   /* see waifu2d_renderer_shared_handle */

typedef struct {
  Waifu2dSurfaceKind kind;
  void *handle;
  void *handle2;
  uint32_t width;    /* pixels */
  uint32_t height;
} Waifu2dSurface;

bool waifu2d_api_is_available(Waifu2dApi api);
Waifu2dRenderer *waifu2d_renderer_new(Waifu2dApi api, const Waifu2dSurface *surface);
void waifu2d_renderer_free(Waifu2dRenderer *renderer);
Waifu2dApi waifu2d_renderer_api(const Waifu2dRenderer *renderer);   /* what DEFAULT became */
bool waifu2d_renderer_resize(Waifu2dRenderer *renderer, uint32_t width, uint32_t height);
/* Draws the stage as it is now over clear (premultiplied RGBA) and presents it. Loads the
   stage's rig first if it is not the one loaded: that draw is slow. */
bool waifu2d_renderer_draw(Waifu2dRenderer *renderer, const Waifu2dStage *stage, const float clear[4]);
/* Offscreen: the last draw, premultiplied RGBA8, top row first; length = width * height * 4. */
bool waifu2d_renderer_read_rgba(Waifu2dRenderer *renderer, uint8_t *out, size_t length);
/* D3D11_SHARED: the DXGI shared handle of the texture last drawn into, for a D3DImage. */
void *waifu2d_renderer_shared_handle(const Waifu2dRenderer *renderer);

#endif
```

A frame, in any host:

```c
waifu2d_stage_step(stage, now - last);
waifu2d_renderer_resize(renderer, pixel_width, pixel_height);   /* cheap when unchanged */
waifu2d_renderer_draw(renderer, stage, (float[4]){0, 0, 0, 0});
```

## The apps afterwards

**macOS.** `Rig.swift`, `RigMotion.swift`, `RigMetal.swift` and `ClosedParts.swift` go.
`PSD.swift` and `PSDWriter.swift` stay: `Session` reads and writes img2layers' documents with
them, which has nothing to do with this page. `WaifuKit/Waifu2D.swift` is the Swift face of
`waifu2d.h`. `RigStage` keeps its `@Observable` settings and `RigView`'s `RigCanvas` -- an
`NSView` backed by a `CAMetalLayer` (`isOpaque = false`), ticked by `NSView.displayLink` --
passes them on every frame with `set_switch`/`set_control`, `wind` set with `idle`, and flips
AppKit's y for the pointer. A press is not passed on: the Mac has never dragged parts. `CWaifu`
gains `waifu2d.h`; the package links `waifu2d` beside `waifu`, and `build-app.sh` puts
`libwaifu2d.dylib` in the app's Frameworks. The rig and motion tests in `RigTests.swift` move to
`waifu2d/tests/`; what stays in Swift is the page's way to the library: a document read, drawn
by Metal with the iris inside the white, and drawn alike by the CPU.

**Windows (the PR after this one).** `Rig.cs`, `RigMotion.cs`, `RigRenderer.cs`, `Psd.cs`,
`ClosedParts.cs` and `Rgba.cs` go, and with them SkiaSharp. `MicLevel.cs` stays and feeds
`TALK_LEVEL`. `Waifu2dView` P/Invokes `waifu2d.dll`, draws offscreen with D3D11 and copies the
frame into its `WriteableBitmap`, and later shows a `D3D11_SHARED` texture in a `D3DImage`
instead (decision 5). It turns `GUSTS` on, which is its wind. What changes on Windows, each a fix
the port carries: the pointer is no longer upside down (decision 1); a jump leaves the ground
(RigMotion.cs set an upward speed that its own test never let move the rig); the mouth moves
(RigMotion.cs looked for a slot named "mouth", which no layer is given: a "mouth" layer is
`mouth_open`); and a rig no longer squashes once as it first appears (its squash began at 0).

**Unchanged on purpose.** At its defaults the Windows-only controls do nothing, so the Mac
draws what it draws now; the Mac's UI does not grow the Windows controls, and the mouth stays
as the document draws it: no stock mouth, and no babble unless a host asks for it. The mouth
is to come back from real mouth shapes, not from Anime2.5DRig's open and closed mouths.

## Tests

- `waifu2d/tests/rig.rs`, `motion.rs`: `RigTests.swift`, test for test (names, slots, eyes, grids,
  strands, springs, the head turning, the blink), plus the Windows port's mouth, wink, jump and
  grab.
- `waifu2d/tests/parity.rs`: frames from Rust against frames recorded from the Swift app
  before it is switched over -- the same fixture, seed and inputs -- to 1e-4 a vertex. Except
  where the Mac was wrong: it lifted a layer cut off by the canvas's foot (a top or hair down to
  the bottom of a portrait) off that foot as she breathed, her chest swung, her head turned or
  she leaned, and showed a band of what was behind it along the bottom. `Motion::deform` holds
  such a layer down, less the farther up it a vertex is (to the chest's reach); those vertices
  are held to moving no farther up or down than the Mac's, and not at all on the foot.
- `waifu2d/tests/render.rs`: the conformance test above. `Cpu` and `Metal` run here; `D3D11`
  and `Vulkan` where they are available.
- `waifu2d/tests/c_api.rs`: the header's promises -- NULLs, unknown numbers, the rig outliving
  its handle.
- All of them fast suites, run before a commit with the rest:
  `cargo test -p waifu2d`.

## Decisions

Reviewed and settled; what was open in the first draft.

1. **The pointer is taken y down, and Windows' upside-down look is fixed.** `Waifu2dView.PointerAt`
   copied the Mac's flip of AppKit's y, but WPF's y already points down, so on Windows she
   looked down when the pointer was above her. `Stage` takes y down and the hosts flip what needs
   flipping; a pointer beyond ±1.5 is clamped, as the Mac does, not dropped.
2. **Mipmaps are made on the CPU** (`Assets::new`, a 2x2 box over premultiplied texels) and
   uploaded level by level, so that every backend samples the same texels and Vulkan, which has
   no `generateMipmaps`, needs no blit chain. The Mac's pixels move by a rounding step where the
   drawing is drawn smaller than it is.
3. **Direct3D 11, not 12.** For a few dozen textured meshes a frame D3D11 is as fast and a fraction
   of the code, and a D3D11 texture reaches a WPF `D3DImage` through a D3D9Ex shared surface. The
   trait would take a D3D12 backend later without change.
4. **Three shaders, all source.** `rig.metal`, compiled at run time as `RigMetal.swift` does now;
   `rig.hlsl`, compiled at run time with `D3DCompile`; `rig.vert` and `rig.frag`, GLSL compiled to
   SPIR-V at build time by the same `glslangValidator` flint's Vulkan kernels are (CMake passes
   its path to `build.rs`; without one the Vulkan backend is left out with a warning). No
   compiled shader is checked in. The conformance test holds the three together.
5. **WPF reads back first, `D3DImage` after.** The Windows PR draws offscreen with D3D11 and copies
   the pixels into the `WriteableBitmap` the page has now -- `read_rgba` is there for the tests
   anyway. A later PR moves it to `D3D11_SHARED` and a `D3DImage`. A child `HWND` is not used: it
   cannot be transparent over the checkerboard.
6. **Names.** `libwaifu2d` (`waifu2d.dll`) with its own `waifu2d.h` and prefix `waifu2d_`, rather
   than new functions in `waifu.h`: the engine's C API is untouched, and an app can show the page
   without loading the engine. `libwaifu` keeps its name. `canvas_size`, not `canvas`, for the
   canvas's width and height.
