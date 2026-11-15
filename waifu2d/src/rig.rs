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

//! A drawing in layers rigged to move, the way Anime2.5DRig (https://github.com/852wa/Anime2.5DRig,
//! MIT, Copyright (c) 2026 hakoniwa) rigs one: the names its layers are known by, where each sits
//! in depth and whether it is the head's or the body's, the anchors it moves about, the strands its
//! hair swings in, and the closed eyes it is given where it has none. Its numbers are
//! Anime2.5DRig's (lib/rigger.js); how it moves is [`crate::motion`]'s.
//!
//! A blink is not a lid. The open eye -- its white, iris and lashes -- is squashed toward a line
//! across it, and when it is a quarter open it fades into a closed eye drawn apart: the document's
//! own, or a stock one ([`crate::closed_parts`]) fitted to the eye and tinted to its lashes. The
//! iris stays inside the white as the white is squashed: the white is a mask ([`MaskDef`]) the iris
//! is cut by. Which eye is which is the picture's: an eye layer drawn as one is cut in two at the
//! middle of the face.
//!
//! A port of waifu-app/swiftui/Sources/Waifu/Rig.swift.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use unicode_normalization::UnicodeNormalization;

use crate::closed_parts::{self, Drawing};
use crate::geom::smooth;
use crate::psd::PsdDocument;
use crate::{Error, Point, Rect, RgbaImage, Size};

/// A part, by its place in [`Rig::parts`] -- which is also its place in
/// [`crate::render::Assets::parts`] and [`crate::Frame::offsets`]. A rig never changes, so neither
/// does a part's id. Its own type so that it is never mistaken for a vertex's index or a mask's id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct PartId(pub u32);

impl PartId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A mask's name for as long as its rig lives: the same mask has the same id every frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct MaskId(pub u32);

/// Which eye, as the picture has them: `Left` is the character's right.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    pub const ALL: [Side; 2] = [Side::Left, Side::Right];

    /// Its place in [`Rig::eyes`].
    pub fn index(self) -> usize {
        match self {
            Side::Left => 0,
            Side::Right => 1,
        }
    }
}

/// What a layer is, for the parts of the rig that treat it as its own: an eye's, a brow, the face,
/// and everything else.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Role {
    EyeWhite,
    Irides,
    EyeLash,
    EyeClose,
    Eyebrow,
    Face,
    Other,
}

/// Whether a layer moves with the head or with the body.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Group {
    Head,
    Body,
}

/// How a layer swings: hair, or a thing hung from something -- an earring, a tail.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phys {
    Hair,
    Sway,
}

/// What a kind of layer is to the rig: Anime2.5DRig's `SLOTS`. Depth is 1 at the face and more in
/// front of it: how far a layer moves as the head turns.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Slot {
    pub depth: f32,
    /// `None` for one that is the head's or the body's by where it is.
    pub group: Option<Group>,
    pub phys: Option<Phys>,
}

/// One strand a layer of hair swings in: where it hangs, its root and its tip, on the canvas.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Strand {
    pub x: f32,
    pub root_y: f32,
    pub tip_y: f32,
}

/// How a layer of hair or a hanging thing swings: its strands, each vertex's weight towards each
/// one, and how far down it each vertex is, 0 at the root to 1 at the tip.
#[derive(Clone, PartialEq, Debug)]
pub struct Swing {
    pub kind: Phys,
    /// Front hair, side hair and ahoge swing as front hair does.
    pub front: bool,
    pub strands: Vec<Strand>,
    /// `strands.len()` a vertex, vertex by vertex.
    pub weights: Vec<f32>,
    pub along: Vec<f32>,
}

/// A layer's grid: its vertices at rest, on the canvas, where each one reads the layer's picture,
/// and the triangles between them. Anime2.5DRig's: cells of 42 pixels at a 768 canvas -- 30 for
/// what swings -- as wide as the canvas is over 768 but never under 0.6 of that, at least two a
/// side.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct RigMesh {
    pub rest: Vec<[f32; 2]>,
    /// 0 to 1 across and down the part's picture.
    pub uv: Vec<[f32; 2]>,
    /// Triangles, three to one.
    pub indices: Vec<u32>,
}

impl RigMesh {
    pub fn grid(rect: Rect, canvas: u32, swings: bool) -> RigMesh {
        let cell = (if swings { 30.0 } else { 42.0 }) * f64::max(0.6, f64::from(canvas) / 768.0);
        RigMesh::cells(rect, cell)
    }

    /// The grid hair is drawn on: [`RigMesh::grid`]'s, but at least 16 cells along its longer
    /// side. Hair bends along its length from where it leaves the head, and a small drawing's few
    /// cells of 30 bend it in a couple of straight pieces with a crease between.
    fn hair_grid(rect: Rect, canvas: u32) -> RigMesh {
        let cell = 30.0 * f64::max(0.6, f64::from(canvas) / 768.0);
        RigMesh::cells(rect, f64::min(cell, rect.width.max(rect.height) / 16.0))
    }

    /// Cells of about `cell` over `rect`, at least two a side.
    fn cells(rect: Rect, cell: f64) -> RigMesh {
        let nx = ((rect.width / cell).round() as i64).max(2) as usize;
        let ny = ((rect.height / cell).round() as i64).max(2) as usize;
        let mut rest = Vec::with_capacity((nx + 1) * (ny + 1));
        let mut uv = Vec::with_capacity((nx + 1) * (ny + 1));
        for j in 0..=ny {
            for i in 0..=nx {
                let (u, v) = (i as f32 / nx as f32, j as f32 / ny as f32);
                uv.push([u, v]);
                rest.push([
                    rect.min_x() as f32 + u * rect.width as f32,
                    rect.min_y() as f32 + v * rect.height as f32,
                ]);
            }
        }
        let mut indices = Vec::with_capacity(nx * ny * 6);
        for j in 0..ny {
            for i in 0..nx {
                let a = (j * (nx + 1) + i) as u32;
                let b = a + 1;
                let c = a + (nx + 1) as u32;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        RigMesh { rest, uv, indices }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct RigPart {
    pub name: String,
    /// Its slot's name -- "front hair", "topwear" -- or the name as it is folded where it has none.
    pub slot: String,
    pub role: Role,
    pub side: Option<Side>,
    /// Where it is on the canvas.
    pub rect: Rect,
    /// RGBA, not premultiplied, the size of `rect`.
    pub pixels: RgbaImage,
    pub opacity: f64,
    pub depth: f32,
    pub group: Group,
    /// The grid it is drawn on, moved vertex by vertex as it moves.
    pub mesh: RigMesh,
    pub swing: Option<Swing>,
}

/// An eye at rest: the box of its white, the middle of its iris, and the line it closes onto.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct EyeAnchor {
    pub white: Rect,
    pub iris: Point,
    pub close_y: f64,
}

/// Where the rig moves about: the face, the neck it turns on, the body's foot and the chest.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Anchors {
    pub face: Rect,
    pub face_center: Point,
    /// What the head turns about: 85 % of the way down the neck.
    pub neck_pivot: Point,
    pub neck_top: f64,
    pub neck_bottom: f64,
    /// What the body leans about: under the neck, at the canvas's foot.
    pub body_pivot: Point,
    /// The middle of the chest and how far it reaches across and down.
    pub chest: Point,
    pub chest_reach: Size,
}

/// A shape that clips other parts, decided once, when the rig is built: today each eye's white,
/// which its iris is kept inside.
#[derive(Clone, PartialEq, Debug)]
pub struct MaskDef {
    pub id: MaskId,
    /// For logs and tests: "eyewhite-left", "eyewhite-right".
    pub name: String,
    /// What it is made of: how much of these parts covers a pixel is how much the mask does.
    pub parts: Vec<PartId>,
    /// The parts it cuts.
    pub clips: Vec<PartId>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Rig {
    /// Which rig this is, for a renderer to tell whether it has it loaded: unique in the process.
    pub id: u64,
    pub width: u32,
    pub height: u32,
    /// Bottom first, the hidden ones left out: the order they are drawn in. `PartId(i)` is
    /// `parts[i]`.
    pub parts: Vec<RigPart>,
    /// By [`Side::index`]; `None` for an eye that does not blink.
    pub eyes: [Option<EyeAnchor>; 2],
    pub anchors: Anchors,
    /// How wide the face is over 333: what every distance the rig moves by is scaled by.
    pub face_scale: f64,
    /// What it could not do, to say under the picture.
    pub notes: Vec<String>,
    pub masks: Vec<MaskDef>,
    /// The mouth made over the drawn one, which moves through the shapes speech makes; `None`
    /// until [`crate::mouths::with_drawn_mouths`] makes one.
    pub mouth: Option<crate::mouths::MouthRig>,
}

static NEXT_RIG_ID: AtomicU64 = AtomicU64::new(1);

/// A new [`Rig::id`].
pub fn next_rig_id() -> u64 {
    NEXT_RIG_ID.fetch_add(1, Ordering::Relaxed)
}

impl Rig {
    pub fn part(&self, id: PartId) -> &RigPart {
        &self.parts[id.index()]
    }

    pub fn eye(&self, side: Side) -> Option<&EyeAnchor> {
        self.eyes[side.index()].as_ref()
    }

    /// The white of `side`'s eye, which its iris is kept inside.
    pub fn white(&self, side: Side) -> Option<PartId> {
        self.parts
            .iter()
            .rposition(|p| p.role == Role::EyeWhite && p.side == Some(side))
            .map(|i| PartId(i as u32))
    }

    /// Rigs a document: Rig.swift's `Rig.build`.
    pub fn build(document: &PsdDocument) -> Rig {
        build(document)
    }

    /// Reads a document and rigs it.
    pub fn read_psd(bytes: &[u8]) -> Result<Rig, Error> {
        Ok(Rig::build(&PsdDocument::read(bytes)?))
    }
}

// ---- names ---------------------------------------------------------------------------------------

const fn slot(depth: f32, group: Option<Group>, phys: Option<Phys>) -> Slot {
    Slot { depth, group, phys }
}

/// Anime2.5DRig's slots, by name.
pub const SLOTS: &[(&str, Slot)] = &[
    ("wings", slot(0.48, Some(Group::Body), None)),
    ("tail", slot(0.50, Some(Group::Body), Some(Phys::Sway))),
    ("back hair", slot(0.55, Some(Group::Head), Some(Phys::Hair))),
    ("footwear", slot(0.83, Some(Group::Body), None)),
    ("legwear", slot(0.84, Some(Group::Body), None)),
    ("bottomwear", slot(0.88, Some(Group::Body), None)),
    ("neck", slot(0.95, Some(Group::Body), None)),
    ("topwear", slot(0.90, Some(Group::Body), None)),
    ("neckwear", slot(0.98, Some(Group::Body), None)),
    ("handwear", slot(0.86, Some(Group::Body), None)),
    ("objects", slot(1.00, None, None)),
    ("earwear", slot(0.97, Some(Group::Head), Some(Phys::Sway))),
    ("ears", slot(0.96, Some(Group::Head), None)),
    ("face", slot(1.00, Some(Group::Head), None)),
    ("facedetail", slot(1.02, Some(Group::Head), None)),
    ("headwear", slot(1.20, Some(Group::Head), None)),
    ("mouth_close", slot(1.08, Some(Group::Head), None)),
    ("mouth_open", slot(1.08, Some(Group::Head), None)),
    ("nose", slot(1.15, Some(Group::Head), None)),
    ("eyewhite", slot(1.06, Some(Group::Head), None)),
    ("eyebrow", slot(1.14, Some(Group::Head), None)),
    ("irides", slot(1.08, Some(Group::Head), None)),
    ("eyelash", slot(1.12, Some(Group::Head), None)),
    ("eye_close", slot(1.12, Some(Group::Head), None)),
    ("eye_close2", slot(1.12, Some(Group::Head), None)),
    ("eyewear", slot(1.18, Some(Group::Head), None)),
    ("side hair", slot(1.22, Some(Group::Head), Some(Phys::Hair))),
    (
        "front hair",
        slot(1.28, Some(Group::Head), Some(Phys::Hair)),
    ),
    ("ahoge", slot(1.30, Some(Group::Head), Some(Phys::Hair))),
];

/// The slot of that name, where Anime2.5DRig has one.
pub fn slot_named(name: &str) -> Option<Slot> {
    SLOTS.iter().find(|(n, _)| *n == name).map(|(_, s)| *s)
}

/// The names each slot is known by once folded (see [`fold`]): Anime2.5DRig's, with See-through's
/// among them.
fn aliases() -> &'static HashMap<String, &'static str> {
    static ALIASES: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    ALIASES.get_or_init(|| {
        let groups: [(&'static str, &str); 29] = [
            (
                "front hair",
                "fronthair hairfront bangs fringe 前髪 まえがみ",
            ),
            ("back hair", "backhair hairback 後ろ髪 後髪 うしろがみ"),
            (
                "side hair",
                "sidehair hairside sidelock sidelocks 横髪 サイド髪 もみあげ",
            ),
            ("ahoge", "ahoge アホ毛 あほ毛"),
            ("hair", "hair 髪 髪の毛 かみ"),
            ("face", "face 顔 かお 輪郭 facebase"),
            (
                "facedetail",
                "facedetail blush cheek cheeks 頬 頬染め チーク",
            ),
            ("eyewhite", "eyewhite eyewhites sclera 白目"),
            ("irides", "irides iris irises pupil pupils 瞳 黒目 虹彩"),
            (
                "eyelash",
                "eyelash eyelashes lash lashes eyeline まつ毛 まつげ 睫毛 アイライン",
            ),
            ("eyebrow", "eyebrow eyebrows brow brows 眉 眉毛 まゆ まゆげ"),
            (
                "eye_close",
                "eyeclose eyeclosed eyesclosed closedeye closedeyes eyelashc 閉じ目 目閉じ 閉眼",
            ),
            (
                "mouth_open",
                "mouthopen openmouth mouth 口 開き口 口開き 開口",
            ),
            (
                "mouth_close",
                "mouthclose mouthclosed closedmouth mouthc 閉じ口 口閉じ",
            ),
            ("nose", "nose 鼻 はな"),
            ("ears", "ears ear 耳"),
            (
                "earwear",
                "earwear earring earrings イヤリング ピアス 耳飾り",
            ),
            ("neck", "neck 首"),
            (
                "neckwear",
                "neckwear necktie tie necklace choker ネクタイ ネックレス チョーカー 首飾り",
            ),
            (
                "topwear",
                "topwear top tops shirt body torso clothes 服 上着 体 胴体 上半身 トップス",
            ),
            (
                "bottomwear",
                "bottomwear bottom bottoms skirt pants スカート ズボン ボトムス 下半身",
            ),
            ("handwear", "handwear hand hands arm arms gloves 手 腕 手袋"),
            (
                "legwear",
                "legwear leg legs socks tights 脚 足 靴下 ソックス タイツ",
            ),
            ("footwear", "footwear shoes shoe boots 靴 ブーツ"),
            (
                "headwear",
                "headwear hat cap hairpin hairband 帽子 カチューシャ 髪飾り ヘアピン",
            ),
            ("eyewear", "eyewear glasses 眼鏡 メガネ めがね"),
            ("tail", "tail しっぽ 尻尾 シッポ"),
            ("wings", "wings wing 羽 翼 羽根"),
            ("objects", "objects object 小物 持ち物"),
        ];
        let mut map = HashMap::new();
        for (slot, names) in groups {
            for name in names.split(' ') {
                map.insert(fold(name), slot);
            }
        }
        map
    })
}

/// A layer's name as the aliases are written: compatibility-normalized, lower case, without a
/// copy's suffix or the marks between words.
pub fn fold(name: &str) -> String {
    let mut folded = trim_whitespaces(&compatibility_mapped(name).to_lowercase()).to_string();
    for suffix in ["のコピー"] {
        if folded.ends_with(suffix) {
            folded.truncate(folded.len() - suffix.len());
        }
    }
    if let Some(copy) = copy_suffix(&folded) {
        folded.truncate(copy);
    }
    folded.chars().filter(|c| !" _-・.".contains(*c)).collect()
}

/// A layer's slot and which eye it is of, by its name: "前髪2" is a piece of `front hair`,
/// "eye_close2" the long closed eye, See-through's "mouth" an open one. The side is the
/// character's in the name -- See-through's "-r", "右" -- and the picture's here: the character's
/// right is on the picture's left. A name no slot has comes back folded, its own.
pub fn slot_of(name: &str) -> (String, Option<Side>) {
    let mut name = trim_whitespaces(&compatibility_mapped(name).to_lowercase()).to_string();
    let mut side = None;
    let marks = [
        ("-r", Side::Left),
        ("_r", Side::Left),
        (" r", Side::Left),
        (".r", Side::Left),
        (" right", Side::Left),
        ("右", Side::Left),
        ("-l", Side::Right),
        ("_l", Side::Right),
        (" l", Side::Right),
        (".l", Side::Right),
        (" left", Side::Right),
        ("左", Side::Right),
    ];
    for (mark, which) in marks {
        if name.ends_with(mark) {
            name.truncate(name.len() - mark.len());
            side = Some(which);
            break;
        }
        if mark.chars().count() == 1 && name.starts_with(mark) {
            name.drain(..mark.len());
            side = Some(which);
            break;
        }
    }
    let folded = fold(&name);
    if ["eyeclose2", "閉じ目2"].contains(&folded.as_str()) {
        return ("eye_close2".to_string(), side);
    }
    if let Some(slot) = aliases().get(&folded) {
        return (slot.to_string(), side);
    }
    // A number on the end is another piece of the same thing: "前髪2", "eyelash_2".
    let bare = folded.trim_end_matches(is_decimal_digit);
    if let Some(slot) = aliases().get(bare) {
        return (slot.to_string(), side);
    }
    (folded, side)
}

/// What a layer is, for the eyes, brows and face, and which eye.
pub fn role_of(name: &str) -> (Role, Option<Side>) {
    let (slot, side) = slot_of(name);
    (role_of_slot(&slot), side)
}

pub fn role_of_slot(slot: &str) -> Role {
    match slot {
        "eyewhite" => Role::EyeWhite,
        "irides" => Role::Irides,
        "eyelash" => Role::EyeLash,
        "eye_close" | "eye_close2" => Role::EyeClose,
        "eyebrow" => Role::Eyebrow,
        "face" => Role::Face,
        _ => Role::Other,
    }
}

/// Whether a layer's name has a number on its end: one strand group among several.
pub fn is_numbered(name: &str) -> bool {
    fold(name).chars().next_back().is_some_and(is_decimal_digit)
}

/// `s` without what Foundation's `.whitespaces` trims from either end: the spaces (Unicode's Zs)
/// and the tab, but not a line's end.
fn trim_whitespaces(s: &str) -> &str {
    s.trim_matches(|c: char| {
        matches!(
            c,
            '\t' | ' ' | '\u{A0}' | '\u{1680}' | '\u{2000}'
                ..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
        )
    })
}

/// What `\s` is to the regular expressions Foundation runs (ICU's): `[\t\n\f\r\p{Z}]`.
fn is_regex_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{C}' | '\r' | ' ' | '\u{A0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200A}' | '\u{2028}' | '\u{2029}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// What `\d` is to ICU: a decimal digit, Unicode's Nd -- each of which is one of a run of ten
/// starting at one of these zeros.
fn is_decimal_digit(c: char) -> bool {
    const ZEROS: [u32; 61] = [
        0x0030, 0x0660, 0x06F0, 0x07C0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66,
        0x0CE6, 0x0D66, 0x0DE6, 0x0E50, 0x0ED0, 0x0F20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946,
        0x19D0, 0x1A80, 0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0,
        0xA9F0, 0xAA50, 0xABF0, 0xFF10, 0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0,
        0x112F0, 0x11450, 0x114D0, 0x11650, 0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50,
        0x11DA0, 0x16A60, 0x16AC0, 0x16B50, 0x1E140, 0x1E2F0, 0x1E950, 0x1FBF0,
    ];
    let u = c as u32;
    c.is_numeric()
        && (ZEROS.iter().any(|&zero| (zero..zero + 10).contains(&u))
            || (0x1D7CE..=0x1D7FF).contains(&u))
}

/// Where `\s+copy(\s+\d+)?$` -- a copy's suffix, "eyelash copy 2" -- begins in `s`, if it ends
/// with one: the start of the match Foundation's `range(of:options: .regularExpression)` finds.
fn copy_suffix(s: &str) -> Option<usize> {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    // Where the run of `pred` ending at `end` (exclusive) starts.
    let run_start = |end: usize, pred: &dyn Fn(char) -> bool| {
        let mut i = end;
        while i > 0 && pred(chars[i - 1].1) {
            i -= 1;
        }
        i
    };
    let copy_at = |end: usize| -> Option<usize> {
        let word = ['c', 'o', 'p', 'y'];
        if end < word.len() || chars[end - word.len()..end].iter().map(|c| c.1).ne(word) {
            return None;
        }
        let before = end - word.len();
        let start = run_start(before, &is_regex_space);
        (start < before).then(|| chars.get(start).map_or(s.len(), |c| c.0))
    };
    // With a number after it, the number's own space before it; or with none.
    let digits = run_start(chars.len(), &is_decimal_digit);
    if digits < chars.len() {
        let spaces = run_start(digits, &is_regex_space);
        if spaces < digits {
            if let Some(start) = copy_at(spaces) {
                return Some(start);
            }
        }
    }
    copy_at(chars.len())
}

/// `s` under Unicode's compatibility mapping, NFKC, as Foundation's
/// `precomposedStringWithCompatibilityMapping` has it: "前髪２" is "前髪2", "ﾏﾂｹﾞ" is "マツゲ".
fn compatibility_mapped(s: &str) -> String {
    s.nfkc().collect()
}

// ---- pixels --------------------------------------------------------------------------------------

/// What a layer's pixels say about it, in its own coordinates.
pub struct Pixels;

impl Pixels {
    /// The box of what is drawn, alpha above `floor`; `None` for nothing. Rig.swift's `Pixels.box`.
    pub fn bounds(image: &RgbaImage, floor: u8) -> Option<Rect> {
        let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, -1i64, -1i64);
        for y in 0..image.height {
            for x in 0..image.width {
                if image.rgba[(y * image.width + x) * 4 + 3] > floor {
                    x0 = x0.min(x as i64);
                    x1 = x1.max(x as i64);
                    y0 = y0.min(y as i64);
                    y1 = y1.max(y as i64);
                }
            }
        }
        if x1 < 0 {
            return None;
        }
        Some(Rect::new(
            x0 as f64,
            y0 as f64,
            (x1 - x0 + 1) as f64,
            (y1 - y0 + 1) as f64,
        ))
    }

    /// Clears every speck: each piece of what is drawn above `floor`, its pixels touching side to
    /// side, smaller than `smallest`. A model's layers are scattered with them, faint and alone,
    /// all over the canvas -- enough to make every layer's box the whole of it.
    pub fn denoise(image: &mut RgbaImage, floor: u8, smallest: usize) {
        let (w, h) = (image.width, image.height);
        let px = &mut image.rgba;
        let mut seen = vec![false; w * h];
        let mut piece: Vec<usize> = Vec::new();
        for start in 0..w * h {
            if seen[start] || px[start * 4 + 3] <= floor {
                continue;
            }
            piece.clear();
            piece.push(start);
            seen[start] = true;
            let mut next = 0;
            while next < piece.len() {
                let p = piece[next];
                next += 1;
                let (x, y) = (p % w, p / w);
                let neighbours = [
                    (x > 0).then(|| p.wrapping_sub(1)),
                    (x + 1 < w).then_some(p + 1),
                    (y > 0).then(|| p.wrapping_sub(w)),
                    (y + 1 < h).then_some(p + w),
                ];
                for q in neighbours.into_iter().flatten() {
                    if !seen[q] && px[q * 4 + 3] > floor {
                        seen[q] = true;
                        piece.push(q);
                    }
                }
            }
            if piece.len() < smallest {
                for &p in &piece {
                    px[p * 4 + 3] = 0;
                }
            }
        }
    }

    /// The middle of what is drawn above `floor`, each pixel weighed by its alpha.
    pub fn centroid(image: &RgbaImage, floor: u8) -> Option<Point> {
        let (mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0);
        for y in 0..image.height {
            for x in 0..image.width {
                let alpha = image.rgba[(y * image.width + x) * 4 + 3];
                if alpha <= floor {
                    continue;
                }
                let a = f64::from(alpha);
                sx += a * (x as f64 + 0.5);
                sy += a * (y as f64 + 0.5);
                sw += a;
            }
        }
        (sw > 0.0).then(|| Point::new(sx / sw, sy / sw))
    }

    /// The colour of what is drawn, weighed by alpha: 0 to 1 each.
    pub fn mean_colour(image: &RgbaImage) -> Option<(f64, f64, f64)> {
        let (mut r, mut g, mut b, mut w) = (0.0, 0.0, 0.0, 0.0);
        for px in image.rgba.chunks_exact(4) {
            let a = f64::from(px[3]);
            r += a * f64::from(px[0]);
            g += a * f64::from(px[1]);
            b += a * f64::from(px[2]);
            w += a;
        }
        (w > 0.0).then(|| (r / w / 255.0, g / w / 255.0, b / w / 255.0))
    }

    /// The part of it inside `rect`, which is in its own pixels.
    pub fn crop(image: &RgbaImage, rect: Rect) -> RgbaImage {
        let (x0, y0) = (rect.min_x() as usize, rect.min_y() as usize);
        let (w, h) = (rect.width as usize, rect.height as usize);
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            let from = ((y0 + y) * image.width + x0) * 4;
            rgba[y * w * 4..(y + 1) * w * 4].copy_from_slice(&image.rgba[from..from + w * 4]);
        }
        RgbaImage::new(w, h, rgba)
    }
}

fn offset(rect: Rect, dx: f64, dy: f64) -> Rect {
    Rect::new(rect.x + dx, rect.y + dy, rect.width, rect.height)
}

// ---- strands -------------------------------------------------------------------------------------

/// The first of `items` nearest `x`: Swift's `min(by:)`, which keeps the first of equals.
fn nearest(items: &[i64], x: i64) -> i64 {
    let mut best = items[0];
    for &item in &items[1..] {
        if (item - x).abs() < (best - x).abs() {
            best = item;
        }
    }
    best
}

impl Rig {
    /// Where a layer of hair hangs in strands: Anime2.5DRig's `detectStrands`. The bottom edge of
    /// what is drawn, column by column, smoothed over 41 columns, has a peak at each strand's tip:
    /// up to `want` of the most prominent, at least `min_sep` apart, and more where the layer is
    /// wide enough that a column is that far from every one. Each strand runs from the top of its
    /// column to the bottom, in the layer's own pixels.
    pub fn strands(image: &RgbaImage, want: usize, min_sep: usize) -> Vec<Strand> {
        let (w, h) = (image.width, image.height);
        let mut top = vec![-1i64; w];
        let mut bottom = vec![0i64; w];
        for x in 0..w {
            for y in 0..h {
                if image.rgba[(y * w + x) * 4 + 3] > 16 {
                    if top[x] < 0 {
                        top[x] = y as i64;
                    }
                    bottom[x] = y as i64;
                }
            }
        }
        let painted: Vec<i64> = (0..w as i64).filter(|&x| top[x as usize] >= 0).collect();
        let (Some(&min_x), Some(&max_x)) = (painted.first(), painted.last()) else {
            return vec![];
        };

        // Box-smoothed over 41 columns, always divided by 41, beyond the edges 0.
        let mut smoothed = vec![0.0; w];
        for (x, s) in smoothed.iter_mut().enumerate() {
            let mut sum = 0.0;
            for k in -20i64..=20 {
                let at = x as i64 + k;
                if at >= 0 && at < w as i64 {
                    sum += bottom[at as usize] as f64;
                }
            }
            *s = sum / 41.0;
        }
        let mut picked: Vec<i64> = Vec::new();
        for peak in Rig::peaks(&smoothed, min_sep, 10.0) {
            if picked.len() < want {
                picked.push(peak as i64);
            }
        }
        // More, at whichever of 40 columns across it is farthest from every strand, while that is
        // at least min_sep away.
        let inset = 30.min((max_x - min_x) / 4);
        let candidates: Vec<i64> = (0..40)
            .map(|k| {
                let span = (max_x - min_x - 2 * inset) as f64;
                let x = min_x + inset + (span * f64::from(k) / 39.0).round() as i64;
                nearest(&painted, x)
            })
            .collect();
        for _ in 0..50 {
            if picked.len() >= want {
                continue;
            }
            let distance = |x: i64| {
                picked
                    .iter()
                    .map(|p| (p - x).abs())
                    .min()
                    .unwrap_or(i64::MAX)
            };
            // The first of the farthest, as Swift's `max(by:)` keeps.
            let mut far = candidates[0];
            for &c in &candidates[1..] {
                if distance(far) < distance(c) {
                    far = c;
                }
            }
            if distance(far) < (min_sep as i64).max(1) {
                break;
            }
            picked.push(far);
        }
        let mut columns: Vec<i64> = picked.iter().map(|&x| nearest(&painted, x)).collect();
        columns.sort_unstable();
        columns.dedup();
        columns
            .into_iter()
            .map(|c| Strand {
                x: c as f32,
                root_y: top[c as usize] as f32,
                tip_y: bottom[c as usize] as f32,
            })
            .collect()
    }

    /// The peaks of `a`, most prominent first, none within `min_sep` of one kept: a rise to a flat
    /// top counts at the top's middle, unless it rises again after. A peak's prominence is how far
    /// it stands over the higher of the lowest points on either side before something higher.
    pub fn peaks(a: &[f64], min_sep: usize, min_prominence: f64) -> Vec<usize> {
        let n = a.len() as i64;
        let at = |i: i64| a[i as usize];
        let mut found: Vec<(i64, f64)> = Vec::new();
        let mut i = 1i64;
        while i < n - 1 {
            if !(at(i) > at(i - 1)) {
                i += 1;
                continue;
            }
            let mut j = i;
            while j + 1 < n && at(j + 1) == at(i) {
                j += 1;
            }
            if j + 1 < n && at(j + 1) < at(i) {
                let p = (i + j) / 2;
                let (mut left, mut right) = (at(p), at(p));
                let mut k = p - 1;
                while k >= 0 && at(k) <= at(p) {
                    left = left.min(at(k));
                    k -= 1;
                }
                k = p + 1;
                while k < n && at(k) <= at(p) {
                    right = right.min(at(k));
                    k += 1;
                }
                let prominence = at(p) - left.max(right);
                if prominence >= min_prominence {
                    found.push((p, prominence));
                }
            }
            i = j + 1;
        }
        // Stable, as Swift's sort is: equally prominent ones in the order they were found.
        found.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept: Vec<i64> = Vec::new();
        for (p, _) in found {
            if kept.iter().all(|&k| (k - p).abs() >= min_sep as i64) {
                kept.push(p);
            }
        }
        kept.into_iter().map(|p| p as usize).collect()
    }

    /// Each vertex's weight towards each strand -- a Gaussian over 0.6 of the strands' median
    /// spacing, summing to 1 -- and how far down the weighted strand it is.
    pub fn swing(kind: Phys, front: bool, strands: &[Strand], mesh: &RigMesh) -> Swing {
        let mut gaps: Vec<f32> = strands.windows(2).map(|s| s[1].x - s[0].x).collect();
        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let spacing = if gaps.is_empty() {
            120.0
        } else {
            gaps[gaps.len() / 2]
        };
        let sigma = f32::max(1.0, spacing * 0.6);
        let mut weights = Vec::with_capacity(mesh.rest.len() * strands.len());
        let mut along = Vec::with_capacity(mesh.rest.len());
        for p in &mesh.rest {
            let mut w: Vec<f32> = strands
                .iter()
                .map(|s| (-((s.x - p[0]) / sigma).powf(2.0)).exp())
                .collect();
            let sum = w.iter().fold(0.0f32, |a, b| a + b);
            if sum < 1e-6 {
                w = (0..strands.len())
                    .map(|i| if i == 0 { 1.0 } else { 0.0 })
                    .collect();
            } else {
                w = w.iter().map(|v| v / sum).collect();
            }
            let root = w
                .iter()
                .zip(strands)
                .map(|(w, s)| w * s.root_y)
                .fold(0.0f32, |a, b| a + b);
            let tip = w
                .iter()
                .zip(strands)
                .map(|(w, s)| w * s.tip_y)
                .fold(0.0f32, |a, b| a + b);
            weights.extend_from_slice(&w);
            along.push(((p[1] - root) / f32::max(1.0, tip - root)).clamp(0.0, 1.0));
        }
        Swing {
            kind,
            front,
            strands: strands.to_vec(),
            weights,
            along,
        }
    }
}

// ---- where hair hangs from -----------------------------------------------------------------------

/// The head, as hair hangs from it: the face as it is drawn, a little wider, and over it an ellipse
/// from the top of the hair to the chin, 1.2 times as wide as the face -- the skull and the hair on
/// it.
///
/// How far down hair a vertex is ([`Swing::along`]) is how far it is from where the hair leaves the
/// head, walking inside the hair -- not straight down from a strand's root, which a lock that
/// curves or a layer that holds both the back of the head and a tail hanging from it get wrong.
/// Hair drawn behind the face hangs from what of it the head covers, and that does not swing: it
/// is the back of the head. Hair drawn in front of it hangs from above the top of the face. A piece
/// that touches neither hangs from its own top.
///
/// The face is the largest piece of the face layer: img2layers puts a stray ornament or a strand in
/// it now and then, and that would move the head off it.
struct Head {
    /// The face, 3 % of its width wider all round, on cells `cell` across from `origin`.
    face: Vec<bool>,
    face_size: (usize, usize),
    cell: f64,
    origin: (f64, f64),
    centre: Point,
    radius: (f64, f64),
    /// Hair drawn in front of the face hangs from above this: the top of the face.
    hairline: f64,
}

impl Head {
    /// `hair_top`: the top of the highest layer of hair.
    fn new(face: &Piece, hair_top: f64) -> Head {
        let image = &face.image;
        let (w, h) = (image.width, image.height);
        let cell = ((w.max(h) as f64 / 256.0).ceil() as usize).max(1);
        let (gw, gh) = (w.div_ceil(cell), h.div_ceil(cell));
        let mut drawn = vec![false; gw * gh];
        for y in 0..h {
            for x in 0..w {
                if image.rgba[(y * w + x) * 4 + 3] > 16 {
                    drawn[(y / cell) * gw + x / cell] = true;
                }
            }
        }
        // Its largest piece.
        let mut seen = vec![false; gw * gh];
        let mut largest: Vec<usize> = Vec::new();
        for start in 0..gw * gh {
            if !drawn[start] || seen[start] {
                continue;
            }
            seen[start] = true;
            let mut cells = vec![start];
            let mut k = 0;
            while k < cells.len() {
                let g = cells[k];
                k += 1;
                for (n, _) in neighbours(g, gw, gh) {
                    if drawn[n] && !seen[n] {
                        seen[n] = true;
                        cells.push(n);
                    }
                }
            }
            if cells.len() > largest.len() {
                largest = cells;
            }
        }
        let size = cell as f64;
        let (left, top) = (face.left as f64, face.top as f64);
        let (x0, x1) = largest
            .iter()
            .fold((gw, 0), |(a, b), &g| (a.min(g % gw), b.max(g % gw)));
        let (y0, y1) = largest
            .iter()
            .fold((gh, 0), |(a, b), &g| (a.min(g / gw), b.max(g / gw)));
        let face_box = Rect::new(
            left + x0 as f64 * size,
            top + y0 as f64 * size,
            (x1 + 1).saturating_sub(x0) as f64 * size,
            (y1 + 1).saturating_sub(y0) as f64 * size,
        );
        // A little wider.
        let reach = (0.03 * face_box.width / size).ceil() as i64;
        let mut wider = vec![false; gw * gh];
        for &g in &largest {
            let (gx, gy) = ((g % gw) as i64, (g / gw) as i64);
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let (nx, ny) = (gx + dx, gy + dy);
                    if nx >= 0 && ny >= 0 && nx < gw as i64 && ny < gh as i64 {
                        wider[ny as usize * gw + nx as usize] = true;
                    }
                }
            }
        }
        let (crown, chin) = (hair_top.min(face_box.min_y()), face_box.max_y());
        Head {
            face: wider,
            face_size: (gw, gh),
            cell: size,
            origin: (left, top),
            centre: Point::new(face_box.mid_x(), (crown + chin) / 2.0),
            radius: (
                f64::max(1.0, 0.6 * face_box.width),
                f64::max(1.0, (chin - crown) / 2.0),
            ),
            hairline: face_box.min_y(),
        }
    }

    /// Whether the head covers canvas point (`x`, `y`).
    fn covers(&self, x: f64, y: f64) -> bool {
        let u = (x - self.centre.x) / self.radius.0;
        let v = (y - self.centre.y) / self.radius.1;
        if u * u + v * v <= 1.0 {
            return true;
        }
        let gx = ((x - self.origin.0) / self.cell).floor();
        let gy = ((y - self.origin.1) / self.cell).floor();
        let (gw, gh) = self.face_size;
        gx >= 0.0
            && gy >= 0.0
            && (gx as usize) < gw
            && (gy as usize) < gh
            && self.face[gy as usize * gw + gx as usize]
    }

    /// Each vertex of `mesh` -- over a layer of hair, `image` at (`left`, `top`) on the canvas --
    /// as far down the hair as it is, 0 where it leaves the head to 1 at the farthest of it;
    /// `behind` for hair drawn behind the face. Worked on cells of a few pixels: at most 320 across.
    fn along(
        &self,
        image: &RgbaImage,
        left: i64,
        top: i64,
        mesh: &RigMesh,
        behind: bool,
    ) -> Vec<f32> {
        let (w, h) = (image.width, image.height);
        let cell = ((w.max(h) as f64 / 320.0).ceil() as usize).max(1);
        let (gw, gh) = (w.div_ceil(cell), h.div_ceil(cell));
        let mut hair = vec![false; gw * gh];
        for y in 0..h {
            for x in 0..w {
                if image.rgba[(y * w + x) * 4 + 3] > 16 {
                    hair[(y / cell) * gw + x / cell] = true;
                }
            }
        }
        let size = cell as f64;
        let centre = |g: usize| {
            (
                left as f64 + ((g % gw) as f64 + 0.5) * size,
                top as f64 + ((g / gw) as f64 + 0.5) * size,
            )
        };
        let mut seeds: Vec<bool> = (0..gw * gh)
            .map(|g| {
                hair[g] && {
                    let (x, y) = centre(g);
                    if behind {
                        self.covers(x, y)
                    } else {
                        y < self.hairline
                    }
                }
            })
            .collect();
        // A piece of hair the head does not reach hangs from its own top.
        let mut piece = vec![usize::MAX; gw * gh];
        for start in 0..gw * gh {
            if !hair[start] || piece[start] != usize::MAX {
                continue;
            }
            let mut cells = vec![start];
            piece[start] = start;
            let mut k = 0;
            while k < cells.len() {
                let g = cells[k];
                k += 1;
                for n in neighbours(g, gw, gh) {
                    if hair[n.0] && piece[n.0] == usize::MAX {
                        piece[n.0] = start;
                        cells.push(n.0);
                    }
                }
            }
            if cells.iter().all(|&g| !seeds[g]) {
                let first = cells.iter().map(|&g| g / gw).min().unwrap_or(0);
                for &g in &cells {
                    seeds[g] = g / gw <= first + 1;
                }
            }
        }
        let start: Vec<f64> = seeds
            .iter()
            .map(|&s| if s { 0.0 } else { f64::INFINITY })
            .collect();
        let inside = geodesic(gw, gh, &hair, start, size);
        let longest = inside
            .iter()
            .zip(&hair)
            .filter(|(d, &h)| h && d.is_finite())
            .map(|(d, _)| *d)
            .fold(1.0, f64::max);
        // Beyond the hair, the nearest hair's own: what the grid's vertices between the locks are
        // given, so that a triangle over a lock's edge has the edge's on both sides of it. The
        // nearest, not the least far along plus the way to it, which would hand the gap beside a
        // tail the back of the head's 0 from across the gap.
        let everywhere = nearest_value(gw, gh, &hair, &inside);
        let at = |gx: i64, gy: i64| {
            everywhere
                [gy.clamp(0, gh as i64 - 1) as usize * gw + gx.clamp(0, gw as i64 - 1) as usize]
        };
        mesh.rest
            .iter()
            .map(|p| {
                let fx = (f64::from(p[0]) - left as f64) / size - 0.5;
                let fy = (f64::from(p[1]) - top as f64) / size - 0.5;
                let (x0, y0) = (fx.floor(), fy.floor());
                let (tx, ty) = (fx - x0, fy - y0);
                let (x0, y0) = (x0 as i64, y0 as i64);
                let d = (at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx) * (1.0 - ty)
                    + (at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx) * ty;
                (d / longest).clamp(0.0, 1.0) as f32
            })
            .collect()
    }
}

/// The cells around cell `g` of a `gw` by `gh` grid, with how far each is in cells.
fn neighbours(g: usize, gw: usize, gh: usize) -> impl Iterator<Item = (usize, f64)> {
    let (x, y) = ((g % gw) as i64, (g / gw) as i64);
    [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]
    .into_iter()
    .filter_map(move |(dx, dy): (i64, i64)| {
        let (nx, ny) = (x + dx, y + dy);
        (nx >= 0 && ny >= 0 && nx < gw as i64 && ny < gh as i64).then(|| {
            let step = if dx != 0 && dy != 0 {
                std::f64::consts::SQRT_2
            } else {
                1.0
            };
            (ny as usize * gw + nx as usize, step)
        })
    })
}

/// Distances over a `gw` by `gh` grid of cells `size` apart, through the cells `open` lets through,
/// from what `distance` already holds: Dijkstra's, eight ways.
fn geodesic(gw: usize, gh: usize, open: &[bool], mut distance: Vec<f64>, size: f64) -> Vec<f64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    // A distance is never negative, so its bits order as it does.
    let mut queue: BinaryHeap<Reverse<(u64, usize)>> = distance
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .map(|(g, d)| Reverse((d.to_bits(), g)))
        .collect();
    while let Some(Reverse((bits, g))) = queue.pop() {
        let d = f64::from_bits(bits);
        if d > distance[g] {
            continue;
        }
        for (n, step) in neighbours(g, gw, gh) {
            let next = d + step * size;
            if open[n] && next < distance[n] {
                distance[n] = next;
                queue.push(Reverse((next.to_bits(), n)));
            }
        }
    }
    distance
}

/// `value` over a `gw` by `gh` grid, and each cell `from` does not mark given the value of the
/// nearest cell it does, walking eight ways.
fn nearest_value(gw: usize, gh: usize, from: &[bool], value: &[f64]) -> Vec<f64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut way = vec![f64::INFINITY; gw * gh];
    let mut out = value.to_vec();
    let mut queue: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for (g, &f) in from.iter().enumerate() {
        if f {
            way[g] = 0.0;
            queue.push(Reverse((0f64.to_bits(), g)));
        }
    }
    while let Some(Reverse((bits, g))) = queue.pop() {
        let d = f64::from_bits(bits);
        if d > way[g] {
            continue;
        }
        for (n, step) in neighbours(g, gw, gh) {
            let next = d + step;
            if !from[n] && next < way[n] {
                way[n] = next;
                out[n] = out[g];
                queue.push(Reverse((next.to_bits(), n)));
            }
        }
    }
    out
}

// ---- building ------------------------------------------------------------------------------------

/// Each shown layer, trimmed to what is drawn.
#[derive(Clone)]
struct Piece {
    name: String,
    slot: String,
    side: Option<Side>,
    left: i64,
    top: i64,
    image: RgbaImage,
    opacity: f64,
}

/// The last of `pieces` in slot `slot`, of `side` where it is given.
fn last<'a>(pieces: &'a [Piece], slot: &str, side: Option<Side>) -> Option<&'a Piece> {
    pieces
        .iter()
        .rev()
        .find(|p| p.slot == slot && (side.is_none() || p.side == side))
}

/// What moves in `document`, and how. Anything it does not know is drawn as it is, with the head
/// or the body by where it is.
fn build(document: &PsdDocument) -> Rig {
    let (width, height) = (f64::from(document.width), f64::from(document.height));
    // Cleared of specks first, as Anime2.5DRig does, and trimmed to what is left above 16.
    let mut pieces: Vec<Piece> = Vec::new();
    for layer in document.layers.iter().filter(|l| l.visible) {
        let mut image = layer.image.clone();
        Pixels::denoise(&mut image, 16, 40);
        let Some(bounds) = Pixels::bounds(&image, 16) else {
            continue;
        };
        let (slot, side) = slot_of(&layer.name);
        pieces.push(Piece {
            name: layer.name.clone(),
            slot,
            side,
            left: i64::from(layer.left) + bounds.min_x() as i64,
            top: i64::from(layer.top) + bounds.min_y() as i64,
            image: Pixels::crop(&image, bounds),
            opacity: layer.opacity,
        });
    }
    // A plain "hair" is the front hair above the face and the back hair under it.
    if let Some(face_at) = pieces.iter().rposition(|p| p.slot == "face") {
        for (i, piece) in pieces.iter_mut().enumerate() {
            if piece.slot == "hair" {
                piece.slot = if i > face_at {
                    "front hair"
                } else {
                    "back hair"
                }
                .to_string();
            }
        }
    }

    // The face: its box and middle, or where a face would be.
    let face = last(&pieces, "face", None);
    let face_box = face
        .and_then(|f| Pixels::bounds(&f.image, 8).map(|b| offset(b, f.left as f64, f.top as f64)))
        .unwrap_or(Rect::new(
            0.35 * width,
            0.1 * height,
            0.3 * width,
            0.4 * height,
        ));
    let face_center = face
        .and_then(|f| {
            Pixels::centroid(&f.image, 16)
                .map(|c| Point::new(c.x + f.left as f64, c.y + f.top as f64))
        })
        .unwrap_or(Point::new(width / 2.0, 0.3 * height));
    let face_width = face
        .and_then(|f| Pixels::bounds(&f.image, 16))
        .map(|b| b.width);

    // Eyes drawn as one layer each, cut at the middle of the face.
    const EYEISH: [&str; 6] = [
        "eyewhite",
        "irides",
        "eyelash",
        "eye_close",
        "eye_close2",
        "eyebrow",
    ];
    let mut cut: Vec<Piece> = Vec::new();
    for piece in pieces {
        if !EYEISH.contains(&piece.slot.as_str()) || piece.side.is_some() {
            cut.push(piece);
            continue;
        }
        let image_width = piece.image.width as i64;
        let at = (face_center.x.round() as i64 - piece.left)
            .max(0)
            .min(image_width);
        for (side, from, to) in [(Side::Left, 0, at), (Side::Right, at, image_width)] {
            if to <= from {
                continue;
            }
            let half = Pixels::crop(
                &piece.image,
                Rect::new(
                    from as f64,
                    0.0,
                    (to - from) as f64,
                    piece.image.height as f64,
                ),
            );
            let Some(bounds) = Pixels::bounds(&half, 16) else {
                continue;
            };
            cut.push(Piece {
                side: Some(side),
                left: piece.left + from + bounds.min_x() as i64,
                top: piece.top + bounds.min_y() as i64,
                image: Pixels::crop(&half, bounds),
                ..piece.clone()
            });
        }
    }
    let mut pieces = cut;

    // Each eye's anchor: its white's box, its iris's middle, and where it closes -- its own closed
    // eye's middle, or 62 % of the way down its white.
    let mut eyes: [Option<EyeAnchor>; 2] = [None, None];
    let mut notes: Vec<String> = Vec::new();
    for side in Side::ALL {
        let Some(white) = last(&pieces, "eyewhite", Some(side)) else {
            continue;
        };
        let Some(bounds) = Pixels::bounds(&white.image, 16) else {
            continue;
        };
        let rect = offset(bounds, white.left as f64, white.top as f64);
        let iris = last(&pieces, "irides", Some(side));
        let middle = iris
            .and_then(|iris| {
                Pixels::centroid(&iris.image, 16)
                    .map(|c| Point::new(c.x + iris.left as f64, c.y + iris.top as f64))
            })
            .unwrap_or(Point::new(rect.mid_x(), rect.mid_y()));
        let closed = last(&pieces, "eye_close", Some(side));
        let close_y = closed
            .and_then(|closed| Pixels::centroid(&closed.image, 16).map(|c| c.y + closed.top as f64))
            .unwrap_or(rect.min_y() + 0.62 * rect.height);
        eyes[side.index()] = Some(EyeAnchor {
            white: rect,
            iris: middle,
            close_y,
        });
    }
    if eyes.iter().all(Option::is_none) {
        notes.push("No eye white layer (eyewhite, 白目): the eyes cannot blink.".to_string());
    }

    // The neck the head turns on: 85 % down it, or under the face where there is none.
    let neck = last(&pieces, "neck", None);
    let neck_box = neck
        .and_then(|n| Pixels::bounds(&n.image, 8).map(|b| offset(b, n.left as f64, n.top as f64)));
    let neck_center_x =
        neck.and_then(|n| Pixels::centroid(&n.image, 16).map(|c| c.x + n.left as f64));
    let neck_pivot = neck_box
        .map(|b| {
            Point::new(
                neck_center_x.unwrap_or(b.mid_x()),
                b.min_y() + 0.85 * b.height,
            )
        })
        .unwrap_or(Point::new(face_center.x, face_box.max_y() + 20.0));
    let (neck_top, neck_bottom) = neck_box
        .map(|b| (b.min_y(), b.max_y()))
        .unwrap_or((face_box.max_y(), face_box.max_y() + 60.0));

    // A closed eye for each eye without one: the stock drawing, 1.1 times as wide as the white,
    // its middle on the white's, 55 % of it above the line it closes onto, in the colour of that
    // eye's lashes (or brow).
    let mut made: Vec<Piece> = Vec::new();
    for side in Side::ALL {
        let Some(eye) = eyes[side.index()] else {
            continue;
        };
        if pieces
            .iter()
            .any(|p| p.slot == "eye_close" && p.side == Some(side))
        {
            continue;
        }
        let lash =
            last(&pieces, "eyelash", Some(side)).or_else(|| last(&pieces, "eyebrow", Some(side)));
        let tint = lash
            .and_then(|l| Pixels::mean_colour(&l.image))
            .unwrap_or((0.2, 0.15, 0.15));
        let Some(drawing) = Rig::closed_eye(side, eye.white.width * 1.1, tint) else {
            continue;
        };
        made.push(Piece {
            name: "eye_close (made)".to_string(),
            slot: "eye_close".to_string(),
            side: Some(side),
            left: (eye.white.mid_x() - drawing.width as f64 / 2.0).round() as i64,
            top: (eye.close_y - 0.55 * drawing.height as f64).round() as i64,
            image: drawing,
            opacity: 1.0,
        });
    }
    if !made.is_empty() {
        // Over the lashes, or the irides, or the face: where a closed eye would be drawn.
        let at = pieces
            .iter()
            .rposition(|p| p.slot == "eyelash")
            .or_else(|| pieces.iter().rposition(|p| p.slot == "irides"))
            .or_else(|| pieces.iter().rposition(|p| p.slot == "face"))
            .map_or(pieces.len(), |i| i + 1);
        pieces.splice(at..at, made);
    }

    // The head hair hangs from, and which layers are drawn behind the face.
    let face_at = pieces.iter().rposition(|p| p.slot == "face");
    let head = face_at.map(|at| {
        let hair_top = pieces
            .iter()
            .filter(|p| slot_named(&p.slot).and_then(|s| s.phys) == Some(Phys::Hair))
            .map(|p| p.top as f64)
            .fold(f64::INFINITY, f64::min);
        Head::new(&pieces[at], hair_top)
    });

    let parts: Vec<RigPart> = pieces
        .into_iter()
        .enumerate()
        .map(|(index, piece)| {
            let rect = Rect::new(
                piece.left as f64,
                piece.top as f64,
                piece.image.width as f64,
                piece.image.height as f64,
            );
            let slot = slot_named(&piece.slot);
            // One no slot places is the head's above the bottom of the face and the body's under
            // it.
            let group = slot.and_then(|s| s.group).unwrap_or_else(|| {
                let y = Pixels::centroid(&piece.image, 16)
                    .map(|c| c.y + piece.top as f64)
                    .unwrap_or(rect.mid_y());
                if y < face_box.max_y() {
                    Group::Head
                } else {
                    Group::Body
                }
            });
            let phys = slot.and_then(|s| s.phys);
            let mesh = if phys == Some(Phys::Hair) && head.is_some() {
                RigMesh::hair_grid(rect, document.width)
            } else {
                RigMesh::grid(rect, document.width, phys.is_some())
            };
            let mut swing = None;
            if let Some(phys) = phys {
                let width = piece.image.width as f64;
                let (want, min_sep) = match phys {
                    Phys::Hair => {
                        let want = if is_numbered(&piece.name) {
                            ((width / 110.0).round() as i64).clamp(2, 6)
                        } else {
                            6
                        };
                        (want, 30.max((width / (want as f64 * 1.6)).round() as i64))
                    }
                    Phys::Sway => {
                        let want = ((width / 140.0).round() as i64).clamp(1, 4);
                        (want, 20.max((width / (want as f64 * 1.6)).round() as i64))
                    }
                };
                let found: Vec<Strand> =
                    Rig::strands(&piece.image, want as usize, min_sep as usize)
                        .into_iter()
                        .map(|s| Strand {
                            x: s.x + piece.left as f32,
                            root_y: s.root_y + piece.top as f32,
                            tip_y: s.tip_y + piece.top as f32,
                        })
                        .collect();
                if !found.is_empty() {
                    let front = ["front hair", "side hair", "ahoge"].contains(&piece.slot.as_str());
                    let mut s = Rig::swing(phys, front, &found, &mesh);
                    // Hair hangs from the head: how far down it a vertex is, along the hair.
                    if let (Phys::Hair, Some(head), Some(at)) = (phys, &head, face_at) {
                        s.along =
                            head.along(&piece.image, piece.left, piece.top, &mesh, index < at);
                    }
                    swing = Some(s);
                }
            }
            RigPart {
                role: role_of_slot(&piece.slot),
                name: piece.name,
                slot: piece.slot,
                side: piece.side,
                rect,
                pixels: piece.image,
                opacity: piece.opacity,
                depth: slot.map_or(1.0, |s| s.depth),
                group,
                mesh,
                swing,
            }
        })
        .collect();

    // Each eye's white, which its iris is drawn inside: a mask of the whites of that side, cutting
    // its irides. RigMetal.swift drew them into a channel of one texture each.
    let mut masks = Vec::new();
    for side in Side::ALL {
        let of = |role: Role| -> Vec<PartId> {
            parts
                .iter()
                .enumerate()
                .filter(|(_, p)| p.role == role && p.side == Some(side))
                .map(|(i, _)| PartId(i as u32))
                .collect()
        };
        let whites = of(Role::EyeWhite);
        if whites.is_empty() {
            continue;
        }
        let name = match side {
            Side::Left => "eyewhite-left",
            Side::Right => "eyewhite-right",
        };
        masks.push(MaskDef {
            id: MaskId(side.index() as u32),
            name: name.to_string(),
            parts: whites,
            clips: of(Role::Irides),
        });
    }

    let face_height = face_box.height;
    let anchors = Anchors {
        face: face_box,
        face_center,
        neck_pivot,
        neck_top,
        neck_bottom,
        body_pivot: Point::new(neck_pivot.x, height),
        chest: Point::new(neck_pivot.x, neck_bottom + face_height * 0.60),
        chest_reach: Size::new(
            f64::max(1.0, face_box.width * 0.60),
            f64::max(1.0, face_height * 0.45),
        ),
    };
    Rig {
        id: next_rig_id(),
        width: document.width,
        height: document.height,
        parts,
        eyes,
        anchors,
        face_scale: face_width.unwrap_or(width / 3.0) / 333.0,
        notes,
        masks,
        mouth: None,
    }
}

// ---- closed eyes ---------------------------------------------------------------------------------

impl Rig {
    /// The stock closed eye for `side`, scaled to `width` and painted `tint`.
    pub fn closed_eye(side: Side, width: f64, tint: (f64, f64, f64)) -> Option<RgbaImage> {
        let drawing = match side {
            Side::Left => &closed_parts::LEFT,
            Side::Right => &closed_parts::RIGHT,
        };
        Rig::stock(drawing, width, tint)
    }

    /// A stock drawing's ink scaled to `width` and painted `tint`, each pixel as light as it was
    /// against the ink's mean, at most 2.2 times; the skin it is edged in left out.
    pub fn stock(drawing: &Drawing, width: f64, tint: (f64, f64, f64)) -> Option<RgbaImage> {
        let source = drawing.rgba()?;
        if !(width >= 1.0) {
            return None;
        }
        let (dw, dh) = (drawing.width, drawing.height);
        let w = (width.round() as i64).max(1) as usize;
        let h = ((dh as f64 * width / dw as f64).round() as i64).max(1) as usize;
        let luma = |i: usize| {
            (0.299 * f64::from(source[i])
                + 0.587 * f64::from(source[i + 1])
                + 0.114 * f64::from(source[i + 2]))
                / 255.0
        };
        // How much of a pixel is ink: the drawings are edged in skin -- a line's soft edge, a lower
        // lip's highlight -- which, tinted, would be a red blot on the face. Skin, at a luma of
        // 0.85 or more, is left out; the line, under 0.5, is kept whole.
        let ink = |luma: f64| smooth((0.85 - luma) / 0.35);
        let (mut sum, mut weight) = (0.0, 0.0);
        for i in (0..source.len()).step_by(4) {
            let a = f64::from(source[i + 3]) * ink(luma(i));
            sum += a * luma(i);
            weight += a;
        }
        let mean = if weight > 0.0 {
            f64::max(sum / weight, 1e-3)
        } else {
            1.0
        };
        // Bilinear, in premultiplied alpha so that the edge does not darken, then back.
        let mut out = vec![0u8; w * h * 4];
        let sample = |x: f64, y: f64, c: usize| -> f64 {
            let fx = x.max(0.0).min((dw - 1) as f64);
            let fy = y.max(0.0).min((dh - 1) as f64);
            let (x0, y0) = (fx as usize, fy as usize);
            let (x1, y1) = ((x0 + 1).min(dw - 1), (y0 + 1).min(dh - 1));
            let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
            let at = |x: usize, y: usize| {
                let i = (y * dw + x) * 4;
                let a = f64::from(source[i + 3]) / 255.0;
                if c == 3 {
                    a
                } else {
                    a * luma(i)
                }
            };
            (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty)
                + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty
        };
        for y in 0..h {
            for x in 0..w {
                let sx = (x as f64 + 0.5) * dw as f64 / w as f64 - 0.5;
                let sy = (y as f64 + 0.5) * dh as f64 / h as f64 - 0.5;
                let covered = sample(sx, sy, 3);
                if !(covered > 0.0) {
                    continue;
                }
                let lightness = sample(sx, sy, 0) / covered;
                let a = covered * ink(lightness);
                if !(a > 0.0) {
                    continue;
                }
                let shade = f64::min(2.2, lightness / mean);
                let i = (y * w + x) * 4;
                out[i] = f64::min(255.0, (tint.0 * shade * 255.0).round()) as u8;
                out[i + 1] = f64::min(255.0, (tint.1 * shade * 255.0).round()) as u8;
                out[i + 2] = f64::min(255.0, (tint.2 * shade * 255.0).round()) as u8;
                out[i + 3] = f64::min(255.0, (a * 255.0).round()) as u8;
            }
        }
        Some(RgbaImage::new(w, h, out))
    }
}
