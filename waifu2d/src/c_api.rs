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

//! The C API, `waifu2d.h`: everything an app outside this crate uses. docs/waifu2d.md is its
//! contract.
//!
//! Three things to hold: a rig, read once and never changed, on any thread; a stage, the rig
//! moving, on one thread at a time; and a renderer, one backend drawing into one surface, on the
//! thread the surface belongs to. A call that fails returns NULL or false, and
//! `waifu2d_last_error` says why on the thread that made it. Every `_free` is fine with NULL.

// What each function asks of its pointers is in its doc comment, which is waifu2d.h's.
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{self, AssertUnwindSafe};
use std::ptr::{self, NonNull};
use std::sync::Arc;

use crate::mouths::Viseme;
use crate::render::{self, Api, Renderer, Surface};
use crate::rig::Rig;
use crate::stage::Stage;

// -- versions ---------------------------------------------------------------------------------

/// The major version: bumped for whatever an older caller would misread -- a number that means
/// something else, a function or a type that changed or went, a default.
pub const WAIFU2D_ABI_VERSION_MAJOR: u32 = 1;

/// The minor version: bumped for what an older caller never uses -- a switch, a control, an
/// action, a function. Back to 0 with each major.
///
/// 1.1: the mouth shapes and a reading said -- waifu2d_stage_set_viseme, waifu2d_stage_speak and
/// the waifu2d_stage_speech_ functions, waifu2d_stage_silence. Provisional: they may change in a
/// later minor, before any release has them.
pub const WAIFU2D_ABI_VERSION_MINOR: u32 = 1;

/// (major << 16) | minor, as waifu2d_abi_version says it.
pub const WAIFU2D_ABI_VERSION: u32 = (WAIFU2D_ABI_VERSION_MAJOR << 16) | WAIFU2D_ABI_VERSION_MINOR;

/// The ABI this library was built with: WAIFU2D_ABI_VERSION. A caller whose major differs must
/// not use it.
#[no_mangle]
pub extern "C" fn waifu2d_abi_version() -> u32 {
    WAIFU2D_ABI_VERSION
}

/// Why the last call on this thread failed; valid until the next failing call on it. "" when
/// none has.
#[no_mangle]
pub extern "C" fn waifu2d_last_error() -> *const c_char {
    LAST_ERROR.with(|last| last.borrow().as_ptr())
}

// -- the rig ----------------------------------------------------------------------------------

/// A drawing in layers, read and rigged: never changed, so it may be used from any thread. A
/// stage made from it keeps it alive, so the caller's handle may be freed at once.
pub struct Waifu2dRig {
    rig: Arc<Rig>,
    notes: Vec<CString>,
}

/// What can be asked of a rig: each an integer.
pub type Waifu2dRigAttribute = i32;
/// The canvas's width, in its pixels.
pub const WAIFU2D_RIG_WIDTH: Waifu2dRigAttribute = 0;
/// The canvas's height, in its pixels.
pub const WAIFU2D_RIG_HEIGHT: Waifu2dRigAttribute = 1;
/// How many layers it draws.
pub const WAIFU2D_RIG_PART_COUNT: Waifu2dRigAttribute = 2;
/// How many of its eyes blink: 0 to 2.
pub const WAIFU2D_RIG_EYE_COUNT: Waifu2dRigAttribute = 3;
/// How many notes there are for waifu2d_rig_note.
pub const WAIFU2D_RIG_NOTE_COUNT: Waifu2dRigAttribute = 4;

/// Reads the Photoshop document at `psd_path` (UTF-8) and rigs it. Slow -- it reads every layer
/// -- so call it off the UI thread. NULL, with waifu2d_last_error saying why, where it cannot.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_rig_open(psd_path: *const c_char) -> *mut Waifu2dRig {
    caught(ptr::null_mut(), || {
        let path = match string_in(psd_path, "psd_path") {
            Ok(path) => path,
            Err(why) => return fail(why, ptr::null_mut()),
        };
        match std::fs::read(&path) {
            Ok(bytes) => rig_out(Rig::read_psd(&bytes)),
            Err(error) => fail(format!("{path}: {error}"), ptr::null_mut()),
        }
    })
}

/// A Photoshop document already in memory, `length` bytes at `bytes`, read and rigged; as
/// waifu2d_rig_open. The bytes are the caller's again once it returns.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_rig_read(bytes: *const u8, length: usize) -> *mut Waifu2dRig {
    caught(ptr::null_mut(), || {
        if bytes.is_null() {
            return fail("bytes is NULL".into(), ptr::null_mut());
        }
        rig_out(Rig::read_psd(std::slice::from_raw_parts(bytes, length)))
    })
}

/// NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_rig_free(rig: *mut Waifu2dRig) {
    if !rig.is_null() {
        drop(Box::from_raw(rig));
    }
}

/// Writes `attribute` of `rig` to `value`. False, and nothing written, for an attribute this
/// library does not know.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_rig_get_int_attribute(
    rig: *const Waifu2dRig,
    attribute: Waifu2dRigAttribute,
    value: *mut i64,
) -> bool {
    caught(false, || {
        let (Some(rig), false) = (rig.as_ref(), value.is_null()) else {
            return fail("rig or value is NULL".into(), false);
        };
        let r = &rig.rig;
        let answer = match attribute {
            WAIFU2D_RIG_WIDTH => r.width as i64,
            WAIFU2D_RIG_HEIGHT => r.height as i64,
            WAIFU2D_RIG_PART_COUNT => r.parts.len() as i64,
            WAIFU2D_RIG_EYE_COUNT => r.eyes.iter().flatten().count() as i64,
            WAIFU2D_RIG_NOTE_COUNT => rig.notes.len() as i64,
            other => return fail(format!("no rig attribute {other}"), false),
        };
        *value = answer;
        true
    })
}

/// Note `index` about what the rig could not do, in words to show under the picture: "No eye
/// whites, so the eyes do not blink." Valid while the rig is. NULL past the last.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_rig_note(rig: *const Waifu2dRig, index: usize) -> *const c_char {
    rig.as_ref()
        .and_then(|rig| rig.notes.get(index))
        .map_or(ptr::null(), |note| note.as_ptr())
}

// -- the stage --------------------------------------------------------------------------------

/// A rig moving: its motion, its settings, the pointer over it. One thread at a time.
pub struct Waifu2dStage {
    stage: Stage,
}

/// What moves on its own: each on or off. The default in brackets.
pub type Waifu2dSwitch = i32;
/// Turns the head and looks toward the pointer over the drawing. [on]
pub const WAIFU2D_SWITCH_FOLLOW_POINTER: Waifu2dSwitch = 0;
/// Sways a little all the time. [on]
pub const WAIFU2D_SWITCH_IDLE: Waifu2dSwitch = 1;
/// Looks somewhere else every 1.4 to 4 seconds while not following the pointer. [on]
pub const WAIFU2D_SWITCH_LOOK_AROUND: Waifu2dSwitch = 2;
/// The hair swings in strands; earrings and tails sway. [on]
pub const WAIFU2D_SWITCH_PHYSICS: Waifu2dSwitch = 3;
/// Blinks every 1.6 to 5.4 seconds, now and then twice. [on]
pub const WAIFU2D_SWITCH_BLINK: Waifu2dSwitch = 4;
/// The wind in the hair. [on]
pub const WAIFU2D_SWITCH_WIND: Waifu2dSwitch = 5;
/// The mouth babbles, as if talking, with no sound. [off]
pub const WAIFU2D_SWITCH_AUTO_TALK: Waifu2dSwitch = 6;
/// The wind comes in gusts over WAIFU2D_CONTROL_WIND_STRENGTH, and swings a skirt's hem: the
/// Windows app's wind. [off]
pub const WAIFU2D_SWITCH_GUSTS: Waifu2dSwitch = 7;

/// What the settings say, each a number in a range. The default and the range in brackets.
pub type Waifu2dControl = i32;
/// As open as the eyes are when they are not blinking: 1 as drawn, 0 shut. [1, 0..1]
pub const WAIFU2D_CONTROL_EYES_OPEN: Waifu2dControl = 0;
/// How soon the open eye gives way to the closed one; higher shows it sooner. [0.3, 0..1]
pub const WAIFU2D_CONTROL_CLOSES_AT: Waifu2dControl = 1;
/// How far the head turns up and down: 1 as the rig turns it, 0 not at all. [1, 0..2]
pub const WAIFU2D_CONTROL_NOD: Waifu2dControl = 2;
/// How far the body turns and leans after the head. [1, 0..2]
pub const WAIFU2D_CONTROL_BODY_FOLLOWS: Waifu2dControl = 3;
/// How loud the talk is -- the host's microphone, measured by the host. [0, 0..1]
pub const WAIFU2D_CONTROL_TALK_LEVEL: Waifu2dControl = 4;
/// The mouth as open as this, whatever the talk. [0, 0..1]
pub const WAIFU2D_CONTROL_MOUTH_OPEN: Waifu2dControl = 5;
/// The mouth's corners up, or down. [0, -1..1]
pub const WAIFU2D_CONTROL_SMILE: Waifu2dControl = 6;
/// The brows up, or down. [0, -1..1]
pub const WAIFU2D_CONTROL_BROWS: Waifu2dControl = 7;
/// The upper body twisted one way or the other. [0, -1..1]
pub const WAIFU2D_CONTROL_TWIST: Waifu2dControl = 8;
/// How far the picture's left eye is winked shut. [0, 0..1]
pub const WAIFU2D_CONTROL_WINK_LEFT: Waifu2dControl = 9;
/// How far the picture's right eye is winked shut. [0, 0..1]
pub const WAIFU2D_CONTROL_WINK_RIGHT: Waifu2dControl = 10;
/// How gusty the wind is. [1, 0..3]
pub const WAIFU2D_CONTROL_WIND_STRENGTH: Waifu2dControl = 11;

/// What it can be told to do once.
pub type Waifu2dAction = i32;
/// Blinks now, whatever the clock says.
pub const WAIFU2D_ACTION_BLINK: Waifu2dAction = 0;
/// Hops, and lands with a squash.
pub const WAIFU2D_ACTION_JUMP: Waifu2dAction = 1;
/// Shakes the head once.
pub const WAIFU2D_ACTION_SHAKE: Waifu2dAction = 2;
/// Nods once.
pub const WAIFU2D_ACTION_NOD: Waifu2dAction = 3;

/// A stage for `rig`, with every switch and control at its default. `seed` is where its looking
/// around and its blinks come from: any number, the same one for the same moves.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_new(rig: *const Waifu2dRig, seed: u64) -> *mut Waifu2dStage {
    caught(ptr::null_mut(), || match rig.as_ref() {
        Some(rig) => Box::into_raw(Box::new(Waifu2dStage {
            stage: Stage::new(rig.rig.clone(), seed),
        })),
        None => fail("rig is NULL".into(), ptr::null_mut()),
    })
}

/// NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_free(stage: *mut Waifu2dStage) {
    if !stage.is_null() {
        drop(Box::from_raw(stage));
    }
}

/// False, and nothing changed, for a switch this library does not know.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_set_switch(
    stage: *mut Waifu2dStage,
    which: Waifu2dSwitch,
    on: bool,
) -> bool {
    caught(false, || {
        let Some(stage) = stage.as_mut() else {
            return fail("stage is NULL".into(), false);
        };
        let s = &mut stage.stage.controls_mut().switches;
        let field = match which {
            WAIFU2D_SWITCH_FOLLOW_POINTER => &mut s.follow_pointer,
            WAIFU2D_SWITCH_IDLE => &mut s.idle,
            WAIFU2D_SWITCH_LOOK_AROUND => &mut s.look_around,
            WAIFU2D_SWITCH_PHYSICS => &mut s.physics,
            WAIFU2D_SWITCH_BLINK => &mut s.blink,
            WAIFU2D_SWITCH_WIND => &mut s.wind,
            WAIFU2D_SWITCH_AUTO_TALK => &mut s.auto_talk,
            WAIFU2D_SWITCH_GUSTS => &mut s.gusts,
            other => return fail(format!("no switch {other}"), false),
        };
        *field = on;
        true
    })
}

/// False, and nothing changed, for a control this library does not know or a value outside its
/// range (NaN included).
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_set_control(
    stage: *mut Waifu2dStage,
    which: Waifu2dControl,
    value: f32,
) -> bool {
    caught(false, || {
        let Some(stage) = stage.as_mut() else {
            return fail("stage is NULL".into(), false);
        };
        let c = stage.stage.controls_mut();
        let (field, lo, hi) = match which {
            WAIFU2D_CONTROL_EYES_OPEN => (&mut c.eyes_open, 0.0, 1.0),
            WAIFU2D_CONTROL_CLOSES_AT => (&mut c.closes_at, 0.0, 1.0),
            WAIFU2D_CONTROL_NOD => (&mut c.nod, 0.0, 2.0),
            WAIFU2D_CONTROL_BODY_FOLLOWS => (&mut c.body_follows, 0.0, 2.0),
            WAIFU2D_CONTROL_TALK_LEVEL => (&mut c.talk_level, 0.0, 1.0),
            WAIFU2D_CONTROL_MOUTH_OPEN => (&mut c.mouth_open, 0.0, 1.0),
            WAIFU2D_CONTROL_SMILE => (&mut c.smile, -1.0, 1.0),
            WAIFU2D_CONTROL_BROWS => (&mut c.brows, -1.0, 1.0),
            WAIFU2D_CONTROL_TWIST => (&mut c.twist, -1.0, 1.0),
            WAIFU2D_CONTROL_WINK_LEFT => (&mut c.wink_left, 0.0, 1.0),
            WAIFU2D_CONTROL_WINK_RIGHT => (&mut c.wink_right, 0.0, 1.0),
            WAIFU2D_CONTROL_WIND_STRENGTH => (&mut c.wind_strength, 0.0, 3.0),
            other => return fail(format!("no control {other}"), false),
        };
        if !(lo..=hi).contains(&value) {
            return fail(
                format!("control {which} is {lo} to {hi}, not {value}"),
                false,
            );
        }
        *field = value;
        true
    })
}

/// False for an action this library does not know.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_act(
    stage: *mut Waifu2dStage,
    action: Waifu2dAction,
) -> bool {
    caught(false, || {
        let Some(stage) = stage.as_mut() else {
            return fail("stage is NULL".into(), false);
        };
        let stage = &mut stage.stage;
        match action {
            WAIFU2D_ACTION_BLINK => stage.blink_now(),
            WAIFU2D_ACTION_JUMP => stage.jump(),
            WAIFU2D_ACTION_SHAKE => stage.shake(),
            WAIFU2D_ACTION_NOD => stage.nod_once(),
            other => return fail(format!("no action {other}"), false),
        }
        true
    })
}

/// The view the drawing is fitted into, in the unit the pointer comes in -- points, DIPs, not
/// necessarily pixels.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_set_view_size(
    stage: *mut Waifu2dStage,
    width: f32,
    height: f32,
) {
    with_stage(stage, |stage| stage.set_view_size(width, height));
}

/// The pointer at (x, y) in the view: origin top left, y down. A host whose y is up flips it.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_pointer_moved(stage: *mut Waifu2dStage, x: f32, y: f32) {
    with_stage(stage, |stage| stage.pointer_moved(x, y));
}

/// The pointer has left the view.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_pointer_left(stage: *mut Waifu2dStage) {
    with_stage(stage, |stage| stage.pointer_left());
}

/// A press at (x, y): true where it took hold of a part, which follows the pointer until
/// waifu2d_stage_pointer_up.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_pointer_down(
    stage: *mut Waifu2dStage,
    x: f32,
    y: f32,
) -> bool {
    with_stage(stage, |stage| stage.pointer_down(x, y)).unwrap_or(false)
}

/// Lets go of what a press took hold of.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_pointer_up(stage: *mut Waifu2dStage) {
    with_stage(stage, |stage| stage.pointer_up());
}

/// A mouth shape, as libwaifu's text to speech says them: WaifuViseme, number for number.
/// **Provisional**: here to try the mouth shapes waifu2d draws; its design is still to be reviewed.
pub type Waifu2dViseme = i32;
pub const WAIFU2D_VISEME_A: Waifu2dViseme = 0;
pub const WAIFU2D_VISEME_E: Waifu2dViseme = 1;
pub const WAIFU2D_VISEME_I: Waifu2dViseme = 2;
pub const WAIFU2D_VISEME_O: Waifu2dViseme = 3;
pub const WAIFU2D_VISEME_U: Waifu2dViseme = 4;
pub const WAIFU2D_VISEME_SCHWA: Waifu2dViseme = 5;
pub const WAIFU2D_VISEME_MBP: Waifu2dViseme = 6;
/// The drawn mouth, as it is at rest.
pub const WAIFU2D_VISEME_REST: Waifu2dViseme = 7;
/// None: the mouth as at rest.
pub const WAIFU2D_VISEME_NONE: Waifu2dViseme = -1;

/// The mouth shape to make from now on, eased into over about 50 ms; WAIFU2D_VISEME_NONE for the
/// drawn mouth. False for a number this library does not know. **Provisional.**
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_set_viseme(
    stage: *mut Waifu2dStage,
    viseme: Waifu2dViseme,
) -> bool {
    let shape = match viseme {
        WAIFU2D_VISEME_NONE => None,
        other => match Viseme::from_index(other) {
            Some(v) => Some(v),
            None => return fail(format!("no viseme {other}"), false),
        },
    };
    with_stage(stage, |stage| stage.set_viseme(shape)).is_some()
}

/// A reading for the mouth to say from now on, at `sample_rate` samples a second, with none of
/// its sound yet; it replaces the one before. The mouth is as open as the sound is loud, in the
/// shape its voice said, where waifu2d_stage_speech_at says the sound is heard. **Provisional.**
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_speak(stage: *mut Waifu2dStage, sample_rate: u32) -> bool {
    if sample_rate == 0 {
        return fail("a sample rate of 0".into(), false);
    }
    with_stage(stage, |stage| stage.speak(sample_rate)).is_some()
}

/// The reading's next `count` samples, mono, -1 to 1: all of it at once, or a streaming reading's
/// parts as they come. Copied. **Provisional.**
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_speech_samples(
    stage: *mut Waifu2dStage,
    samples: *const f32,
    count: usize,
) -> bool {
    if samples.is_null() && count > 0 {
        return fail("samples is NULL".into(), false);
    }
    let samples = if count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(samples, count)
    };
    with_stage(stage, |stage| stage.speech_samples(samples)).is_some()
}

/// How loud the reading is, in dB, a frame of `frame_ms` each, frame i the milliseconds
/// [i, i + 1) * frame_ms -- libwaifu's TEXT_TO_SPEECH_LOUDNESS, as it hands it over: all of it so
/// far, again as more is settled. Once it is given the samples are not listened to, and need not
/// be handed over; without it the loudness is heard from them. Copied. **Provisional.**
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_speech_loudness(
    stage: *mut Waifu2dStage,
    db: *const f32,
    count: usize,
    frame_ms: u32,
) -> bool {
    if (db.is_null() && count > 0) || frame_ms == 0 {
        return fail("db is NULL, or a frame of 0 ms".into(), false);
    }
    let db = if count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(db, count)
    };
    with_stage(stage, |stage| stage.speech_loudness(db, frame_ms)).is_some()
}

/// The shapes the reading's voice said -- WaifuViseme's numbers, as libwaifu's text to speech
/// hands them over -- every one from its start, a frame of `frame_ms` each: all of them so far,
/// again as more are settled. **Provisional.**
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_speech_visemes(
    stage: *mut Waifu2dStage,
    visemes: *const i32,
    count: usize,
    frame_ms: u32,
) -> bool {
    if (visemes.is_null() && count > 0) || frame_ms == 0 {
        return fail("visemes is NULL, or a frame of 0 ms".into(), false);
    }
    let visemes = if count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(visemes, count)
    };
    with_stage(stage, |stage| stage.speech_visemes(visemes, frame_ms)).is_some()
}

/// Where in the reading the sound being heard is, in seconds -- what the speakers play now, their
/// latency taken off. Negative, or NaN, where none of it is: paused, or not begun, and the mouth
/// at rest. **Provisional.**
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_speech_at(stage: *mut Waifu2dStage, seconds: f64) {
    let at = (seconds.is_finite() && seconds >= 0.0).then_some(seconds);
    with_stage(stage, |stage| stage.speech_at(at));
}

/// No reading any more: the mouth at rest. **Provisional.**
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_silence(stage: *mut Waifu2dStage) {
    with_stage(stage, |stage| stage.silence());
}

/// Moves on by `dt` seconds, at most 0.05 of them counted: 0 for the first frame.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_stage_step(stage: *mut Waifu2dStage, dt: f64) {
    let dt = if dt.is_finite() { dt } else { 0.0 };
    with_stage(stage, |stage| stage.step(dt));
}

// -- the renderer -----------------------------------------------------------------------------

/// One backend drawing into one surface. Made, drawn with and freed on the thread the surface
/// belongs to.
pub struct Waifu2dRenderer {
    renderer: Renderer,
}

/// Which API draws.
pub type Waifu2dApi = i32;
/// Metal on macOS, Direct3D 11 on Windows, Vulkan elsewhere.
pub const WAIFU2D_API_DEFAULT: Waifu2dApi = 0;
/// The reference: on the CPU, offscreen only.
pub const WAIFU2D_API_CPU: Waifu2dApi = 1;
pub const WAIFU2D_API_METAL: Waifu2dApi = 2;
pub const WAIFU2D_API_D3D11: Waifu2dApi = 3;
pub const WAIFU2D_API_VULKAN: Waifu2dApi = 4;

/// What a renderer draws into.
pub type Waifu2dSurfaceKind = i32;
/// A picture the renderer keeps, `width` by `height`, read back with waifu2d_renderer_read_rgba.
pub const WAIFU2D_SURFACE_OFFSCREEN: Waifu2dSurfaceKind = 0;
/// A CAMetalLayer the host owns: `handle` is the CAMetalLayer*. Its device, pixel format and
/// drawable size become the renderer's.
pub const WAIFU2D_SURFACE_METAL_LAYER: Waifu2dSurfaceKind = 1;
/// A window's client area: `handle` is the HWND, `handle2` its HINSTANCE.
pub const WAIFU2D_SURFACE_WIN32_WINDOW: Waifu2dSurfaceKind = 2;
/// A texture shared with a D3D9Ex device, for a WPF D3DImage: see
/// waifu2d_renderer_shared_handle.
pub const WAIFU2D_SURFACE_D3D11_SHARED: Waifu2dSurfaceKind = 3;

/// A surface, as waifu2d_renderer_new is handed it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Waifu2dSurface {
    pub kind: Waifu2dSurfaceKind,
    pub handle: *mut c_void,
    pub handle2: *mut c_void,
    /// Pixels. For a layer or a window, its size now, or 0 and 0 to leave it to
    /// waifu2d_renderer_resize.
    pub width: u32,
    pub height: u32,
}

/// Whether this build has `api` and this machine can run it.
#[no_mangle]
pub extern "C" fn waifu2d_api_is_available(api: Waifu2dApi) -> bool {
    caught(false, || api_in(api).is_some_and(render::is_available))
}

/// A renderer for `surface` through `api`. NULL, with waifu2d_last_error saying why, where the
/// API is not there or will not draw into that kind of surface.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_renderer_new(
    api: Waifu2dApi,
    surface: *const Waifu2dSurface,
) -> *mut Waifu2dRenderer {
    caught(ptr::null_mut(), || {
        let Some(api) = api_in(api) else {
            return fail(format!("no API {api}"), ptr::null_mut());
        };
        let Some(surface) = surface.as_ref() else {
            return fail("surface is NULL".into(), ptr::null_mut());
        };
        let (target, size) = match surface_in(surface) {
            Ok(it) => it,
            Err(why) => return fail(why, ptr::null_mut()),
        };
        let mut renderer = match Renderer::new(api, target) {
            Ok(renderer) => renderer,
            Err(error) => return fail(error.to_string(), ptr::null_mut()),
        };
        if let Some([w, h]) = size {
            if let Err(error) = renderer.resize(w, h) {
                return fail(error.to_string(), ptr::null_mut());
            }
        }
        Box::into_raw(Box::new(Waifu2dRenderer { renderer }))
    })
}

/// NULL is fine.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_renderer_free(renderer: *mut Waifu2dRenderer) {
    if !renderer.is_null() {
        drop(Box::from_raw(renderer));
    }
}

/// The API it draws with: what WAIFU2D_API_DEFAULT became. -1 for NULL.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_renderer_api(renderer: *const Waifu2dRenderer) -> Waifu2dApi {
    match renderer.as_ref().map(|r| r.renderer.api()) {
        Some(Api::Cpu) => WAIFU2D_API_CPU,
        Some(Api::Metal) => WAIFU2D_API_METAL,
        Some(Api::D3D11) => WAIFU2D_API_D3D11,
        Some(Api::Vulkan) => WAIFU2D_API_VULKAN,
        None => -1,
    }
}

/// The size of what it draws into, in pixels. Cheap when it is unchanged, so a host may call it
/// every frame. Neither may be 0.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_renderer_resize(
    renderer: *mut Waifu2dRenderer,
    width: u32,
    height: u32,
) -> bool {
    caught(false, || match renderer.as_mut() {
        Some(r) => result(r.renderer.resize(width, height)),
        None => fail("renderer is NULL".into(), false),
    })
}

/// Draws `stage` as it is now over `clear` -- premultiplied RGBA, 0 to 1, or NULL for
/// transparent -- and presents it. The first draw of a stage's rig loads its pictures, and is
/// slow; so is the first after the stage is another rig's.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_renderer_draw(
    renderer: *mut Waifu2dRenderer,
    stage: *const Waifu2dStage,
    clear: *const f32,
) -> bool {
    caught(false, || {
        let (Some(r), Some(stage)) = (renderer.as_mut(), stage.as_ref()) else {
            return fail("renderer or stage is NULL".into(), false);
        };
        let clear = if clear.is_null() {
            [0.0; 4]
        } else {
            let c = std::slice::from_raw_parts(clear, 4);
            [c[0], c[1], c[2], c[3]]
        };
        result(r.renderer.draw(&stage.stage, clear))
    })
}

/// An offscreen renderer's last draw, premultiplied RGBA8, top row first, into `out`, which is
/// `length` = width * height * 4 bytes.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_renderer_read_rgba(
    renderer: *mut Waifu2dRenderer,
    out: *mut u8,
    length: usize,
) -> bool {
    caught(false, || {
        let (Some(r), false) = (renderer.as_mut(), out.is_null()) else {
            return fail("renderer or out is NULL".into(), false);
        };
        result(
            r.renderer
                .read_rgba(std::slice::from_raw_parts_mut(out, length)),
        )
    })
}

/// For WAIFU2D_SURFACE_D3D11_SHARED: the DXGI shared handle of the texture last drawn into, for
/// a D3D9Ex device to open and a D3DImage to show. NULL for any other surface.
#[no_mangle]
pub unsafe extern "C" fn waifu2d_renderer_shared_handle(
    renderer: *const Waifu2dRenderer,
) -> *mut c_void {
    renderer
        .as_ref()
        .and_then(|r| r.renderer.shared_handle())
        .map_or(ptr::null_mut(), NonNull::as_ptr)
}

// -- support ----------------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Says why this thread's call failed, and returns `otherwise` for it to return.
fn fail<T>(message: String, otherwise: T) -> T {
    let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|last| *last.borrow_mut() = message);
    otherwise
}

fn result(r: Result<(), crate::Error>) -> bool {
    match r {
        Ok(()) => true,
        Err(error) => fail(error.to_string(), false),
    }
}

/// Runs an entry point, turning a panic into its message in the last error and `otherwise` as
/// what the call returns: a panic must not unwind into C.
fn caught<T>(otherwise: T, body: impl FnOnce() -> T) -> T {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(cause) => {
            let said = cause
                .downcast_ref::<&str>()
                .map(|said| said.to_string())
                .or_else(|| cause.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "it panicked".to_string());
            fail(format!("a bug in the library: {said}"), otherwise)
        }
    }
}

unsafe fn with_stage<T>(stage: *mut Waifu2dStage, body: impl FnOnce(&mut Stage) -> T) -> Option<T> {
    caught(None, || match stage.as_mut() {
        Some(stage) => Some(body(&mut stage.stage)),
        None => fail("stage is NULL".into(), None),
    })
}

unsafe fn string_in(pointer: *const c_char, what: &str) -> Result<String, String> {
    if pointer.is_null() {
        return Err(format!("{what} is NULL"));
    }
    CStr::from_ptr(pointer)
        .to_str()
        .map(str::to_string)
        .map_err(|_| format!("{what} is not UTF-8"))
}

fn rig_out(rig: Result<Rig, crate::Error>) -> *mut Waifu2dRig {
    match rig {
        Ok(rig) => {
            // The mouth shapes it has no layers for, drawn: hidden until a viseme is set.
            // Provisional, with waifu2d_stage_set_viseme.
            let rig = crate::mouths::with_drawn_mouths(rig);
            let notes = rig
                .notes
                .iter()
                .map(|note| CString::new(note.replace('\0', " ")).unwrap_or_default())
                .collect();
            Box::into_raw(Box::new(Waifu2dRig {
                rig: Arc::new(rig),
                notes,
            }))
        }
        Err(error) => fail(error.to_string(), ptr::null_mut()),
    }
}

fn api_in(api: Waifu2dApi) -> Option<Api> {
    match api {
        WAIFU2D_API_DEFAULT => Some(render::default_api()),
        WAIFU2D_API_CPU => Some(Api::Cpu),
        WAIFU2D_API_METAL => Some(Api::Metal),
        WAIFU2D_API_D3D11 => Some(Api::D3D11),
        WAIFU2D_API_VULKAN => Some(Api::Vulkan),
        _ => None,
    }
}

/// The surface, and the size to give a layer or a window at once if it came with one.
fn surface_in(s: &Waifu2dSurface) -> Result<(Surface, Option<[u32; 2]>), String> {
    let size = (s.width > 0 && s.height > 0).then_some([s.width, s.height]);
    let handle =
        |p: *mut c_void, what: &str| NonNull::new(p).ok_or_else(|| format!("{what} is NULL"));
    match s.kind {
        WAIFU2D_SURFACE_OFFSCREEN | WAIFU2D_SURFACE_D3D11_SHARED => {
            let [width, height] = size.ok_or("an offscreen surface needs a width and a height")?;
            let surface = if s.kind == WAIFU2D_SURFACE_OFFSCREEN {
                Surface::Offscreen { width, height }
            } else {
                Surface::D3D11Shared { width, height }
            };
            Ok((surface, None))
        }
        WAIFU2D_SURFACE_METAL_LAYER => Ok((
            Surface::MetalLayer {
                layer: handle(s.handle, "the CAMetalLayer")?,
            },
            size,
        )),
        WAIFU2D_SURFACE_WIN32_WINDOW => Ok((
            Surface::Win32Window {
                hwnd: handle(s.handle, "the HWND")?,
                hinstance: handle(s.handle2, "the HINSTANCE")?,
            },
            size,
        )),
        other => Err(format!("no surface kind {other}")),
    }
}
