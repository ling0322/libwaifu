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

//! What waifu2d.h promises, called as C calls it: NULLs, numbers it does not know, values out of
//! range, a rig outliving its handle, and a frame drawn and read back.

mod common;

use std::ffi::{c_char, CStr, CString};
use std::ptr;

use waifu2d::c_api::*;

fn last_error() -> String {
    unsafe { CStr::from_ptr(waifu2d_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn face() -> *mut Waifu2dRig {
    let bytes = common::face_psd();
    let rig = unsafe { waifu2d_rig_read(bytes.as_ptr(), bytes.len()) };
    assert!(!rig.is_null(), "{}", last_error());
    rig
}

fn attribute(rig: *const Waifu2dRig, which: Waifu2dRigAttribute) -> i64 {
    let mut value = -1;
    assert!(
        unsafe { waifu2d_rig_get_int_attribute(rig, which, &mut value) },
        "{}",
        last_error()
    );
    value
}

fn offscreen(api: Waifu2dApi, width: u32, height: u32) -> *mut Waifu2dRenderer {
    let surface = Waifu2dSurface {
        kind: WAIFU2D_SURFACE_OFFSCREEN,
        handle: ptr::null_mut(),
        handle2: ptr::null_mut(),
        width,
        height,
    };
    unsafe { waifu2d_renderer_new(api, &surface) }
}

#[test]
fn the_version_is_major_and_minor() {
    assert_eq!(waifu2d_abi_version(), WAIFU2D_ABI_VERSION);
    assert_eq!(waifu2d_abi_version() >> 16, WAIFU2D_ABI_VERSION_MAJOR);
    assert_eq!(waifu2d_abi_version() & 0xFFFF, WAIFU2D_ABI_VERSION_MINOR);
}

#[test]
fn a_document_is_read_into_a_rig() {
    let rig = face();
    assert_eq!(attribute(rig, WAIFU2D_RIG_WIDTH), 200);
    assert_eq!(attribute(rig, WAIFU2D_RIG_HEIGHT), 200);
    assert_eq!(attribute(rig, WAIFU2D_RIG_PART_COUNT), 10);
    assert_eq!(attribute(rig, WAIFU2D_RIG_EYE_COUNT), 2);
    assert_eq!(attribute(rig, WAIFU2D_RIG_NOTE_COUNT), 0);
    assert!(unsafe { waifu2d_rig_note(rig, 0) }.is_null());
    let mut value = 7;
    assert!(!unsafe { waifu2d_rig_get_int_attribute(rig, 99, &mut value) });
    assert_eq!(
        value, 7,
        "nothing written for an attribute it does not know"
    );
    assert!(!last_error().is_empty());
    unsafe { waifu2d_rig_free(rig) };
}

#[test]
fn a_rig_without_eyes_has_a_note_to_show() {
    let bytes = common::Writer::document(
        10,
        10,
        &[common::layer("body", (0, 0, 10, 10), [1, 1, 1, 255])],
    );
    let rig = unsafe { waifu2d_rig_read(bytes.as_ptr(), bytes.len()) };
    assert!(!rig.is_null());
    assert_eq!(attribute(rig, WAIFU2D_RIG_NOTE_COUNT), 1);
    let note = unsafe { waifu2d_rig_note(rig, 0) };
    assert!(!note.is_null());
    assert!(!unsafe { CStr::from_ptr(note) }.to_bytes().is_empty());
    unsafe { waifu2d_rig_free(rig) };
}

#[test]
fn what_cannot_be_read_is_null_and_says_why() {
    let junk = b"not a document";
    assert!(unsafe { waifu2d_rig_read(junk.as_ptr(), junk.len()) }.is_null());
    assert!(last_error().contains("Photoshop"), "{}", last_error());

    assert!(unsafe { waifu2d_rig_read(ptr::null(), 0) }.is_null());
    assert!(unsafe { waifu2d_rig_open(ptr::null()) }.is_null());
    let missing = CString::new("/no/such/drawing.psd").unwrap();
    assert!(unsafe { waifu2d_rig_open(missing.as_ptr()) }.is_null());
    assert!(
        last_error().contains("/no/such/drawing.psd"),
        "{}",
        last_error()
    );
    let not_utf8: [c_char; 3] = [0xFFu8 as c_char, 0x41, 0];
    assert!(unsafe { waifu2d_rig_open(not_utf8.as_ptr()) }.is_null());
}

#[test]
fn null_is_fine_to_every_free_and_refused_elsewhere() {
    unsafe {
        waifu2d_rig_free(ptr::null_mut());
        waifu2d_stage_free(ptr::null_mut());
        waifu2d_renderer_free(ptr::null_mut());
        assert!(waifu2d_stage_new(ptr::null(), 1).is_null());
        assert!(!waifu2d_stage_set_switch(
            ptr::null_mut(),
            WAIFU2D_SWITCH_IDLE,
            true
        ));
        assert!(!waifu2d_stage_set_control(
            ptr::null_mut(),
            WAIFU2D_CONTROL_NOD,
            1.0
        ));
        assert!(!waifu2d_stage_act(ptr::null_mut(), WAIFU2D_ACTION_BLINK));
        assert!(!waifu2d_stage_pointer_down(ptr::null_mut(), 0.0, 0.0));
        waifu2d_stage_step(ptr::null_mut(), 0.1);
        waifu2d_stage_pointer_moved(ptr::null_mut(), 0.0, 0.0);
        assert!(waifu2d_renderer_new(WAIFU2D_API_CPU, ptr::null()).is_null());
        assert_eq!(waifu2d_renderer_api(ptr::null()), -1);
        assert!(!waifu2d_renderer_resize(ptr::null_mut(), 1, 1));
        assert!(!waifu2d_renderer_draw(
            ptr::null_mut(),
            ptr::null(),
            ptr::null()
        ));
        assert!(waifu2d_renderer_shared_handle(ptr::null()).is_null());
        assert!(waifu2d_rig_note(ptr::null(), 0).is_null());
    }
}

#[test]
fn numbers_it_does_not_know_and_values_out_of_range_are_refused() {
    let rig = face();
    let stage = unsafe { waifu2d_stage_new(rig, 1) };
    unsafe {
        assert!(waifu2d_stage_set_switch(stage, WAIFU2D_SWITCH_GUSTS, true));
        assert!(!waifu2d_stage_set_switch(stage, 99, true));
        assert!(waifu2d_stage_set_control(
            stage,
            WAIFU2D_CONTROL_SMILE,
            -1.0
        ));
        assert!(waifu2d_stage_set_control(
            stage,
            WAIFU2D_CONTROL_WIND_STRENGTH,
            3.0
        ));
        assert!(!waifu2d_stage_set_control(
            stage,
            WAIFU2D_CONTROL_EYES_OPEN,
            1.5
        ));
        assert!(last_error().contains("0 to 1"), "{}", last_error());
        assert!(!waifu2d_stage_set_control(
            stage,
            WAIFU2D_CONTROL_NOD,
            f32::NAN
        ));
        assert!(!waifu2d_stage_set_control(stage, 99, 0.0));
        assert!(waifu2d_stage_act(stage, WAIFU2D_ACTION_JUMP));
        assert!(!waifu2d_stage_act(stage, 99));
        assert!(!waifu2d_api_is_available(99));
        assert!(waifu2d_api_is_available(WAIFU2D_API_CPU));
        assert!(offscreen(99, 4, 4).is_null());
        assert!(offscreen(WAIFU2D_API_CPU, 0, 4).is_null());
        let layer = Waifu2dSurface {
            kind: WAIFU2D_SURFACE_METAL_LAYER,
            handle: ptr::null_mut(),
            handle2: ptr::null_mut(),
            width: 0,
            height: 0,
        };
        assert!(waifu2d_renderer_new(WAIFU2D_API_DEFAULT, &layer).is_null());
        let unknown = Waifu2dSurface { kind: 99, ..layer };
        assert!(waifu2d_renderer_new(WAIFU2D_API_CPU, &unknown).is_null());
        waifu2d_stage_free(stage);
        waifu2d_rig_free(rig);
    }
}

#[test]
fn a_stage_keeps_its_rig_and_is_drawn_and_read_back() {
    let rig = face();
    let stage = unsafe { waifu2d_stage_new(rig, 1) };
    // The caller's handle may go at once: the stage keeps the rig.
    unsafe { waifu2d_rig_free(rig) };
    unsafe {
        for which in [
            WAIFU2D_SWITCH_FOLLOW_POINTER,
            WAIFU2D_SWITCH_IDLE,
            WAIFU2D_SWITCH_WIND,
            WAIFU2D_SWITCH_LOOK_AROUND,
            WAIFU2D_SWITCH_PHYSICS,
            WAIFU2D_SWITCH_BLINK,
        ] {
            assert!(waifu2d_stage_set_switch(stage, which, false));
        }
        waifu2d_stage_set_view_size(stage, 100.0, 100.0);
        waifu2d_stage_pointer_moved(stage, 50.0, 10.0);
        waifu2d_stage_step(stage, 0.0);
    }

    let renderer = offscreen(WAIFU2D_API_CPU, 200, 200);
    assert!(!renderer.is_null(), "{}", last_error());
    assert_eq!(unsafe { waifu2d_renderer_api(renderer) }, WAIFU2D_API_CPU);
    assert!(unsafe { waifu2d_renderer_shared_handle(renderer) }.is_null());
    assert!(
        unsafe { waifu2d_renderer_draw(renderer, stage, ptr::null()) },
        "{}",
        last_error()
    );

    let mut short = vec![0u8; 10];
    assert!(!unsafe { waifu2d_renderer_read_rgba(renderer, short.as_mut_ptr(), short.len()) });
    let mut pixels = vec![0u8; 200 * 200 * 4];
    assert!(unsafe { waifu2d_renderer_read_rgba(renderer, pixels.as_mut_ptr(), pixels.len()) });
    let at = |x: usize, y: usize| &pixels[(y * 200 + x) * 4..(y * 200 + x) * 4 + 4];
    assert_eq!(at(5, 5), [0, 0, 0, 0], "nothing in the corner");
    assert_eq!(at(100, 140), [250, 220, 200, 255], "the face");

    // Over an opaque clear colour, and at another size.
    assert!(unsafe { waifu2d_renderer_resize(renderer, 100, 100) });
    let white = [1.0f32, 1.0, 1.0, 1.0];
    assert!(
        unsafe { waifu2d_renderer_draw(renderer, stage, white.as_ptr()) },
        "{}",
        last_error()
    );
    let mut small = vec![0u8; 100 * 100 * 4];
    assert!(unsafe { waifu2d_renderer_read_rgba(renderer, small.as_mut_ptr(), small.len()) });
    assert_eq!(&small[..4], [255, 255, 255, 255]);
    assert!(!unsafe { waifu2d_renderer_resize(renderer, 0, 100) });

    unsafe {
        waifu2d_renderer_free(renderer);
        waifu2d_stage_free(stage);
    }
}

#[test]
fn a_press_takes_hold_of_a_part_and_lets_it_go() {
    let rig = face();
    let stage = unsafe { waifu2d_stage_new(rig, 1) };
    unsafe {
        waifu2d_stage_set_view_size(stage, 200.0, 200.0);
        // Nothing in the corner; the hair across the top.
        assert!(!waifu2d_stage_pointer_down(stage, 2.0, 2.0));
        assert!(waifu2d_stage_pointer_down(stage, 100.0, 30.0));
        waifu2d_stage_pointer_moved(stage, 120.0, 40.0);
        waifu2d_stage_step(stage, 1.0 / 60.0);
        waifu2d_stage_pointer_up(stage);
        waifu2d_stage_step(stage, f64::NAN);
        waifu2d_stage_free(stage);
        waifu2d_rig_free(rig);
    }
}
