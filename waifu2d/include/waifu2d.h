/* The MIT License (MIT). Copyright (c) 2026 Xiaoyang Chen. See LICENSE. */

#ifndef LIBWAIFU_WAIFU2D_H
#define LIBWAIFU_WAIFU2D_H

/* Written by cbindgen from waifu2d/src/c_api.rs. Change that, not this. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// The major version: bumped for whatever an older caller would misread -- a number that means
// something else, a function or a type that changed or went, a default.
#define WAIFU2D_ABI_VERSION_MAJOR 1

// The minor version: bumped for what an older caller never uses -- a switch, a control, an
// action, a function. Back to 0 with each major.
#define WAIFU2D_ABI_VERSION_MINOR 0

// (major << 16) | minor, as waifu2d_abi_version says it.
#define WAIFU2D_ABI_VERSION ((WAIFU2D_ABI_VERSION_MAJOR << 16) | WAIFU2D_ABI_VERSION_MINOR)

// One backend drawing into one surface. Made, drawn with and freed on the thread the surface
// belongs to.
typedef struct Waifu2dRenderer Waifu2dRenderer;

// A drawing in layers, read and rigged: never changed, so it may be used from any thread. A
// stage made from it keeps it alive, so the caller's handle may be freed at once.
typedef struct Waifu2dRig Waifu2dRig;

// A rig moving: its motion, its settings, the pointer over it. One thread at a time.
typedef struct Waifu2dStage Waifu2dStage;

// What can be asked of a rig: each an integer.
typedef int32_t Waifu2dRigAttribute;

// What moves on its own: each on or off. The default in brackets.
typedef int32_t Waifu2dSwitch;

// What the settings say, each a number in a range. The default and the range in brackets.
typedef int32_t Waifu2dControl;

// What it can be told to do once.
typedef int32_t Waifu2dAction;

// A mouth shape, as libwaifu's text to speech says them: WaifuViseme, number for number.
// **Provisional**: here to try the mouth shapes waifu2d draws; its design is still to be reviewed.
typedef int32_t Waifu2dViseme;

// Which API draws.
typedef int32_t Waifu2dApi;

// What a renderer draws into.
typedef int32_t Waifu2dSurfaceKind;

// A surface, as waifu2d_renderer_new is handed it.
typedef struct {
  Waifu2dSurfaceKind kind;
  void *handle;
  void *handle2;
  // Pixels. For a layer or a window, its size now, or 0 and 0 to leave it to
  // waifu2d_renderer_resize.
  uint32_t width;
  uint32_t height;
} Waifu2dSurface;

// The canvas's width, in its pixels.
#define WAIFU2D_RIG_WIDTH 0

// The canvas's height, in its pixels.
#define WAIFU2D_RIG_HEIGHT 1

// How many layers it draws.
#define WAIFU2D_RIG_PART_COUNT 2

// How many of its eyes blink: 0 to 2.
#define WAIFU2D_RIG_EYE_COUNT 3

// How many notes there are for waifu2d_rig_note.
#define WAIFU2D_RIG_NOTE_COUNT 4

// Turns the head and looks toward the pointer over the drawing. [on]
#define WAIFU2D_SWITCH_FOLLOW_POINTER 0

// Sways a little all the time. [on]
#define WAIFU2D_SWITCH_IDLE 1

// Looks somewhere else every 1.4 to 4 seconds while not following the pointer. [on]
#define WAIFU2D_SWITCH_LOOK_AROUND 2

// The hair swings in strands; earrings and tails sway. [on]
#define WAIFU2D_SWITCH_PHYSICS 3

// Blinks every 1.6 to 5.4 seconds, now and then twice. [on]
#define WAIFU2D_SWITCH_BLINK 4

// The wind in the hair. [on]
#define WAIFU2D_SWITCH_WIND 5

// The mouth babbles, as if talking, with no sound. [off]
#define WAIFU2D_SWITCH_AUTO_TALK 6

// The wind comes in gusts over WAIFU2D_CONTROL_WIND_STRENGTH, and swings a skirt's hem: the
// Windows app's wind. [off]
#define WAIFU2D_SWITCH_GUSTS 7

// As open as the eyes are when they are not blinking: 1 as drawn, 0 shut. [1, 0..1]
#define WAIFU2D_CONTROL_EYES_OPEN 0

// How soon the open eye gives way to the closed one; higher shows it sooner. [0.3, 0..1]
#define WAIFU2D_CONTROL_CLOSES_AT 1

// How far the head turns up and down: 1 as the rig turns it, 0 not at all. [1, 0..2]
#define WAIFU2D_CONTROL_NOD 2

// How far the body turns and leans after the head. [1, 0..2]
#define WAIFU2D_CONTROL_BODY_FOLLOWS 3

// How loud the talk is -- the host's microphone, measured by the host. [0, 0..1]
#define WAIFU2D_CONTROL_TALK_LEVEL 4

// The mouth as open as this, whatever the talk. [0, 0..1]
#define WAIFU2D_CONTROL_MOUTH_OPEN 5

// The mouth's corners up, or down. [0, -1..1]
#define WAIFU2D_CONTROL_SMILE 6

// The brows up, or down. [0, -1..1]
#define WAIFU2D_CONTROL_BROWS 7

// The upper body twisted one way or the other. [0, -1..1]
#define WAIFU2D_CONTROL_TWIST 8

// How far the picture's left eye is winked shut. [0, 0..1]
#define WAIFU2D_CONTROL_WINK_LEFT 9

// How far the picture's right eye is winked shut. [0, 0..1]
#define WAIFU2D_CONTROL_WINK_RIGHT 10

// How gusty the wind is. [1, 0..3]
#define WAIFU2D_CONTROL_WIND_STRENGTH 11

// Blinks now, whatever the clock says.
#define WAIFU2D_ACTION_BLINK 0

// Hops, and lands with a squash.
#define WAIFU2D_ACTION_JUMP 1

// Shakes the head once.
#define WAIFU2D_ACTION_SHAKE 2

// Nods once.
#define WAIFU2D_ACTION_NOD 3

#define WAIFU2D_VISEME_A 0

#define WAIFU2D_VISEME_E 1

#define WAIFU2D_VISEME_I 2

#define WAIFU2D_VISEME_O 3

#define WAIFU2D_VISEME_U 4

#define WAIFU2D_VISEME_SCHWA 5

#define WAIFU2D_VISEME_MBP 6

// The drawn mouth, as it is at rest.
#define WAIFU2D_VISEME_REST 7

// None: the mouth as at rest.
#define WAIFU2D_VISEME_NONE -1

// Metal on macOS, Direct3D 11 on Windows, Vulkan elsewhere.
#define WAIFU2D_API_DEFAULT 0

// The reference: on the CPU, offscreen only.
#define WAIFU2D_API_CPU 1

#define WAIFU2D_API_METAL 2

#define WAIFU2D_API_D3D11 3

#define WAIFU2D_API_VULKAN 4

// A picture the renderer keeps, `width` by `height`, read back with waifu2d_renderer_read_rgba.
#define WAIFU2D_SURFACE_OFFSCREEN 0

// A CAMetalLayer the host owns: `handle` is the CAMetalLayer*. Its device, pixel format and
// drawable size become the renderer's.
#define WAIFU2D_SURFACE_METAL_LAYER 1

// A window's client area: `handle` is the HWND, `handle2` its HINSTANCE.
#define WAIFU2D_SURFACE_WIN32_WINDOW 2

// A texture shared with a D3D9Ex device, for a WPF D3DImage: see
// waifu2d_renderer_shared_handle.
#define WAIFU2D_SURFACE_D3D11_SHARED 3

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// The ABI this library was built with: WAIFU2D_ABI_VERSION. A caller whose major differs must
// not use it.
uint32_t waifu2d_abi_version(void);

// Why the last call on this thread failed; valid until the next failing call on it. "" when
// none has.
const char *waifu2d_last_error(void);

// Reads the Photoshop document at `psd_path` (UTF-8) and rigs it. Slow -- it reads every layer
// -- so call it off the UI thread. NULL, with waifu2d_last_error saying why, where it cannot.
Waifu2dRig *waifu2d_rig_open(const char *psd_path);

// A Photoshop document already in memory, `length` bytes at `bytes`, read and rigged; as
// waifu2d_rig_open. The bytes are the caller's again once it returns.
Waifu2dRig *waifu2d_rig_read(const uint8_t *bytes, size_t length);

// NULL is fine.
void waifu2d_rig_free(Waifu2dRig *rig);

// Writes `attribute` of `rig` to `value`. False, and nothing written, for an attribute this
// library does not know.
bool waifu2d_rig_get_int_attribute(const Waifu2dRig *rig,
                                   Waifu2dRigAttribute attribute,
                                   int64_t *value);

// Note `index` about what the rig could not do, in words to show under the picture: "No eye
// whites, so the eyes do not blink." Valid while the rig is. NULL past the last.
const char *waifu2d_rig_note(const Waifu2dRig *rig, size_t index);

// A stage for `rig`, with every switch and control at its default. `seed` is where its looking
// around and its blinks come from: any number, the same one for the same moves.
Waifu2dStage *waifu2d_stage_new(const Waifu2dRig *rig, uint64_t seed);

// NULL is fine.
void waifu2d_stage_free(Waifu2dStage *stage);

// False, and nothing changed, for a switch this library does not know.
bool waifu2d_stage_set_switch(Waifu2dStage *stage, Waifu2dSwitch which, bool on);

// False, and nothing changed, for a control this library does not know or a value outside its
// range (NaN included).
bool waifu2d_stage_set_control(Waifu2dStage *stage, Waifu2dControl which, float value);

// False for an action this library does not know.
bool waifu2d_stage_act(Waifu2dStage *stage, Waifu2dAction action);

// The view the drawing is fitted into, in the unit the pointer comes in -- points, DIPs, not
// necessarily pixels.
void waifu2d_stage_set_view_size(Waifu2dStage *stage, float width, float height);

// The pointer at (x, y) in the view: origin top left, y down. A host whose y is up flips it.
void waifu2d_stage_pointer_moved(Waifu2dStage *stage, float x, float y);

// The pointer has left the view.
void waifu2d_stage_pointer_left(Waifu2dStage *stage);

// A press at (x, y): true where it took hold of a part, which follows the pointer until
// waifu2d_stage_pointer_up.
bool waifu2d_stage_pointer_down(Waifu2dStage *stage, float x, float y);

// Lets go of what a press took hold of.
void waifu2d_stage_pointer_up(Waifu2dStage *stage);

// The mouth shape to make from now on, eased into over about 50 ms; WAIFU2D_VISEME_NONE for the
// drawn mouth. False for a number this library does not know. **Provisional.**
bool waifu2d_stage_set_viseme(Waifu2dStage *stage, Waifu2dViseme viseme);

// A reading for the mouth to say from now on, at `sample_rate` samples a second, with none of
// its sound yet; it replaces the one before. The mouth is as open as the sound is loud, in the
// shape its voice said, where waifu2d_stage_speech_at says the sound is heard. **Provisional.**
bool waifu2d_stage_speak(Waifu2dStage *stage, uint32_t sample_rate);

// The reading's next `count` samples, mono, -1 to 1: all of it at once, or a streaming reading's
// parts as they come. Copied. **Provisional.**
bool waifu2d_stage_speech_samples(Waifu2dStage *stage, const float *samples, size_t count);

// How loud the reading is, in dB, a frame of `frame_ms` each, frame i the milliseconds
// [i, i + 1) * frame_ms -- libwaifu's TEXT_TO_SPEECH_LOUDNESS, as it hands it over: all of it so
// far, again as more is settled. Once it is given the samples are not listened to, and need not
// be handed over; without it the loudness is heard from them. Copied. **Provisional.**
bool waifu2d_stage_speech_loudness(Waifu2dStage *stage,
                                   const float *db,
                                   size_t count,
                                   uint32_t frame_ms);

// The shapes the reading's voice said -- WaifuViseme's numbers, as libwaifu's text to speech
// hands them over -- every one from its start, a frame of `frame_ms` each: all of them so far,
// again as more are settled. **Provisional.**
bool waifu2d_stage_speech_visemes(Waifu2dStage *stage,
                                  const int32_t *visemes,
                                  size_t count,
                                  uint32_t frame_ms);

// Where in the reading the sound being heard is, in seconds -- what the speakers play now, their
// latency taken off. Negative, or NaN, where none of it is: paused, or not begun, and the mouth
// at rest. **Provisional.**
void waifu2d_stage_speech_at(Waifu2dStage *stage, double seconds);

// No reading any more: the mouth at rest. **Provisional.**
void waifu2d_stage_silence(Waifu2dStage *stage);

// Moves on by `dt` seconds, at most 0.05 of them counted: 0 for the first frame.
void waifu2d_stage_step(Waifu2dStage *stage, double dt);

// Whether this build has `api` and this machine can run it.
bool waifu2d_api_is_available(Waifu2dApi api);

// A renderer for `surface` through `api`. NULL, with waifu2d_last_error saying why, where the
// API is not there or will not draw into that kind of surface.
Waifu2dRenderer *waifu2d_renderer_new(Waifu2dApi api, const Waifu2dSurface *surface);

// NULL is fine.
void waifu2d_renderer_free(Waifu2dRenderer *renderer);

// The API it draws with: what WAIFU2D_API_DEFAULT became. -1 for NULL.
Waifu2dApi waifu2d_renderer_api(const Waifu2dRenderer *renderer);

// The size of what it draws into, in pixels. Cheap when it is unchanged, so a host may call it
// every frame. Neither may be 0.
bool waifu2d_renderer_resize(Waifu2dRenderer *renderer, uint32_t width, uint32_t height);

// Draws `stage` as it is now over `clear` -- premultiplied RGBA, 0 to 1, or NULL for
// transparent -- and presents it. The first draw of a stage's rig loads its pictures, and is
// slow; so is the first after the stage is another rig's.
bool waifu2d_renderer_draw(Waifu2dRenderer *renderer,
                           const Waifu2dStage *stage,
                           const float *clear);

// An offscreen renderer's last draw, premultiplied RGBA8, top row first, into `out`, which is
// `length` = width * height * 4 bytes.
bool waifu2d_renderer_read_rgba(Waifu2dRenderer *renderer, uint8_t *out, size_t length);

// For WAIFU2D_SURFACE_D3D11_SHARED: the DXGI shared handle of the texture last drawn into, for
// a D3D9Ex device to open and a D3DImage to show. NULL for any other surface.
void *waifu2d_renderer_shared_handle(const Waifu2dRenderer *renderer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* LIBWAIFU_WAIFU2D_H */
