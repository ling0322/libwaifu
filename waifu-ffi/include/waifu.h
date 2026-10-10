/* The MIT License (MIT). Copyright (c) 2026 Xiaoyang Chen. See LICENSE. */

#ifndef LIBWAIFU_WAIFU_H
#define LIBWAIFU_WAIFU_H

/* Written by cbindgen from waifu-ffi/src/lib.rs. Change that, not this. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// What a caller compiled against this header expects. Bumped when anything in it changes in a
// way an older caller would misread.
#define WAIFU_ABI_VERSION 1

typedef enum {
  WAIFU_LOG_DEBUG = 0,
  WAIFU_LOG_INFO = 1,
  WAIFU_LOG_WARNING = 2,
  WAIFU_LOG_ERROR = 3,
  // The last line: the process aborts when the callback returns.
  WAIFU_LOG_FATAL = 4,
} WaifuLogLevel;

// What a call or a job came to.
typedef enum {
  WAIFU_OK = 0,
  WAIFU_CANCELLED = 1,
  // A NULL, a string that is not UTF-8, a size out of range, a request that names an image
  // it does not give.
  WAIFU_ERR_INVALID_ARGUMENT = 2,
  // Neither a published name nor a manifest that exists.
  WAIFU_ERR_UNKNOWN_MODEL = 3,
  // The download failed.
  WAIFU_ERR_FETCH = 4,
  // The model could not be read, or cannot do what was asked.
  WAIFU_ERR_MODEL = 5,
  // A job with no model loaded, or a model of the wrong kind.
  WAIFU_ERR_NOT_LOADED = 6,
  // A panic in the library: a bug.
  WAIFU_ERR_INTERNAL = 7,
} WaifuStatusCode;

typedef enum {
  // file, done/total in bytes, part/parts.
  WAIFU_STAGE_FETCHING = 0,
  // Reading weights onto the device: no fraction.
  WAIFU_STAGE_READING = 1,
  // The prompt, or the picture img2img starts from.
  WAIFU_STAGE_ENCODING = 2,
  // done/total in steps.
  WAIFU_STAGE_DRAWING = 3,
  WAIFU_STAGE_DECODING = 4,
  // A conversion reading its recordings.
  WAIFU_STAGE_LISTENING = 5,
  // done/total in tokens; total is an estimate and can be passed.
  WAIFU_STAGE_SAYING = 6,
  // The vocoder.
  WAIFU_STAGE_SOUNDING = 7,
} WaifuStage;

typedef enum {
  WAIFU_DEVICE_AUTO = 0,
  WAIFU_DEVICE_CPU = 1,
  WAIFU_DEVICE_METAL = 2,
  WAIFU_DEVICE_CUDA = 3,
  WAIFU_DEVICE_CUDA_CPU_OFFLOAD = 4,
  WAIFU_DEVICE_VULKAN = 5,
} WaifuDevice;

// One thread that holds a model and runs jobs on it, one at a time, in the order asked.
typedef struct WaifuEngine WaifuEngine;

// A download running on a thread of its own.
typedef struct WaifuModelFetch WaifuModelFetch;

typedef struct {
  WaifuLogLevel level;
  // Where it was written: "interface.cc:84" for flint, a module's name for Rust.
  const char *source;
  // The line alone, with no level or time in front of it.
  const char *message;
} WaifuLogEvent;

// Called on whichever thread wrote the line, which can be the caller's own. It must be safe to
// call from any thread, and must not call into the library.
typedef void (*WaifuLogCallback)(void *user_data, const WaifuLogEvent *event);

typedef struct {
  WaifuStage stage;
  // 0..1 of the whole job, or -1 where nothing can say.
  double fraction;
  // What the stage counts; 0, 0 where it counts nothing.
  uint64_t done;
  uint64_t total;
  // FETCHING only: which file of how many.
  uint32_t part;
  uint32_t parts;
  // FETCHING only, else NULL.
  const char *file;
  // "step 3 of 8", for a status line.
  const char *words;
  // Since the job started.
  double seconds;
} WaifuProgressEvent;

typedef void (*WaifuProgressCallback)(void *user_data, const WaifuProgressEvent *event);

// How a job ended, as on_complete is told it.
typedef struct {
  WaifuStatusCode code;
  // Why, when code is not WAIFU_OK; NULL when it is.
  const char *message;
} WaifuStatus;

// How a job that makes nothing ended: a fetch, a load, an unload.
typedef struct {
  WaifuStatus status;
} WaifuCompletionEvent;

typedef void (*WaifuCompleteCallback)(void *user_data, const WaifuCompletionEvent *event);

// 0 is never a job: an _async call returns it when it refuses at the call, with the reason in
// waifu_last_error(), and then calls none of its callbacks. Any other id gets exactly one
// on_complete.
typedef uint64_t WaifuJob;

// A picture as rows of RGB.
typedef struct {
  uint32_t width;
  uint32_t height;
  // width * height * 3 bytes, rows top to bottom.
  const uint8_t *rgb;
  size_t len;
} WaifuImage;

// An image a run is given, by name. A name the prompt writes as <|name|> is read as part of
// what it is told; "start_from" (img2img: see strength) and "control" (a ControlNet's condition:
// see control_scale) are kept for images that are drawn with, and the prompt cannot name them.
typedef struct {
  // Letters, digits and _.
  const char *key;
  // Any size, scaled by the engine.
  const WaifuImage *image;
} WaifuImageInput;

// Refused at the call, as WAIFU_ERR_INVALID_ARGUMENT: two images with one key; a <|name|> in the
// prompt with no image by that name, or naming a kept one; an image the prompt does not name and
// whose key is not a kept one; and any other <|...|>, which the model would read as one of its
// own markers.
typedef struct {
  // sizeof(WaifuDrawRequest), as the caller was compiled with.
  uint32_t struct_size;
  const char *prompt;
  // Words only; "" or NULL for none; ignored by a model without guidance.
  const char *negative;
  // NULL, 0 for none.
  const WaifuImageInput *images;
  size_t image_count;
  // 64..2048, multiples of the model's `alignment` in its description (16; 32 for SDXL).
  uint32_t width;
  uint32_t height;
  uint32_t steps;
  float guidance;
  // The caller's: the same seed and settings draw the same image.
  uint64_t seed;
  // With "start_from": how far it walks away from it, 0..1.
  float strength;
  // With "control": how hard it holds to it.
  float control_scale;
} WaifuDrawRequest;

typedef struct {
  WaifuStatus status;
  // NULL unless status.code is WAIFU_OK.
  const WaifuImage *image;
} WaifuImageCompletionEvent;

typedef void (*WaifuImageCompleteCallback)(void *user_data, const WaifuImageCompletionEvent *event);

// The samples, not a file: what goes in -- a recording to sound like, or to convert -- and what
// comes out. Decoding a file to this and writing this to a file are the caller's.
typedef struct {
  // Mono, -1..1.
  const float *samples;
  size_t count;
  uint32_t rate;
} WaifuAudio;

typedef struct {
  // sizeof(WaifuSpeakRequest), as the caller was compiled with.
  uint32_t struct_size;
  const char *text;
  // A recording to sound like, or NULL.
  const WaifuAudio *like;
  float speed;
  float temperature;
  // NULL or "" for the voice's own way.
  const char *style;
  uint64_t seed;
} WaifuSpeakRequest;

// Shared by speak and voice_conversion: both make audio.
typedef struct {
  WaifuStatus status;
  // NULL unless status.code is WAIFU_OK.
  const WaifuAudio *audio;
} WaifuAudioCompletionEvent;

typedef void (*WaifuAudioCompleteCallback)(void *user_data, const WaifuAudioCompletionEvent *event);

typedef struct {
  // sizeof(WaifuVoiceConversionRequest), as the caller was compiled with.
  uint32_t struct_size;
  // What is said.
  const WaifuAudio *source;
  // Whose voice to say it in.
  const WaifuAudio *reference;
  uint32_t steps;
  bool convert_style;
  uint64_t seed;
} WaifuVoiceConversionRequest;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// The ABI this library was built with: WAIFU_ABI_VERSION.
uint32_t waifu_abi_version(void);

// This thread's last message; valid until this thread's next call.
const char *waifu_last_error(void);

// Frees a string a synchronous call returned. NULL is fine.
void waifu_string_free(char *s);

// Every line the library would print -- flint's and the Rust side's -- goes to the callback
// instead, from whichever thread wrote it. NULL puts them back on the console. Set it before
// anything else: flint writes what hardware it found the first time a device is asked about.
void waifu_set_log_callback(void *user_data, WaifuLogCallback callback);

// Lines below the level are not made at all. INFO unless set.
void waifu_set_log_level(WaifuLogLevel level);

// The processor, the memory, the card and the accelerators this build can use, as JSON. Free
// with waifu_string_free. NULL on failure.
char *waifu_machine_json(void);

// The published models, aliases only, as JSON. Free with waifu_string_free. NULL on failure.
char *waifu_modelmanager_catalog_json(void);

// What a model is, without reading it, as JSON: a published name or a manifest path. Free with
// waifu_string_free. NULL, with waifu_last_error, for a name that is neither.
char *waifu_modelmanager_describe_json(const char *model);

// Where downloads go. Free with waifu_string_free. NULL on failure.
char *waifu_modelmanager_directory(void);

// Downloads go to `path` from now on, saved to config.toml; NULL or "" for the default.
WaifuStatusCode waifu_modelmanager_set_directory(const char *path);

// Reads the manifests of published models out of `directory` where the download directory has
// none: an app built with every model's manifest (the `fetch_manifests` example fetches them)
// can say what each model suggests before anything is downloaded. Laid out as the download
// directory is, and only read. NULL or "" for none. Not saved.
WaifuStatusCode waifu_modelmanager_set_bundled_manifests(const char *directory);

// Deletes a downloaded package.
WaifuStatusCode waifu_modelmanager_remove(const char *name);

// Starts fetching a published model on a thread of its own and returns at once. Fetching one
// already here finishes at once. Where it went is waifu_modelmanager_describe_json's to say.
// NULL, with waifu_last_error and no callback ever, where it cannot start.
WaifuModelFetch *waifu_modelmanager_fetch_async(const char *model,
                                                void *user_data,
                                                WaifuProgressCallback on_progress,
                                                WaifuCompleteCallback on_complete);

// Starts fetching the manifest of every published model that is not here yet -- a couple of
// kilobytes each, and none of their packages -- on a thread of its own, and returns at once.
// What a model suggests is in its manifest, so a screen that has called this can fill in a
// model's settings the moment it is chosen. Each manifest is a part of the whole in the progress;
// with every one already here it finishes at once. Cancelled and freed as a model's fetch is.
// NULL, with waifu_last_error and no callback ever, where it cannot start.
WaifuModelFetch *waifu_modelmanager_fetch_manifests_async(void *user_data,
                                                          WaifuProgressCallback on_progress,
                                                          WaifuCompleteCallback on_complete);

void waifu_modelmanager_fetch_cancel(WaifuModelFetch *fetch);

// After on_complete; cancels first if it has not had it, and waits for the thread to end.
void waifu_modelmanager_fetch_free(WaifuModelFetch *fetch);

// Starts the engine's thread. NULL, with waifu_last_error, if the device is not available.
WaifuEngine *waifu_engine_new(WaifuDevice device);

// Stops what is running, cancels what waits (each gets on_complete with WAIFU_CANCELLED), drops
// the model and joins the thread -- which can take until the end of the step that is running.
void waifu_engine_free(WaifuEngine *engine);

// Stops a job: after the step it is on where it runs, before it starts where it waits.
void waifu_engine_cancel(WaifuEngine *engine, WaifuJob job);

void waifu_engine_cancel_all(WaifuEngine *engine);

// Fetches if needed, then reads onto the device, dropping the model before. What was loaded is
// waifu_modelmanager_describe_json's to say.
WaifuJob waifu_engine_load_async(WaifuEngine *engine,
                                 const char *model,
                                 void *user_data,
                                 WaifuProgressCallback on_progress,
                                 WaifuCompleteCallback on_complete);

WaifuJob waifu_engine_unload_async(WaifuEngine *engine,
                                   void *user_data,
                                   WaifuCompleteCallback on_complete);

// Draws an image. 0, with the reason, for a request no model could run.
WaifuJob waifu_engine_draw_async(WaifuEngine *engine,
                                 const WaifuDrawRequest *request,
                                 void *user_data,
                                 WaifuProgressCallback on_progress,
                                 WaifuImageCompleteCallback on_complete);

// Reads text out in the loaded voice.
WaifuJob waifu_engine_speak_async(WaifuEngine *engine,
                                  const WaifuSpeakRequest *request,
                                  void *user_data,
                                  WaifuProgressCallback on_progress,
                                  WaifuAudioCompleteCallback on_complete);

// Says a recording again in another voice: what is said, and how, is the source's; whose voice
// it is, the reference's.
WaifuJob waifu_engine_voice_conversion_async(WaifuEngine *engine,
                                             const WaifuVoiceConversionRequest *request,
                                             void *user_data,
                                             WaifuProgressCallback on_progress,
                                             WaifuAudioCompleteCallback on_complete);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* LIBWAIFU_WAIFU_H */
