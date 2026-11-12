# libwaifu's C API, ABI 6

Status: implemented. `waifu/include/waifu.h`, generated from `waifu/src/c_api.rs` and
`waifu/src/c_api/{support,request,task}.rs`, is ABI 6.1: 6.0, and the visemes of
[What 6.1 adds](#what-61-adds). What is still open is under [Not decided](#not-decided).

ABI 6 changes how a caller talks to a pipeline, and nothing about what a pipeline does. A request
is a handle the caller sets attributes on, one at a time, instead of a struct; a run, a load and an
unload are each a **task**, a handle the caller waits on and reads events and results from,
instead of three callbacks; and the version is a major and a minor, so an attribute, a stage or an
event added later is a minor version and does not turn an older caller away. Pipelines, models,
devices, stages, status codes and the log callback are ABI 5's ([ffi-abi5.md](ffi-abi5.md)).

## Words

- A **pipeline type** -- `WAIFU_PIPELINE_TEXT_TO_SPEECH` -- is what a pipeline is made for, named
  by one noun phrase everywhere: the pipeline type, the request's and the task's attributes, the
  stages, the model's `SUPPORTS_`.
- A **task** is one load, unload or run on a pipeline, as the caller holds it: `WaifuTask *`.
- An **attribute** is a numbered, typed value: set on a request, read of a task, a model or a
  device. ABI 4 and 5 called a model's and a device's *properties*; they are attributes now, as
  everything else is.

## Rules

ABI 5's rules hold except where this says otherwise: no JSON; enums only grow; every function may
be called from any thread, except that freeing a handle must not race with any other call on it;
two ways to fail, refused at the call or ended with a status; requests are copied at the call; one
name per pipeline type; nothing only a test needs; encoding files is the caller's. Every enum is an
`int32_t` and a `#define` for each value.

- **No callbacks but the log's.** Nothing of a caller's runs on a pipeline's thread. A caller asks
  a task what happened -- `waifu_task_next_event`, `waifu_task_wait` -- on whichever thread it
  likes, and is told in the order it happened.
- **Requests are set an attribute at a time.** `waifu_request_new(type)`, then
  `waifu_request_set_int_attribute(request, WAIFU_REQUEST_IMAGE_GENERATION_STEPS, 28)` and its
  siblings, one for each type of value. An attribute never set has the default its comment gives:
  the library's, not one each binding chooses. An attribute this library does not know, or another
  pipeline type's, is refused with `WAIFU_ERR_UNKNOWN_ATTRIBUTE`; a value of the wrong type or out
  of range, with `WAIFU_ERR_INVALID_ARGUMENT`. What depends on other attributes or on the model --
  a size the model cannot draw at, a transcript with no recording -- is checked at
  `waifu_pipeline_run` or ends the task, as in ABI 5.
- **Results are asked for, one attribute at a time**, as a model's are:
  `waifu_task_get_int_attribute(task, WAIFU_TASK_STATUS, &v)`. A getter's `false` or `NULL` means
  the value is not there -- not yet, not this pipeline type's, or not known to this library -- and
  sets no error.
- **What a task hands out is the task's.** Every pointer a task's getters return, and every pointer
  in a `WaifuImage`, `WaifuRgbaImage` or `WaifuAudio` they fill in, is valid until
  `waifu_task_free`. A caller that keeps a result past that copies it.
- **Four things to free**: a pipeline, a model, a request and a task. NULL is fine to each. A task
  may outlive its pipeline; a request may be freed as soon as `waifu_pipeline_run` returns.
- **Four structs, and they never change**: `WaifuImage`, `WaifuRgbaImage`, `WaifuAudio` and
  `WaifuLog`. Everything else that might grow is an attribute.
- **Versions.** `waifu_abi_version()` is `major << 16 | minor`; see [Versions](#versions).

## The header

What is new; `...` is ABI 5's, unchanged. The header itself has every number and comment.

```c
#define WAIFU_ABI_VERSION_MAJOR 6
#define WAIFU_ABI_VERSION_MINOR 0
#define WAIFU_ABI_VERSION ((WAIFU_ABI_VERSION_MAJOR << 16) | WAIFU_ABI_VERSION_MINOR)
#define WAIFU_WAIT_FOREVER UINT32_MAX

typedef struct WaifuRequest WaifuRequest;
typedef struct WaifuTask WaifuTask;

// WaifuStatusCode: ABI 5's, and
#define WAIFU_ERR_UNKNOWN_ATTRIBUTE 9   // a request attribute this library does not know, or another type's

// WaifuEventType: what waifu_task_next_event says happened.
#define WAIFU_EVENT_NONE 0             // nothing before the timeout
#define WAIFU_EVENT_PROGRESS 1         // the PROGRESS_ attributes say where the task is now
#define WAIFU_EVENT_AUDIO_PART 2       // STREAMING_TEXT_TO_SPEECH_AUDIO_PART is the next part
#define WAIFU_EVENT_END 3              // STATUS is there; and the result, for WAIFU_OK

// WaifuRequestAttribute, by pipeline type as the stages are; type and default in the header.
//   IMAGE_GENERATION_ 100..111: PROMPT (required), NEGATIVE, WIDTH, HEIGHT (1024), STEPS (20),
//     GUIDANCE (5.0), SEED (0), START_IMAGE, START_IMAGE_STRENGTH (0.8), CONTROL_IMAGE,
//     CONTROL_IMAGE_SCALE (1.0), PROMPT_IMAGES (waifu_request_add_prompt_image)
//   TEXT_TO_SPEECH_ 200..206: TEXT (required), REFERENCE, REFERENCE_TRANSCRIPT, SPEED (1.0),
//     TEMPERATURE (1.0), STYLE, SEED
//   VOICE_CONVERSION_ 300..304: SOURCE, REFERENCE (both required), STEPS (10), CONVERT_STYLE, SEED
//   LAYER_DECOMPOSITION_ 400..406: IMAGE (required), STEPS, DEPTH_STEPS (the model's own),
//     RESOLUTION (1280), SPLIT_PAIRS, STRATIFY_LAYERS, SEED
//   SPEECH_TO_TEXT_ 500..502: AUDIO (required), LANGUAGE (the model tells), CONTEXT
//   STREAMING_TEXT_TO_SPEECH_ 600..605: TEXT (required), REFERENCE, REFERENCE_TRANSCRIPT,
//     TEMPERATURE (1.0), STYLE, SEED -- no speed

// WaifuTaskAttribute
//   STATUS 1, STATUS_MESSAGE 2
//   PROGRESS_STAGE 10, _DONE 11, _TOTAL 12, _FRACTION 13, _SECONDS 14: the last PROGRESS event read
//   IMAGE_GENERATION_IMAGE 100; TEXT_TO_SPEECH_AUDIO 200; VOICE_CONVERSION_AUDIO 300
//   LAYER_DECOMPOSITION_CANVAS 400, _LAYER_COUNT 401, and indexed: _LAYER_TYPE 402, _ID 403,
//     _NAME 404, _LEFT 405, _TOP 406, _IMAGE 407, _DEPTH 408, _DERIVED_FROM_ID 409
//   SPEECH_TO_TEXT_TEXT 500, _LANGUAGE 501
//   STREAMING_TEXT_TO_SPEECH_AUDIO_PART 600 (the last AUDIO_PART event read), _AUDIO 601

uint32_t waifu_abi_version(void);  // (major << 16) | minor

WaifuRequest *waifu_request_new(WaifuPipelineType type);
void waifu_request_free(WaifuRequest *request);
WaifuPipelineType waifu_request_get_type(const WaifuRequest *request);
bool waifu_request_set_int_attribute(WaifuRequest *, WaifuRequestAttribute, int64_t value);
bool waifu_request_set_float_attribute(WaifuRequest *, WaifuRequestAttribute, double value);
bool waifu_request_set_string_attribute(WaifuRequest *, WaifuRequestAttribute, const char *value);
bool waifu_request_set_image_attribute(WaifuRequest *, WaifuRequestAttribute, const WaifuImage *value);
bool waifu_request_set_rgba_image_attribute(WaifuRequest *, WaifuRequestAttribute, const WaifuRgbaImage *value);
bool waifu_request_set_audio_attribute(WaifuRequest *, WaifuRequestAttribute, const WaifuAudio *value);
bool waifu_request_add_prompt_image(WaifuRequest *, const char *key, const WaifuImage *image);
bool waifu_request_unset_attribute(WaifuRequest *, WaifuRequestAttribute);

// Each returns at once; NULL where refused.
WaifuTask *waifu_pipeline_load_model(WaifuPipeline *pipeline, const WaifuModel *model);
WaifuTask *waifu_pipeline_unload_model(WaifuPipeline *pipeline);
WaifuTask *waifu_pipeline_run(WaifuPipeline *pipeline, const WaifuRequest *request);
void waifu_pipeline_cancel_all(WaifuPipeline *pipeline);

WaifuEventType waifu_task_next_event(WaifuTask *task, uint32_t timeout_ms);
bool waifu_task_wait(WaifuTask *task, uint32_t timeout_ms);
void waifu_task_cancel(WaifuTask *task);
void waifu_task_free(WaifuTask *task);
bool waifu_task_get_int_attribute(const WaifuTask *, WaifuTaskAttribute, int64_t *value);
bool waifu_task_get_float_attribute(const WaifuTask *, WaifuTaskAttribute, double *value);
const char *waifu_task_get_string_attribute(const WaifuTask *, WaifuTaskAttribute);
bool waifu_task_get_image_attribute(const WaifuTask *, WaifuTaskAttribute, WaifuImage *value);
bool waifu_task_get_rgba_image_attribute(const WaifuTask *, WaifuTaskAttribute, WaifuRgbaImage *value);
bool waifu_task_get_audio_attribute(const WaifuTask *, WaifuTaskAttribute, WaifuAudio *value);
bool waifu_task_get_int_attribute_at(const WaifuTask *, WaifuTaskAttribute, size_t index, int64_t *value);
bool waifu_task_get_float_attribute_at(const WaifuTask *, WaifuTaskAttribute, size_t index, double *value);
const char *waifu_task_get_string_attribute_at(const WaifuTask *, WaifuTaskAttribute, size_t index);
bool waifu_task_get_rgba_image_attribute_at(const WaifuTask *, WaifuTaskAttribute, size_t index, WaifuRgbaImage *value);
```

### A task's events

`waifu_task_next_event` hands the events out in the order they happened, waiting up to
`timeout_ms` (0 only looks, `WAIFU_WAIT_FOREVER` does not stop): `NONE` where nothing came in time,
`END` once the task has ended and every event before it has been read, and `END` for every call
after. Progress not yet read is one event, the latest; audio parts are never merged or dropped. One
thread at a time reads a task's events; any number may `waifu_task_wait` on it, which reads none.

A task is sent only the event types that were there in 6.0, or that its request has an attribute
asking for, so a caller is never handed one it does not know.

### Freeing

`waifu_task_free` on a task that has not ended cancels it, and the library lets it go when it does;
it returns at once. `waifu_pipeline_free` cancels what waits, stops what runs after its step, and
returns once every task of the pipeline has ended -- `WAIFU_CANCELLED` for what had not -- and the
tasks are still the caller's to free. A task's streaming parts are kept until it is freed, as well
as the whole: a reading's audio, which is small, twice.

## Using it

A picture, waited for on the caller's thread:

```c
WaifuRequest *request = waifu_request_new(WAIFU_PIPELINE_IMAGE_GENERATION);
waifu_request_set_string_attribute(request, WAIFU_REQUEST_IMAGE_GENERATION_PROMPT, "a cat");
waifu_request_set_int_attribute(request, WAIFU_REQUEST_IMAGE_GENERATION_STEPS, 28);

WaifuTask *task = waifu_pipeline_run(pipeline, request);   // NULL: waifu_last_error says why
waifu_request_free(request);

waifu_task_wait(task, WAIFU_WAIT_FOREVER);
int64_t status;
waifu_task_get_int_attribute(task, WAIFU_TASK_STATUS, &status);
WaifuImage image;
if (status == WAIFU_OK && waifu_task_get_image_attribute(task, WAIFU_TASK_IMAGE_GENERATION_IMAGE, &image))
  save(image.rgb, image.width, image.height);
waifu_task_free(task);
```

A streaming reading, played as it is said:

```c
WaifuTask *task = waifu_pipeline_run(pipeline, request);
for (;;) {
  WaifuEventType event = waifu_task_next_event(task, WAIFU_WAIT_FOREVER);
  if (event == WAIFU_EVENT_PROGRESS) {
    int64_t stage;
    waifu_task_get_int_attribute(task, WAIFU_TASK_PROGRESS_STAGE, &stage);
    show(stage);
  } else if (event == WAIFU_EVENT_AUDIO_PART) {
    WaifuAudio part;
    waifu_task_get_audio_attribute(task, WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART, &part);
    jitter_buffer_add(part.samples, part.count);
  } else if (event == WAIFU_EVENT_END) {
    break;
  }
}
waifu_task_free(task);
```

A binding does this once, underneath. The Rust binding starts a thread for each job that blocks in
`waifu_task_next_event`, hands each event to the job and wakes its `Waker`, and frees the task once
it has read the end; Swift and .NET read a task the same way, on a thread of their own.

## Versions

A caller built against 6.*m* checks, before anything else, that the library's major is 6 and its
minor is *m* or more.

**A minor version** adds what a caller built against an older one never uses: a request, task,
model or device attribute; a stage, a layer type, a status code; an event type, sent only when the
request has an attribute asking for it -- a picture's previews, say, would be
`IMAGE_GENERATION_PREVIEWS` and `WAIFU_EVENT_IMAGE_PREVIEW`; a function. A caller that sets a newer
attribute on an older library is refused by the setter with `WAIFU_ERR_UNKNOWN_ATTRIBUTE`, and can
carry on without it; a newer task attribute asked of an older library is `false`.

**A major version** removes or renames anything, changes what a number means, a function's
signature or an attribute's default, or changes one of the four structs.

## What changes from ABI 5

| ABI 5 | ABI 6 |
|---|---|
| `WAIFU_ABI_VERSION 5` | `WAIFU_ABI_VERSION_MAJOR 6`, `_MINOR 0`; `waifu_abi_version()` is `major << 16 \| minor` |
| `Waifu<Type>Request` structs, `struct_size` | `WaifuRequest`, `waifu_request_new(type)`, `waifu_request_set_<type>_attribute`, `WaifuRequestAttribute`; `waifu_request_unset_attribute`, `_free`, `_get_type` |
| `WaifuPromptImage`, `prompt_images`, `prompt_image_count` | `waifu_request_add_prompt_image(request, key, image)` |
| defaults chosen by each caller (20 steps in the Rust binding, 30 in waifu-cli) | each attribute's default, the library's; "0 for the model's own" is now "unset" |
| `waifu_pipeline_run_<type>_async`, six of them | `waifu_pipeline_run(pipeline, request)`: the request says the pipeline type |
| `waifu_pipeline_load_model_async`, `_unload_model_async` | `waifu_pipeline_load_model`, `_unload_model`, each returning a `WaifuTask *` |
| `typedef uint64_t WaifuJob`; 0 refused | `WaifuTask *`; NULL refused; `waifu_task_free` |
| `waifu_pipeline_cancel(pipeline, job)` | `waifu_task_cancel(task)`; `waifu_pipeline_cancel_all` stays |
| `WaifuProgressCallback`, `WaifuProgress` | `WAIFU_EVENT_PROGRESS` and the `WAIFU_TASK_PROGRESS_` attributes |
| `WaifuAudioPartCallback` | `WAIFU_EVENT_AUDIO_PART` and `WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART` |
| `on_complete`, `WaifuCallback`, `WaifuResult`, `WaifuStatus`, every `Waifu<Type>Result` and `Waifu<Type>Callback` | `WAIFU_EVENT_END`, `waifu_task_wait`; `WAIFU_TASK_STATUS`, `_STATUS_MESSAGE` and each type's result attributes |
| `WaifuLayer`, `layers`, `layer_count`, `canvas` | `WAIFU_TASK_LAYER_DECOMPOSITION_*`, the layers indexed, with the `_at` getters |
| everything a callback is handed valid until it returns | everything a task hands out valid until `waifu_task_free` |
| two things to free | four: pipeline, model, request, task |
| `WaifuModelProperty`, `waifu_model_get_int_property`, `_get_string_property`; `WaifuDeviceProperty`, `waifu_device_get_int_property` | `WaifuModelAttribute`, `waifu_model_get_int_attribute`, `_get_string_attribute`; `WaifuDeviceAttribute`, `waifu_device_get_int_attribute`: the same numbers |
| -- | `WAIFU_ERR_UNKNOWN_ATTRIBUTE = 9` |

No number that was there is reused.

## What 6.1 adds

A reading's **visemes**: the mouth shape of each frame of its audio, for a face to move to as it
plays -- `A E I O U @ MBP REST` (`WAIFU_VISEME_A` 0 .. `WAIFU_VISEME_REST` 7). A minor version: an
older caller never sets the request attribute, and is never handed any of it.

```c
// WaifuViseme
#define WAIFU_VISEME_A 0      // .. E 1, I 2, O 3, U 4
#define WAIFU_VISEME_SCHWA 5  // @: half open, not spread -- Mandarin's e and en, English's schwa
#define WAIFU_VISEME_MBP 6    // the lips pressed shut
#define WAIFU_VISEME_REST 7   // relaxed and lightly closed; a shape, not silence

#define WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_VISEMES 204           // a CosyVoice3 package with the model

#define WAIFU_REQUEST_TEXT_TO_SPEECH_VISEMES 207                  // boolean; 0
#define WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_VISEMES 606        // boolean; 0

#define WAIFU_TASK_TEXT_TO_SPEECH_VISEMES 201                     // int array: a WaifuViseme a frame
#define WAIFU_TASK_TEXT_TO_SPEECH_VISEME_FRAME_MS 202             // int: 40 for CosyVoice3
#define WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEMES 602           // as of the last AUDIO_PART read
#define WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEME_FRAME_MS 603

const int32_t *waifu_task_get_int_array_attribute(const WaifuTask *, WaifuTaskAttribute, size_t *count);
```

- **Frame `i` is milliseconds `[i, i + 1) * FRAME_MS` of the reading's audio**, and there are
  `ceil(milliseconds / FRAME_MS)` of them: whatever the speed, and across sentences. The frame's
  length is the task's to say, not the request's: it is the model's -- CosyVoice3's is a speech
  token, 40 ms -- and a caller that indexes by `t / FRAME_MS` needs no change for a model whose is
  not.
- **One viseme a frame**, not a distribution: the library picks it, and is where smoothing a
  flicker between two close ones belongs (`VisemeSmoother`, which for now takes each frame's
  likeliest).
- **Streaming**, each `AUDIO_PART` event read makes `STREAMING_TEXT_TO_SPEECH_VISEMES` the frames
  that part settles, every one of them inside the audio handed over so far; it only grows, and each
  array a read hands out stays as it was until `waifu_task_free`. Once the task has ended well it is
  all of them. A frame may come a part late -- once there is a smoother that waits a few frames it
  will -- never early.
- **A model without them** answers `SUPPORTS_TEXT_TO_SPEECH_VISEMES` 0 with why, and a run that asks
  for them ends with `WAIFU_ERR_MODEL`: what a pipeline has loaded is known only on its thread, as
  for a style.
- **`waifu_task_get_int_array_attribute`** is the getter for an array: the values and `*count`,
  valid until `waifu_task_free`; NULL and `*count` untouched where it is not there. An empty array is
  not NULL.

The package's side is in [cosyvoice3.md](cosyvoice3.md#visemes).

## Why

**Every caller turned the callbacks into something else.** waifu-cli's `Engine::load` started a
job and blocked on a channel its `on_complete` wrote to. The Rust binding's `Job` was a queue of
events under a mutex, with a `Waker`, that the callbacks pushed to -- what ABI 6 puts in the
library. Swift's was a continuation in an `Unmanaged` box. WPF's copied what each callback was
handed and posted the copy to the `Dispatcher`, kept its delegates in fields so that they outlived
the call, and took a job's next step only on the UI thread because a callback must not call into
the library. Each had to learn the same rules to do it safely: return quickly, call nothing but
cancel, keep a job's state in `user_data` because the callback may run before the call returns,
copy everything before returning. A task asked rather than a task that calls is those four
wrappers done once, with none of the rules.

**Order comes for free.** `StreamPlayer.swift` fed its `JitterBuffer` from the pipeline's thread,
inside `on_part`, because the main actor's tasks keep no order and the parts must go in in order.
A caller reading events off one queue gets them in the order they happened, on any thread.

**ABI 4 chose not to have this,** "no result-taking API, by choice": a callback's view needed
nothing freed. A task handle costs one free, and buys a result that lives as long as the caller
needs, and no copy for a caller that saves a picture and lets it go.

**Attributes, not structs, because every struct field was a major version.** A request struct is
read at its size; ABI 5 refused one smaller than the library's, so `reference_transcript` turned
every ABI 4 caller away. Reading a smaller one with the rest as zero -- Linux's `openat2` and
`clone3` do, with `copy_struct_from_user` -- was weighed and not taken: it makes zero mean "the
default" in every field, which `temperature` and `seed` cannot spare, and adds fields only at the
end. A value set by number is how libcurl's `curl_easy_setopt` and FFmpeg's AVOptions have grown
for decades, and how ABI 4 already asks a model and a device what they are. The setters are typed,
one for each kind of value, rather than one variadic `setopt`, where an `int` passed for a `long`
compiles and is read wrong.

**Defaults in one place.** With structs every field had a value, so every caller chose one, and
they did not agree. An attribute left unset is the library's default, given once beside it.

**One run function.** The request carries its pipeline type, so `waifu_pipeline_run` is one
function where there were six.

**"Attribute", one word for all of it.** A request's values are set and a task's, a model's and a
device's are read, and one word covers both: `option` and `parameter` are things set, `info`
something read, `field` a struct's member in C. Attribute is what POSIX (`pthread_attr_*`), CUDA
(`cudaDeviceGetAttribute`) and Media Foundation (`IMFAttributes`) call a numbered, typed value of a
handle, set and read alike. ABI 6 is a major version anyway, so the model's and the device's
getters were renamed with the rest rather than left a second word.

**"Task", not "job".** It is what an asynchronous operation one waits on and cancels is called in
.NET, Swift, asyncio and tokio. The kind a pipeline is made for is then always the *pipeline type*
in prose, never a task.

**A major and a minor.** With attributes, nearly everything a release adds is a new number an older
caller never uses and an older library refuses or answers `false` to. Only what would break a caller
built against an older header is a major.

## Not decided

- **A way to be told, for an async wrapper.** Swift, .NET and Rust read a task on a thread of
  their own, blocked in `waifu_task_next_event`. A notification --
  `waifu_task_set_ready_callback(task, fn, user_data)`, saying only that there is an event to
  read -- would let them read from a continuation instead. It can come in a minor version if a
  thread per task turns out to cost something.
- **The Rust binding's names.** Its `Job` is the C API's task, and its marker trait for what a
  pipeline is made for is `Task`.
- **`waifu_request_add_prompt_image`** is the only list a request has, so it has its own function;
  a second list would ask for a general `waifu_request_add_<type>_attribute`.
- **Getters on a request.** Nothing needs to read one back, so there are none.
- **A model's own defaults as attributes** -- `WAIFU_MODEL_IMAGE_GENERATION_STEPS` and the like --
  so that a screen fills its boxes with what an unset attribute would be, and an unset
  `IMAGE_GENERATION_STEPS` that is the model's own rather than 20 for every model.
- ABI 5's `get_` question, on the library's own getters, still stands.
