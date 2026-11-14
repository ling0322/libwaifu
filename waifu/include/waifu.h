/* The MIT License (MIT). Copyright (c) 2026 Xiaoyang Chen. See LICENSE. */

#ifndef LIBWAIFU_WAIFU_H
#define LIBWAIFU_WAIFU_H

/* Written by cbindgen from waifu/src/c_api.rs. Change that, not this. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// The major version: bumped for whatever an older caller would misread -- a number that means
// something else, a function or a type that changed or went, an attribute's default.
//
// 4: pipelines made for one pipeline type, models and devices asked about by attribute, no JSON.
// 5: the text-to-speech requests' reference_transcript, after reference.
// 6: requests set an attribute at a time, tasks read rather than callbacks, a minor version.
#define WAIFU_ABI_VERSION_MAJOR 6

// The minor version: bumped for what an older caller never uses -- an attribute, a stage, an event
// a request has to ask for, a function. Back to 0 with each major.
//
// 6.1: models attached to the one a pipeline has loaded -- waifu_pipeline_attach_model and
// _detach_model, WaifuAttachedModelType, MODEL_ATTACHES_AS -- and the text-to-speech readings'
// visemes: the VISEMES request and task attributes, SUPPORTS_TEXT_TO_SPEECH_VISEMES, WaifuViseme,
// waifu_task_get_int_array_attribute.
#define WAIFU_ABI_VERSION_MINOR 1

// (major << 16) | minor, as waifu_abi_version says it.
#define WAIFU_ABI_VERSION ((WAIFU_ABI_VERSION_MAJOR << 16) | WAIFU_ABI_VERSION_MINOR)

// waifu_task_next_event and waifu_task_wait: no timeout.
#define WAIFU_WAIT_FOREVER UINT32_MAX

// A model's manifest, read, and what can be asked of it, answered once when it is opened.
typedef struct WaifuModel WaifuModel;

// One pipeline type, one thread, at most one model: its tasks run one at a time, in the order
// asked.
typedef struct WaifuPipeline WaifuPipeline;

// What one run on a pipeline of one type is asked to do. Not shared between threads while it is
// being set; run as often as liked, with attributes changed in between: waifu_pipeline_run copies
// it.
typedef struct WaifuRequest WaifuRequest;

// One load, unload or run on a pipeline, as the caller holds it: its events are read with
// waifu_task_next_event, and its status and result with the getters, on any thread.
typedef struct WaifuTask WaifuTask;

// What a refused call or a task came to.
typedef int32_t WaifuStatusCode;

typedef int32_t WaifuLogLevel;

typedef struct {
  WaifuLogLevel level;
  // Where it was written, as a file and a line: "interface.cc:84", "engine.rs:712".
  const char *source;
  // The line alone, with no level or time in front of it.
  const char *message;
  // When it was written, in UTC: "2026-10-23T18:50:41Z".
  const char *time;
} WaifuLog;

// Called on whichever thread wrote the line, which can be the caller's own, possibly from several
// threads at once. Must not call into the library.
typedef void (*WaifuLogCallback)(void *user_data, const WaifuLog *log);

typedef int32_t WaifuDevice;

// What can be asked of a device. All of it is this process's, not the card as other programs see
// it.
typedef int32_t WaifuDeviceAttribute;

// What can be asked of a model. SUPPORTS_ attributes are integers, 1 or 0; where 0, the string
// attribute of the same number is why, in words to show, and is never NULL. A SUPPORTS_ attribute
// this library does not know is absent, which is neither 1 nor 0. The rest are an integer or a
// string, as marked. SUPPORTS_<type> has the number of WAIFU_PIPELINE_<type>.
typedef int32_t WaifuModelAttribute;

// What a pipeline is made for.
typedef int32_t WaifuPipelineType;

// What a model is attached to the one a pipeline has loaded as: waifu_pipeline_attach_model.
// Each is for one pipeline type's model, which its name says. Since 6.1.
typedef int32_t WaifuAttachedModelType;

// A request's attributes, numbered by pipeline type as the stages are. The type after each is the
// setter it takes; "required" is refused at waifu_pipeline_run when unset. A boolean is an int, 0
// or 1.
typedef int32_t WaifuRequestAttribute;

typedef struct {
  uint32_t width;
  uint32_t height;
  // width * height * 3, rows top to bottom.
  const uint8_t *rgb;
  size_t len;
} WaifuImage;

typedef struct {
  uint32_t width;
  uint32_t height;
  // width * height * 4, not premultiplied.
  const uint8_t *rgba;
  size_t len;
} WaifuRgbaImage;

typedef struct {
  // Mono, -1..1.
  const float *samples;
  size_t count;
  uint32_t sample_rate;
} WaifuAudio;

// What waifu_task_next_event says happened. A task is sent only the types its request asked for
// or that were there in 6.0, so a caller is never handed one it does not know.
typedef int32_t WaifuEventType;

// What a task can be asked. The type after each is the getter that reads it; "indexed" ones are
// read with the _at getters, 0 .. LAYER_DECOMPOSITION_LAYER_COUNT - 1.
typedef int32_t WaifuTaskAttribute;

typedef int32_t WaifuLayerType;

// A mouth shape, one for each frame of a reading's VISEMES: the sounds a face cannot tell apart
// by looking are one.
typedef int32_t WaifuViseme;

// What a task is doing. Ranges by pipeline type; done/total count what the comment says, 0, 0
// otherwise.
typedef int32_t WaifuStage;

#define WAIFU_DEVICE_AUTO 0

#define WAIFU_DEVICE_CPU 1

#define WAIFU_DEVICE_METAL 2

#define WAIFU_DEVICE_CUDA 3

#define WAIFU_DEVICE_VULKAN 5

#define WAIFU_PIPELINE_IMAGE_GENERATION 1

#define WAIFU_PIPELINE_TEXT_TO_SPEECH 2

#define WAIFU_PIPELINE_VOICE_CONVERSION 3

#define WAIFU_PIPELINE_LAYER_DECOMPOSITION 4

#define WAIFU_PIPELINE_SPEECH_TO_TEXT 5

#define WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH 6

#define WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION 1

#define WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH 2

#define WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION 3

#define WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION 4

#define WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT 5

// The model streams; the device is checked at load.
#define WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH 6

// img2img.
#define WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE 100

// A ControlNet.
#define WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE 101

// Images the prompt names as <|key|>.
#define WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES 102

// In the voice of a recording.
#define WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE 200

// CosyVoice3 does not.
#define WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE 201

// Follows an instruction such as "speak happily".
#define WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE 202

// A reference without its reference_transcript: CosyVoice3 does not, unless a style is given.
#define WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE_TRANSCRIPT 203

// Takes a WAIFU_ATTACHED_MODEL_TEXT_TO_SPEECH_VISEMES model, and with one attached says the
// visemes of what it says, with TEXT_TO_SPEECH_VISEMES or STREAMING_TEXT_TO_SPEECH_VISEMES:
// CosyVoice3. Since 6.1.
#define WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_VISEMES 204

// The manner as well as the voice.
#define WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE 300

// Integer, bytes; absent until every file is here.
#define WAIFU_MODEL_SIZE_ON_DISK 1000

// Integer, of what it says or converts.
#define WAIFU_MODEL_SAMPLE_RATE 1001

// Integer, a WaifuAttachedModelType: what it is attached to a pipeline's model as -- a model that
// is not loaded as one's own. Absent for every model a pipeline loads. Since 6.1.
#define WAIFU_MODEL_ATTACHES_AS 1002

// Integer: 16, or 32 for SDXL and Qwen-Image.
#define WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF 1100

// String: "Euler".
#define WAIFU_MODEL_IMAGE_GENERATION_SAMPLER 1101

// Integer.
#define WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES 1102

// Integer: 64.
#define WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF 1400

// String: "Chinese,English,...", comma separated.
#define WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES 1500

// The most it may hold there: the memory limit, or the device's memory.
#define WAIFU_DEVICE_MEMORY_BUDGET 1

// What its allocator holds now, what it keeps for reuse included.
#define WAIFU_DEVICE_MEMORY_HELD 2

// BUDGET - HELD, never below 0.
#define WAIFU_DEVICE_MEMORY_AVAILABLE 3

// What waifu_device_set_memory_limit set; absent where nothing did.
#define WAIFU_DEVICE_MEMORY_LIMIT 4

// A part as the model drew it.
#define WAIFU_LAYER_PART 0

// stratify_layers: the near half of a part.
#define WAIFU_LAYER_STRATIFIED_NEAR 1

// stratify_layers: the far half of a part.
#define WAIFU_LAYER_STRATIFIED_FAR 2

// The depth of the layer derived_from_id names, pixel by pixel.
#define WAIFU_LAYER_DEPTH 3

// The five open vowels, as a Japanese face draws them.
#define WAIFU_VISEME_A 0

#define WAIFU_VISEME_E 1

#define WAIFU_VISEME_I 2

#define WAIFU_VISEME_O 3

#define WAIFU_VISEME_U 4

// The schwa, @: half open and not spread -- Mandarin's e and en, English's unstressed vowel.
#define WAIFU_VISEME_SCHWA 5

// The lips pressed shut: b, p and m.
#define WAIFU_VISEME_MBP 6

// Relaxed and lightly closed. A shape, not silence: whether anybody is speaking is the sound's.
#define WAIFU_VISEME_REST 7

// Which mouth shape goes with each frame of a reading, for TEXT_TO_SPEECH_VISEMES: a
// cosyvoice3_visemes package, attached to CosyVoice3 on a text to speech or streaming text to
// speech pipeline, with no key.
#define WAIFU_ATTACHED_MODEL_TEXT_TO_SPEECH_VISEMES 1

#define WAIFU_OK 0

#define WAIFU_CANCELLED 1

// What a request cannot be run with. Most of it is refused at the call instead -- a NULL, a
// string that is not UTF-8, a size out of range, a model the pipeline's type is not supported by,
// a request of another pipeline type -- which a task never sees.
#define WAIFU_ERR_INVALID_ARGUMENT 2

// No manifest at the path.
#define WAIFU_ERR_UNKNOWN_MODEL 3

// The model could not be read, or cannot do what was asked.
#define WAIFU_ERR_MODEL 5

// A run on a pipeline with no model loaded.
#define WAIFU_ERR_NOT_LOADED 6

// A panic in the library: a bug.
#define WAIFU_ERR_INTERNAL 7

// A pipeline asked for on a device this build or machine cannot use.
#define WAIFU_ERR_DEVICE_UNAVAILABLE 8

// A request attribute this library does not know, or one of another pipeline type's. A task's
// getter asked about an attribute it does not know says false, and sets nothing.
#define WAIFU_ERR_UNKNOWN_ATTRIBUTE 9

#define WAIFU_LOG_DEBUG 0

#define WAIFU_LOG_INFO 1

#define WAIFU_LOG_WARNING 2

#define WAIFU_LOG_ERROR 3

// The process aborts when the callback returns.
#define WAIFU_LOG_FATAL 4

#define WAIFU_STAGE_LOADING_WEIGHTS 1

#define WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT 100

// Steps.
#define WAIFU_STAGE_IMAGE_GENERATION_DENOISING 101

#define WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE 102

#define WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT 200

// Tokens; total is an estimate, and can be passed.
#define WAIFU_STAGE_TEXT_TO_SPEECH_GENERATING_TOKENS 201

#define WAIFU_STAGE_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO 202

#define WAIFU_STAGE_VOICE_CONVERSION_READING_RECORDINGS 300

// Tokens; total is an estimate.
#define WAIFU_STAGE_VOICE_CONVERSION_GENERATING_TOKENS 301

// Steps.
#define WAIFU_STAGE_VOICE_CONVERSION_DENOISING 302

// Vocoder windows.
#define WAIFU_STAGE_VOICE_CONVERSION_SYNTHESIZING_AUDIO 303

// The whole picture, then the head, then the depth model that orders the layers.
#define WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_IMAGE 400

// Steps.
#define WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_LAYERS 401

// Layers.
#define WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_LAYERS 402

#define WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_HEAD 403

// Steps.
#define WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_HEAD_LAYERS 404

// Layers.
#define WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_HEAD_LAYERS 405

// Layers.
#define WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_FOR_DEPTH 406

// Steps.
#define WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_DEPTH 407

// Layers.
#define WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_DEPTH 408

#define WAIFU_STAGE_LAYER_DECOMPOSITION_CUTTING_INTO_PARTS 409

// Pieces the recording is cut into.
#define WAIFU_STAGE_SPEECH_TO_TEXT_READING_RECORDING 500

// Tokens; total is an estimate, and can be passed.
#define WAIFU_STAGE_SPEECH_TO_TEXT_GENERATING_TOKENS 501

// Saying and sounding take turns, a hop of tokens at a time.
#define WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_READING_TEXT 600

// Tokens; total is an estimate.
#define WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_GENERATING_TOKENS 601

// Parts handed over.
#define WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO 602

// String, required.
#define WAIFU_REQUEST_IMAGE_GENERATION_PROMPT 100

// String; none.
#define WAIFU_REQUEST_IMAGE_GENERATION_NEGATIVE 101

// Int, 64..2048, a multiple of the model's IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF; 1024.
#define WAIFU_REQUEST_IMAGE_GENERATION_WIDTH 102

// Int, as WIDTH; 1024.
#define WAIFU_REQUEST_IMAGE_GENERATION_HEIGHT 103

// Int, 1..; 20.
#define WAIFU_REQUEST_IMAGE_GENERATION_STEPS 104

// Float, 0..; 5.0, ignored by a model without guidance.
#define WAIFU_REQUEST_IMAGE_GENERATION_GUIDANCE 105

// Int, read as uint64; 0.
#define WAIFU_REQUEST_IMAGE_GENERATION_SEED 106

// Image; none (img2img).
#define WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE 107

// Float, 0..1, how far it walks away from it; 0.8.
#define WAIFU_REQUEST_IMAGE_GENERATION_START_IMAGE_STRENGTH 108

// Image; none (a ControlNet's condition).
#define WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE 109

// Float, 0.., how hard it holds to it; 1.0.
#define WAIFU_REQUEST_IMAGE_GENERATION_CONTROL_IMAGE_SCALE 110

// waifu_request_add_prompt_image; none.
#define WAIFU_REQUEST_IMAGE_GENERATION_PROMPT_IMAGES 111

// String, required.
#define WAIFU_REQUEST_TEXT_TO_SPEECH_TEXT 200

// Audio, a recording to sound like; none.
#define WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE 201

// String, what REFERENCE says, word for word; none. CosyVoice3 reads on from it, as upstream's
// zero-shot does, and takes it or a style, not both; a model that reads none ignores it.
#define WAIFU_REQUEST_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT 202

// Float, above 0; 1.0.
#define WAIFU_REQUEST_TEXT_TO_SPEECH_SPEED 203

// Float, 0..; 1.0.
#define WAIFU_REQUEST_TEXT_TO_SPEECH_TEMPERATURE 204

// String, any instruction; the voice's own way.
#define WAIFU_REQUEST_TEXT_TO_SPEECH_STYLE 205

// Int, read as uint64; 0.
#define WAIFU_REQUEST_TEXT_TO_SPEECH_SEED 206

// Boolean, the visemes as well, TEXT_TO_SPEECH_VISEMES; 0. With no
// WAIFU_ATTACHED_MODEL_TEXT_TO_SPEECH_VISEMES model attached the task ends with WAIFU_ERR_MODEL.
// Since 6.1.
#define WAIFU_REQUEST_TEXT_TO_SPEECH_VISEMES 207

// Audio, what is said and how; required.
#define WAIFU_REQUEST_VOICE_CONVERSION_SOURCE 300

// Audio, whose voice to say it in; required.
#define WAIFU_REQUEST_VOICE_CONVERSION_REFERENCE 301

// Int, 1..; 10.
#define WAIFU_REQUEST_VOICE_CONVERSION_STEPS 302

// Boolean; 0.
#define WAIFU_REQUEST_VOICE_CONVERSION_CONVERT_STYLE 303

// Int, read as uint64; 0.
#define WAIFU_REQUEST_VOICE_CONVERSION_SEED 304

// RGBA image, any size, centred and scaled by the engine; required.
#define WAIFU_REQUEST_LAYER_DECOMPOSITION_IMAGE 400

// Int, 1..; the model's own.
#define WAIFU_REQUEST_LAYER_DECOMPOSITION_STEPS 401

// Int, 1..; the model's own.
#define WAIFU_REQUEST_LAYER_DECOMPOSITION_DEPTH_STEPS 402

// Int, a multiple of the model's LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF; 1280.
#define WAIFU_REQUEST_LAYER_DECOMPOSITION_RESOLUTION 403

// Boolean, hands, eyes, eyebrows, ears into a -l and a -r layer; 0.
#define WAIFU_REQUEST_LAYER_DECOMPOSITION_SPLIT_PAIRS 404

// Boolean, hair, hands, top and bottom also cut into a near and a far half; 0.
#define WAIFU_REQUEST_LAYER_DECOMPOSITION_STRATIFY_LAYERS 405

// Int, read as uint64; 0.
#define WAIFU_REQUEST_LAYER_DECOMPOSITION_SEED 406

// Audio, any rate and length; required.
#define WAIFU_REQUEST_SPEECH_TO_TEXT_AUDIO 500

// String, one of the model's SPEECH_TO_TEXT_LANGUAGES -- "English", "chinese"; the model tells.
#define WAIFU_REQUEST_SPEECH_TO_TEXT_LANGUAGE 501

// String, words that help it hear: names, terms, the subject; none.
#define WAIFU_REQUEST_SPEECH_TO_TEXT_CONTEXT 502

// String, required.
#define WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEXT 600

// Audio; none.
#define WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE 601

// String, as TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT; none.
#define WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_REFERENCE_TRANSCRIPT 602

// Float, 0..; 1.0.
#define WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_TEMPERATURE 603

// String; the voice's own way. No speed: a streaming reading is at the speed it is said, as
// upstream's is.
#define WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_STYLE 604

// Int, read as uint64; 0.
#define WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_SEED 605

// Boolean, as TEXT_TO_SPEECH_VISEMES: STREAMING_TEXT_TO_SPEECH_VISEMES with each part; 0.
// Since 6.1.
#define WAIFU_REQUEST_STREAMING_TEXT_TO_SPEECH_VISEMES 606

// Nothing before the timeout.
#define WAIFU_EVENT_NONE 0

// The PROGRESS_ attributes say where the task is now.
#define WAIFU_EVENT_PROGRESS 1

// Streaming text to speech: STREAMING_TEXT_TO_SPEECH_AUDIO_PART is the next part.
#define WAIFU_EVENT_AUDIO_PART 2

// STATUS is there; and the result, for WAIFU_OK.
#define WAIFU_EVENT_END 3

// Int, a WaifuStatusCode; absent until the task has ended.
#define WAIFU_TASK_STATUS 1

// String, why; absent for WAIFU_OK and WAIFU_CANCELLED.
#define WAIFU_TASK_STATUS_MESSAGE 2

// Int, a WaifuStage: the progress the last PROGRESS event said; absent before one.
#define WAIFU_TASK_PROGRESS_STAGE 10

// Int, of the stage, done already.
#define WAIFU_TASK_PROGRESS_DONE 11

// Int.
#define WAIFU_TASK_PROGRESS_TOTAL 12

// Float, 0..1 of the whole task; absent where nothing can say.
#define WAIFU_TASK_PROGRESS_FRACTION 13

// Float, seconds since the task started.
#define WAIFU_TASK_PROGRESS_SECONDS 14

// Image: there once STATUS is WAIFU_OK, as each task's result is.
#define WAIFU_TASK_IMAGE_GENERATION_IMAGE 100

// Audio.
#define WAIFU_TASK_TEXT_TO_SPEECH_AUDIO 200

// Int array, read with waifu_task_get_int_array_attribute: a WaifuViseme for each
// TEXT_TO_SPEECH_VISEME_FRAME_MS of AUDIO, frame i being milliseconds [i, i + 1) * FRAME_MS of it,
// ceil(milliseconds / FRAME_MS) of them; for a request with TEXT_TO_SPEECH_VISEMES. Since 6.1.
#define WAIFU_TASK_TEXT_TO_SPEECH_VISEMES 201

// Int: how many milliseconds a frame of VISEMES holds -- 40 for CosyVoice3, a speech token's.
// Since 6.1.
#define WAIFU_TASK_TEXT_TO_SPEECH_VISEME_FRAME_MS 202

// Audio.
#define WAIFU_TASK_VOICE_CONVERSION_AUDIO 300

// Int, the side of the square canvas.
#define WAIFU_TASK_LAYER_DECOMPOSITION_CANVAS 400

// Int. The layers are in one order, which the library never changes: each part, bottom first --
// which is the stack, unstratified -- then, with STRATIFY_LAYERS, each part's halves, near before
// far; then a depth layer for each of those, in their order. The stratified stack is every part
// that was not split and every half, bottom first: from the largest depth to the smallest; at one
// depth, a far half under a near one; then by their order. No depth layer is in either stack.
#define WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_COUNT 401

// Int, indexed: a WaifuLayerType.
#define WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TYPE 402

// Int, indexed: unique within the result, from 1; it means nothing else.
#define WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_ID 403

// String, indexed: "topwear"; its halves "topwear-0" and "topwear-1"; its depth "topwear_depth".
#define WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_NAME 404

// Int, indexed: where its top left corner is on the canvas.
#define WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_LEFT 405

// Int, indexed.
#define WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_TOP 406

// RGBA image, indexed: only as large as what is drawn. A depth layer: the same size and place as
// its layer, R, G and B the depth, 0 nearest to 255 farthest, under its layer's alpha.
#define WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_IMAGE 407

// Float, indexed: 0 nearest, 1 farthest. A near half has its part's depth, not its pixels': that
// is what keeps it in its part's place in the stratified stack. A depth layer has its layer's.
#define WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DEPTH 408

// Int, indexed: the id of the layer this one is made from -- a half's part, a depth layer's
// layer; 0 for a part.
#define WAIFU_TASK_LAYER_DECOMPOSITION_LAYER_DERIVED_FROM_ID 409

// String: what was said; "" for nothing said.
#define WAIFU_TASK_SPEECH_TO_TEXT_TEXT 500

// String: the language it was said in, as the model names it -- "Chinese" -- or several, comma
// separated, for a recording that changes; "" where none was heard.
#define WAIFU_TASK_SPEECH_TO_TEXT_LANGUAGE 501

// Audio: the part the last AUDIO_PART event said, from the moment it is said, before the task
// ends.
#define WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO_PART 600

// Audio: every part, end to end.
#define WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_AUDIO 601

// Int array, as TEXT_TO_SPEECH_VISEMES, frame 0 the start of the first part: the frames settled
// by the part the last AUDIO_PART event said, from the moment it is said -- never past the end of
// that part, and only ever more of them with each part -- and every frame once the task has ended
// well. For a request with STREAMING_TEXT_TO_SPEECH_VISEMES. Since 6.1.
#define WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEMES 602

// Int, as TEXT_TO_SPEECH_VISEME_FRAME_MS. Since 6.1.
#define WAIFU_TASK_STREAMING_TEXT_TO_SPEECH_VISEME_FRAME_MS 603

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// (major << 16) | minor. A caller built against 6.m runs on any 6.n with n >= m.
uint32_t waifu_abi_version(void);

// This thread's last refused call, as a sentence and as a code; both are set together by a
// refused call and left alone by every other, and reading one does not change the other. The
// string is valid until this thread's next refused call. "" and WAIFU_OK before any.
const char *waifu_last_error(void);

WaifuStatusCode waifu_last_error_code(void);

// "0.1.0"; static.
const char *waifu_version(void);

// The full commit, or NULL for a build made without git; static.
const char *waifu_revision(void);

// NULL: the console again. Waits for a line being handed to the callback before to finish: once
// it returns, the callback and user_data before are never called again, and may be let go of.
void waifu_set_log_callback(void *user_data, WaifuLogCallback callback);

// INFO unless set. A level this library does not know changes nothing.
void waifu_set_log_level(WaifuLogLevel level);

bool waifu_device_is_available(WaifuDevice device);

// false for the processor, a device that cannot be used, or an attribute this library does not
// know. Asking CUDA builds a context on the card, a few hundred megabytes of it.
bool waifu_device_get_int_attribute(WaifuDevice device,
                                    WaifuDeviceAttribute attribute,
                                    int64_t *value);

// For the whole process, one per device, shared by every pipeline on it: set it before a model
// is loaded. 0 for no limit but the device's. The processor has none, and is refused.
bool waifu_device_set_memory_limit(WaifuDevice device, uint64_t bytes);

// Reads the manifest at `manifest_path` -- a few kilobytes, and none of the weights. Refused --
// NULL -- with WAIFU_ERR_UNKNOWN_MODEL where there is none, WAIFU_ERR_MODEL where it cannot be
// read.
WaifuModel *waifu_model_open(const char *manifest_path);

// NULL is fine.
void waifu_model_free(WaifuModel *model);

// false where the model has no such attribute or this library does not know it; *value untouched.
bool waifu_model_get_int_attribute(const WaifuModel *model,
                                   WaifuModelAttribute attribute,
                                   int64_t *value);

// NULL where there is none; valid until waifu_model_free.
const char *waifu_model_get_string_attribute(const WaifuModel *model,
                                             WaifuModelAttribute attribute);

// Starts the pipeline's thread, with no model. Refused -- NULL, WAIFU_ERR_DEVICE_UNAVAILABLE --
// where the device is not available.
WaifuPipeline *waifu_pipeline_new(WaifuPipelineType type, WaifuDevice device);

// Stops what runs, cancels what waits, drops the model and joins the thread. Returns once every
// task of the pipeline has ended -- WAIFU_CANCELLED for what had not -- and the tasks are still
// the caller's to free. NULL is fine.
void waifu_pipeline_free(WaifuPipeline *pipeline);

// 0 for a NULL pipeline.
WaifuPipelineType waifu_pipeline_get_type(const WaifuPipeline *pipeline);

// Never AUTO, which is settled when it starts; AUTO only for a NULL pipeline.
WaifuDevice waifu_pipeline_get_device(const WaifuPipeline *pipeline);

// Every task of the pipeline, waiting or running, is cancelled.
void waifu_pipeline_cancel_all(WaifuPipeline *pipeline);

// Reads the model's weights onto the pipeline's device, dropping the model before first, so two
// are never on it at once. Refused -- NULL -- for a model the pipeline's type is not supported by
// (WAIFU_MODEL_SUPPORTS_<type>). The call copies the manifest, not the weights, which the
// pipeline's thread reads from the files: `model` may be freed as soon as it returns. A run may
// be asked for straight after; it waits its turn. "tones", the stand-in voice the tests use,
// opens as a model too.
WaifuTask *waifu_pipeline_load_model(WaifuPipeline *pipeline, const WaifuModel *model);

// Attaches `model` to the model the pipeline has loaded, as `type`, under `key` -- NULL or "" for
// the one of its type that has no key -- replacing what was attached as that type under that key.
// It stays for every run until it is detached, or the pipeline's model is unloaded or another is
// loaded, which let it go with the model it was attached to. Refused for a type this library does
// not know, a model whose ATTACHES_AS is not `type`, a type that is not for this pipeline type, or
// a key the type does not take: TEXT_TO_SPEECH_VISEMES takes none. The task ends with
// WAIFU_ERR_MODEL where the pipeline has no model, or one that does not take it. As a load, the
// call copies the manifest and not the weights, and `model` may be freed as soon as it returns.
// Since 6.1.
WaifuTask *waifu_pipeline_attach_model(WaifuPipeline *pipeline,
                                       WaifuAttachedModelType type,
                                       const char *key,
                                       const WaifuModel *model);

// Lets go of what is attached to the pipeline's model as `type` under `key`; done where nothing
// is. Refused as waifu_pipeline_attach_model is for the type and the key. Since 6.1.
WaifuTask *waifu_pipeline_detach_model(WaifuPipeline *pipeline,
                                       WaifuAttachedModelType type,
                                       const char *key);

// Drops the model; the pipeline then has none.
WaifuTask *waifu_pipeline_unload_model(WaifuPipeline *pipeline);

// Runs the request on the pipeline's thread, in its turn, and returns at once. Refused -- NULL,
// with the last error and its code -- for a request of another pipeline type, a required attribute
// unset, or an attribute that contradicts another; a run with no model loaded is not refused, and
// ends with WAIFU_ERR_NOT_LOADED.
WaifuTask *waifu_pipeline_run(WaifuPipeline *pipeline, const WaifuRequest *request);

// A request for a pipeline of `type`, with every attribute unset. Refused -- NULL,
// WAIFU_ERR_INVALID_ARGUMENT -- for a type this library does not know.
WaifuRequest *waifu_request_new(WaifuPipelineType type);

// NULL is fine.
void waifu_request_free(WaifuRequest *request);

// 0 for NULL.
WaifuPipelineType waifu_request_get_type(const WaifuRequest *request);

// Each sets one attribute and copies what it is handed: a string, an image's pixels, a recording's
// samples. false, with the last error and its code, for an attribute this library does not know or
// that is another pipeline type's (WAIFU_ERR_UNKNOWN_ATTRIBUTE), and for a value of another type,
// out of the attribute's range, or a string that is not UTF-8 (WAIFU_ERR_INVALID_ARGUMENT); the
// attribute is then as it was. A NULL string, image or audio unsets the attribute.
bool waifu_request_set_int_attribute(WaifuRequest *request,
                                     WaifuRequestAttribute attribute,
                                     int64_t value);

// As waifu_request_set_int_attribute. The value is kept, and checked against the attribute's
// range, as the float the model is handed: one too large for a float is refused, not infinity.
bool waifu_request_set_float_attribute(WaifuRequest *request,
                                       WaifuRequestAttribute attribute,
                                       double value);

// As waifu_request_set_int_attribute.
bool waifu_request_set_string_attribute(WaifuRequest *request,
                                        WaifuRequestAttribute attribute,
                                        const char *value);

// As waifu_request_set_int_attribute.
bool waifu_request_set_image_attribute(WaifuRequest *request,
                                       WaifuRequestAttribute attribute,
                                       const WaifuImage *value);

// As waifu_request_set_int_attribute.
bool waifu_request_set_rgba_image_attribute(WaifuRequest *request,
                                            WaifuRequestAttribute attribute,
                                            const WaifuRgbaImage *value);

// As waifu_request_set_int_attribute.
bool waifu_request_set_audio_attribute(WaifuRequest *request,
                                       WaifuRequestAttribute attribute,
                                       const WaifuAudio *value);

// IMAGE_GENERATION_PROMPT_IMAGES: one more image the prompt names as <|key|>; key is letters,
// digits and _. A key given twice is refused. Unsetting the attribute drops them all.
bool waifu_request_add_prompt_image(WaifuRequest *request,
                                    const char *key,
                                    const WaifuImage *image);

// Back to the attribute's default. false, as a setter's, for an attribute this library does not
// know.
bool waifu_request_unset_attribute(WaifuRequest *request, WaifuRequestAttribute attribute);

// The next thing that happened to the task, in order, waiting up to timeout_ms for it: 0 only
// looks. NONE where nothing did in time. END once it has ended, and END for every call after.
// Progress not yet read is one event, the latest; audio parts are never merged or dropped. One
// thread at a time asks a task for events.
WaifuEventType waifu_task_next_event(WaifuTask *task, uint32_t timeout_ms);

// Waits up to timeout_ms for the task to end, without reading its events: true once it has. Any
// number of threads may wait on one task.
bool waifu_task_wait(WaifuTask *task, uint32_t timeout_ms);

// After the step it is on, or before it starts; it then ends with WAIFU_CANCELLED, waking anyone
// waiting. Nothing once it has ended.
void waifu_task_cancel(WaifuTask *task);

// A task that has not ended is cancelled, and the library lets it go when it does; this returns
// at once. Every pointer the task handed out is then gone. NULL is fine.
void waifu_task_free(WaifuTask *task);

// false, or NULL, where the attribute is not there -- not yet, of another pipeline type, or not
// known to this library -- and *value untouched; nothing is set. A pointer, and every pointer in
// what an image or audio getter fills in, is valid until waifu_task_free.
bool waifu_task_get_int_attribute(const WaifuTask *task,
                                  WaifuTaskAttribute attribute,
                                  int64_t *value);

// As waifu_task_get_int_attribute.
bool waifu_task_get_float_attribute(const WaifuTask *task,
                                    WaifuTaskAttribute attribute,
                                    double *value);

// As waifu_task_get_int_attribute.
const char *waifu_task_get_string_attribute(const WaifuTask *task, WaifuTaskAttribute attribute);

// As waifu_task_get_int_attribute.
bool waifu_task_get_image_attribute(const WaifuTask *task,
                                    WaifuTaskAttribute attribute,
                                    WaifuImage *value);

// As waifu_task_get_int_attribute.
bool waifu_task_get_rgba_image_attribute(const WaifuTask *task,
                                         WaifuTaskAttribute attribute,
                                         WaifuRgbaImage *value);

// As waifu_task_get_int_attribute.
bool waifu_task_get_audio_attribute(const WaifuTask *task,
                                    WaifuTaskAttribute attribute,
                                    WaifuAudio *value);

// The arrays: a pointer to *count values, valid until waifu_task_free, or NULL and *count
// untouched where the attribute is not there, as waifu_task_get_int_attribute. Since 6.1.
const int32_t *waifu_task_get_int_array_attribute(const WaifuTask *task,
                                                  WaifuTaskAttribute attribute,
                                                  size_t *count);

// The indexed attributes: the layers of a decomposition. false, or NULL, past the count as well.
bool waifu_task_get_int_attribute_at(const WaifuTask *task,
                                     WaifuTaskAttribute attribute,
                                     size_t index,
                                     int64_t *value);

// As waifu_task_get_int_attribute_at.
bool waifu_task_get_float_attribute_at(const WaifuTask *task,
                                       WaifuTaskAttribute attribute,
                                       size_t index,
                                       double *value);

// As waifu_task_get_int_attribute_at.
const char *waifu_task_get_string_attribute_at(const WaifuTask *task,
                                               WaifuTaskAttribute attribute,
                                               size_t index);

// As waifu_task_get_int_attribute_at.
bool waifu_task_get_rgba_image_attribute_at(const WaifuTask *task,
                                            WaifuTaskAttribute attribute,
                                            size_t index,
                                            WaifuRgbaImage *value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* LIBWAIFU_WAIFU_H */
