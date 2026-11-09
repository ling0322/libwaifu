/* The MIT License (MIT). Copyright (c) 2026 Xiaoyang Chen. See LICENSE. */

#ifndef LIBWAIFU_WAIFU_H
#define LIBWAIFU_WAIFU_H

/* Written by cbindgen from waifu/src/c_api.rs. Change that, not this. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// What a caller compiled against this header expects. Bumped when anything in it changes in a
// way an older caller would misread.
//
// 4: pipelines made for one task, models and devices asked about by property, no JSON.
// 5: the text-to-speech requests' reference_transcript, after reference.
#define WAIFU_ABI_VERSION 5

// A model's manifest, read, and what can be asked of it, answered once when it is opened.
typedef struct WaifuModel WaifuModel;

// One task, one thread, at most one model: jobs run on it one at a time, in the order asked.
typedef struct WaifuPipeline WaifuPipeline;

// What a refused call or a job came to.
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
typedef int32_t WaifuDeviceProperty;

// What can be asked of a model. SUPPORTS_ properties are integers, 1 or 0; where 0, the string
// property of the same number is why, in words to show, and is never NULL. A SUPPORTS_ property
// this library does not know is absent, which is neither 1 nor 0. The rest are an integer or a
// string, as marked. SUPPORTS_<task> has the number of WAIFU_PIPELINE_<task>.
typedef int32_t WaifuModelProperty;

// What a pipeline is made for.
typedef int32_t WaifuPipelineType;

// 0: refused at the call, with waifu_last_error and _code, and no callback is ever called. Any
// other id: on_progress any number of times, then on_complete exactly once. Callbacks run on the
// pipeline's thread and may start before the _async call has returned: keep a job's state in
// user_data, not under its id. They must return quickly, and may call nothing but
// waifu_pipeline_cancel.
typedef uint64_t WaifuJob;

// What a job is doing. Ranges by task; done/total count what the comment says, 0, 0 otherwise.
typedef int32_t WaifuStage;

typedef struct {
  WaifuStage stage;
  // 0..1 of the whole job, or -1 where nothing can say.
  double fraction;
  // Of the stage, done already.
  uint64_t done;
  uint64_t total;
  // Since the job started.
  double seconds;
} WaifuProgress;

typedef void (*WaifuProgressCallback)(void *user_data, const WaifuProgress *progress);

typedef struct {
  WaifuStatusCode code;
  // Why, or NULL for WAIFU_OK.
  const char *message;
} WaifuStatus;

// Every pointer in a result is valid until the callback returns.
typedef struct {
  WaifuStatus status;
} WaifuResult;

typedef void (*WaifuCallback)(void *user_data, const WaifuResult *result);

typedef struct {
  uint32_t width;
  uint32_t height;
  // width * height * 3, rows top to bottom.
  const uint8_t *rgb;
  size_t len;
} WaifuImage;

typedef struct {
  // The <|key|> the prompt names it by: letters, digits and _.
  const char *key;
  const WaifuImage *image;
} WaifuPromptImage;

typedef struct {
  uint32_t struct_size;
  const char *prompt;
  // NULL or "" for none.
  const char *negative;
  // 64..2048, a multiple of IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF.
  uint32_t width;
  uint32_t height;
  uint32_t steps;
  float guidance;
  uint64_t seed;
  // img2img; NULL for none.
  const WaifuImage *start_image;
  // How far it walks away from it, 0..1.
  float start_image_strength;
  // A ControlNet's condition; NULL for none.
  const WaifuImage *control_image;
  // How hard it holds to it.
  float control_image_scale;
  const WaifuPromptImage *prompt_images;
  size_t prompt_image_count;
} WaifuImageGenerationRequest;

typedef struct {
  WaifuStatus status;
  // NULL unless WAIFU_OK.
  const WaifuImage *image;
} WaifuImageGenerationResult;

typedef void (*WaifuImageGenerationCallback)(void *user_data,
                                             const WaifuImageGenerationResult *result);

typedef struct {
  // Mono, -1..1.
  const float *samples;
  size_t count;
  uint32_t sample_rate;
} WaifuAudio;

typedef struct {
  uint32_t struct_size;
  const char *text;
  // A recording to sound like, or NULL.
  const WaifuAudio *reference;
  // What reference says, word for word; NULL or "" where it is not known. CosyVoice3 reads on
  // from it, as upstream's zero-shot does, and takes it or a style, not both; a model that
  // reads none ignores it.
  const char *reference_transcript;
  float speed;
  float temperature;
  // Any instruction; NULL or "" for the voice's own way. CosyVoice3 takes a style or a
  // reference_transcript, not both: upstream's instruct2 reads no transcript.
  const char *style;
  uint64_t seed;
} WaifuTextToSpeechRequest;

// Both speech tasks make audio.
typedef struct {
  WaifuStatus status;
  // NULL unless WAIFU_OK.
  const WaifuAudio *audio;
} WaifuAudioResult;

typedef void (*WaifuAudioCallback)(void *user_data, const WaifuAudioResult *result);

typedef struct {
  uint32_t struct_size;
  // What is said, and how.
  const WaifuAudio *source;
  // Whose voice to say it in.
  const WaifuAudio *reference;
  uint32_t steps;
  bool convert_style;
  uint64_t seed;
} WaifuVoiceConversionRequest;

// Text to speech with the audio handed over while the rest is said -- CosyVoice3's a hop of
// tokens at a time, as upstream's stream=True -- for a caller that plays the start at once. A
// load is refused at the call for a model that does not stream, and ends with WAIFU_ERR_MODEL
// where the device cannot: anywhere but CUDA at half, on Ampere or later, and Metal.
typedef struct {
  uint32_t struct_size;
  const char *text;
  // A recording to sound like, or NULL.
  const WaifuAudio *reference;
  // As WaifuTextToSpeechRequest's.
  const char *reference_transcript;
  float temperature;
  // As WaifuTextToSpeechRequest's.
  const char *style;
  // No speed: a streaming reading is at the speed it is said, as upstream's is.
  uint64_t seed;
} WaifuStreamingTextToSpeechRequest;

// A part of the reading's audio, in order; the samples are valid until it returns.
typedef void (*WaifuAudioPartCallback)(void *user_data, const WaifuAudio *part);

typedef struct {
  uint32_t width;
  uint32_t height;
  // width * height * 4, not premultiplied.
  const uint8_t *rgba;
  size_t len;
} WaifuRgbaImage;

typedef struct {
  uint32_t struct_size;
  // Any size, centred and scaled by the engine; transparent is fine.
  const WaifuRgbaImage *image;
  // 0 for the model's own.
  uint32_t steps;
  // 0 for the model's own.
  uint32_t depth_steps;
  // A multiple of LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF; 0 for 1280.
  uint32_t resolution;
  // Hands, eyes, eyebrows, ears into a -l and a -r layer.
  bool split_pairs;
  // Hair, hands, top and bottom also cut into a near and a far half.
  bool stratify_layers;
  uint64_t seed;
} WaifuLayerDecompositionRequest;

typedef int32_t WaifuLayerType;

typedef struct {
  WaifuLayerType type;
  // Unique within the result, from 1; it means nothing else.
  uint32_t id;
  // "topwear"; its halves "topwear-0" and "topwear-1"; its depth "topwear_depth".
  const char *name;
  // Where its top left corner is on the canvas.
  int32_t left;
  int32_t top;
  // Only as large as what is drawn. A depth layer: the same size and place as its layer, R, G
  // and B the depth, 0 nearest to 255 farthest, under its layer's alpha.
  WaifuRgbaImage image;
  // How far away it is: 0 nearest, 1 farthest. A near half has its part's depth, not its
  // pixels': that is what keeps it in its part's place in the stratified stack (see the
  // result). A depth layer has its layer's.
  float depth;
  // The id of the layer this one is made from: a half's part, a depth layer's layer; 0 for a
  // part.
  uint32_t derived_from_id;
} WaifuLayer;

// `layers` is in one order, which the library never changes: each part, bottom first -- which is
// the stack, unstratified -- then, with stratify_layers, each part's halves, near before far;
// then a depth layer for each of those, in their order. A part a half names in derived_from_id
// is a split part.
//
// The stratified stack is every part that was not split and every half, bottom first: from the
// largest depth to the smallest; at one depth, a far half under a near one; then by their order
// in `layers`. No depth layer is in either stack.
typedef struct {
  WaifuStatus status;
  // The side of the square canvas; 0 unless WAIFU_OK.
  uint32_t canvas;
  const WaifuLayer *layers;
  size_t layer_count;
} WaifuLayerDecompositionResult;

typedef void (*WaifuLayerDecompositionCallback)(void *user_data,
                                                const WaifuLayerDecompositionResult *result);

typedef struct {
  uint32_t struct_size;
  // Any rate, any length; resampled and cut into pieces by the engine.
  const WaifuAudio *audio;
  // The language it is in, one of SPEECH_TO_TEXT_LANGUAGES -- "English", "chinese" -- or NULL
  // or "" for the model to tell.
  const char *language;
  // Words that help it hear: names, terms, the subject; NULL or "".
  const char *context;
} WaifuSpeechToTextRequest;

typedef struct {
  WaifuStatus status;
  // What was said; "" for nothing said; NULL unless WAIFU_OK.
  const char *text;
  // The language it was said in, as the model names it -- "Chinese" -- or several, comma
  // separated, for a recording that changes; "" where none was heard. NULL unless WAIFU_OK.
  const char *language;
} WaifuSpeechToTextResult;

typedef void (*WaifuSpeechToTextCallback)(void *user_data, const WaifuSpeechToTextResult *result);

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

// The manner as well as the voice.
#define WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE 300

// Integer, bytes; absent until every file is here.
#define WAIFU_MODEL_SIZE_ON_DISK 1000

// Integer, of what it says or converts.
#define WAIFU_MODEL_SAMPLE_RATE 1001

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

#define WAIFU_OK 0

#define WAIFU_CANCELLED 1

// What a request cannot be run with. Most of it is refused at the call instead -- a NULL, a
// string that is not UTF-8, a size out of range, a model the pipeline's task is not supported by,
// another task's run -- which a job never sees.
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

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

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

// false for the processor, a device that cannot be used, or a property this library does not
// know. Asking CUDA builds a context on the card, a few hundred megabytes of it.
bool waifu_device_get_int_property(WaifuDevice device,
                                   WaifuDeviceProperty property,
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

// false where the model has no such property or this library does not know it; *value untouched.
bool waifu_model_get_int_property(const WaifuModel *model,
                                  WaifuModelProperty property,
                                  int64_t *value);

// NULL where there is none; valid until waifu_model_free.
const char *waifu_model_get_string_property(const WaifuModel *model, WaifuModelProperty property);

// Starts the pipeline's thread, with no model. Refused -- NULL, WAIFU_ERR_DEVICE_UNAVAILABLE --
// where the device is not available.
WaifuPipeline *waifu_pipeline_new(WaifuPipelineType type, WaifuDevice device);

// Stops what runs, cancels what waits, drops the model and joins the thread. Returns once every
// job has had its on_complete. Must not be called from one of its own callbacks. NULL is fine.
void waifu_pipeline_free(WaifuPipeline *pipeline);

// 0 for a NULL pipeline.
WaifuPipelineType waifu_pipeline_get_type(const WaifuPipeline *pipeline);

// Never AUTO, which is settled when it starts; AUTO only for a NULL pipeline.
WaifuDevice waifu_pipeline_get_device(const WaifuPipeline *pipeline);

// After the step it is on, or before it starts.
void waifu_pipeline_cancel(WaifuPipeline *pipeline, WaifuJob job);

void waifu_pipeline_cancel_all(WaifuPipeline *pipeline);

// Reads the model's weights onto the pipeline's device, dropping the model before first, so two
// are never on it at once. Refused at the call for a model the pipeline's task is not supported
// by (WAIFU_MODEL_SUPPORTS_<task>). The call copies the manifest, not the weights, which the
// pipeline's thread reads from the files: `model` may be freed as soon as it returns. A run may
// be asked for straight after; it waits its turn. "tones", the stand-in voice the tests use,
// opens as a model too.
WaifuJob waifu_pipeline_load_model_async(WaifuPipeline *pipeline,
                                         const WaifuModel *model,
                                         void *user_data,
                                         WaifuProgressCallback on_progress,
                                         WaifuCallback on_complete);

WaifuJob waifu_pipeline_unload_model_async(WaifuPipeline *pipeline,
                                           void *user_data,
                                           WaifuCallback on_complete);

// Each run function is refused at the call on a pipeline made for another task.
WaifuJob waifu_pipeline_run_image_generation_async(WaifuPipeline *pipeline,
                                                   const WaifuImageGenerationRequest *request,
                                                   void *user_data,
                                                   WaifuProgressCallback on_progress,
                                                   WaifuImageGenerationCallback on_complete);

WaifuJob waifu_pipeline_run_text_to_speech_async(WaifuPipeline *pipeline,
                                                 const WaifuTextToSpeechRequest *request,
                                                 void *user_data,
                                                 WaifuProgressCallback on_progress,
                                                 WaifuAudioCallback on_complete);

WaifuJob waifu_pipeline_run_voice_conversion_async(WaifuPipeline *pipeline,
                                                   const WaifuVoiceConversionRequest *request,
                                                   void *user_data,
                                                   WaifuProgressCallback on_progress,
                                                   WaifuAudioCallback on_complete);

// on_part for each part as it is made; on_complete with the whole of it, the parts end to end.
WaifuJob waifu_pipeline_run_streaming_text_to_speech_async(WaifuPipeline *pipeline,
                                                           const WaifuStreamingTextToSpeechRequest *request,
                                                           void *user_data,
                                                           WaifuProgressCallback on_progress,
                                                           WaifuAudioPartCallback on_part,
                                                           WaifuAudioCallback on_complete);

WaifuJob waifu_pipeline_run_layer_decomposition_async(WaifuPipeline *pipeline,
                                                      const WaifuLayerDecompositionRequest *request,
                                                      void *user_data,
                                                      WaifuProgressCallback on_progress,
                                                      WaifuLayerDecompositionCallback on_complete);

WaifuJob waifu_pipeline_run_speech_to_text_async(WaifuPipeline *pipeline,
                                                 const WaifuSpeechToTextRequest *request,
                                                 void *user_data,
                                                 WaifuProgressCallback on_progress,
                                                 WaifuSpeechToTextCallback on_complete);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* LIBWAIFU_WAIFU_H */
