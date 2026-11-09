# libwaifu's C API, ABI 5

Status: implemented. `waifu/include/waifu.h`, generated from `waifu/src/c_api.rs`, is ABI 5: ABI 4
with the one change under [What changes from ABI 4](#what-changes-from-abi-4). The rest of this
document was written for ABI 4, and holds for ABI 5. What is still open is under
[Not decided](#not-decided).

In the header every enum is an `int32_t` and a `#define` for each value -- `typedef int32_t
WaifuStage;` and `#define WAIFU_STAGE_LOADING_WEIGHTS 1` -- rather than the `typedef enum` the
listing below shows: a number the library or a caller does not know is then a plain integer,
and not a value a Rust or C++ enum must never hold. The numbers and names are the listing's.

ABI 4 takes every JSON document out of libwaifu and asks about models and devices through an enum
and a getter instead; replaces the engine that ran every kind of job with a pipeline made for one
task; names each task the same way everywhere; and drops what only a test or only one screen
needed. A caller built against it has two things to free and nothing to parse.

## Rules

- **No JSON.** What a model can do, what a device has and which build this is are asked one
  property at a time: `waifu_model_get_int_property(model, WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION, &v)`.
- **Enums only grow.** A number, once published, keeps its meaning and is never reused; a new
  property, stage or capability is a new number at the end of its range, and is not an ABI change.
  A library asked about a number it does not know answers as for a property the model does not
  have: `false`. A caller handed a stage it does not know shows a generic line.
- **Requests carry `struct_size`**, as in ABI 3. A library reads the part of a request it knows,
  so a caller built against a newer header, with more fields, is still read. A request smaller
  than the library's is refused at the call, and nothing past its `struct_size` is read. So a
  field that a request gains is an ABI change: ABI 5 is the text-to-speech requests'
  `reference_transcript`.
- **Threads.** Every function may be called from any thread, and calls on one pipeline or one
  model may run at the same time -- loads, runs, cancels and getters -- except that **freeing a
  handle must not race with any other call on it**: that is the caller's to order, as with any C
  handle. A pipeline's work runs on its own thread, and its callbacks come from there; a log line
  comes from whichever thread wrote it.
- **Freeing waits.** `waifu_pipeline_free` cancels what waits, stops what runs after its step, and
  returns only once every job of the pipeline has had its `on_complete`; after it returns, nothing
  of the pipeline calls back again. It must not be called from one of its own callbacks.
- **A job's state is its `user_data`.** Its callbacks run on the pipeline's thread and may start
  before the `_async` call that started it has returned -- a job can be over by then -- so a
  caller keeps what a job needs where `user_data` points, set up before the call, and not in a
  table keyed by the id the call returns, as IOCP's `OVERLAPPED` and gRPC's tags are. (libuv
  can promise otherwise because its callbacks run on the caller's own thread; these run on the
  pipeline's.) A job given an id gets `on_progress` any number of times and then `on_complete`
  exactly once; a call that returned 0 gets neither.
- **Two ways to fail.** A call that is *refused* says so at once -- a NULL from a constructor, job
  0 from an `_async` call, `false` from `waifu_device_set_memory_limit` -- and sets this thread's
  `waifu_last_error` (a sentence) and `waifu_last_error_code` (a `WaifuStatusCode`) together. A
  *query* is never refused: a getter's `false` or NULL means the value is not there, and sets
  nothing, so the last error may be an old one. A job that fails once running ends with a
  `WaifuStatus` in its result instead.
- **Two things to free**: a pipeline (`waifu_pipeline_free`) and a model (`waifu_model_free`).
  Everything else is the library's: a model's strings live until the model is freed, the version
  strings forever, the last error until this thread's next refused call, and everything a callback
  is handed until it returns. A caller that keeps a result copies it.
- **Requests are copied at the call.** Nothing a caller passes in has to outlive the call that took
  it -- not a request, not an image in one, not the `WaifuModel` a load is given.
- **One name per task, a noun phrase**, the field's or the upstream project's own:
  `IMAGE_GENERATION`, `TEXT_TO_SPEECH`, `STREAMING_TEXT_TO_SPEECH`, `SPEECH_TO_TEXT`,
  `VOICE_CONVERSION`, `LAYER_DECOMPOSITION` (See-through's
  paper is "Single-image Layer Decomposition"). It is the same in the pipeline type, the run
  function, the request, the result, the stages and the capability. A capability is `SUPPORTS_`
  and the task; a constraint is the request field it constrains and the constraint
  (`IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF`).
- **Nothing only a test needs.** `"tones"`, the stand-in voice, still opens and loads -- the tests
  run on it without a download -- but no property, enum or string of the API is there for it.
- **Encoding files is the caller's.** The library hands over pixels and samples; PNG, WAV and PSD
  are written by the app.

## The header

```c
#define WAIFU_ABI_VERSION 5

/* -- enums --------------------------------------------------------------------------------- */

typedef enum {
  WAIFU_LOG_DEBUG = 0,
  WAIFU_LOG_INFO = 1,
  WAIFU_LOG_WARNING = 2,
  WAIFU_LOG_ERROR = 3,
  WAIFU_LOG_FATAL = 4,        // the process aborts when the callback returns
} WaifuLogLevel;

typedef enum {
  WAIFU_DEVICE_AUTO = 0,
  WAIFU_DEVICE_CPU = 1,
  WAIFU_DEVICE_METAL = 2,
  WAIFU_DEVICE_CUDA = 3,
  WAIFU_DEVICE_VULKAN = 5,
  // 4 and 6 were CUDA_CPU_OFFLOAD and METAL_DISK_OFFLOAD, aliases kept for older headers: not reused
} WaifuDevice;

// What a pipeline is made for.
typedef enum {
  WAIFU_PIPELINE_IMAGE_GENERATION = 1,
  WAIFU_PIPELINE_TEXT_TO_SPEECH = 2,
  WAIFU_PIPELINE_VOICE_CONVERSION = 3,
  WAIFU_PIPELINE_LAYER_DECOMPOSITION = 4,
  WAIFU_PIPELINE_SPEECH_TO_TEXT = 5,
  WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH = 6,
} WaifuPipelineType;

typedef enum {
  WAIFU_OK = 0,
  WAIFU_CANCELLED = 1,
  // What a request cannot be run with. Most of it is refused at the call instead -- a NULL, a
  // string that is not UTF-8, a size out of range, a model the pipeline's task is not supported
  // by, another task's run -- which a job never sees.
  WAIFU_ERR_INVALID_ARGUMENT = 2,
  WAIFU_ERR_UNKNOWN_MODEL = 3,     // no manifest at the path
  // 4 was FETCH, which libwaifu never said once downloads were libwaifu_modelmanager's: not reused
  WAIFU_ERR_MODEL = 5,             // the model could not be read, or cannot do what was asked
  WAIFU_ERR_NOT_LOADED = 6,        // a run on a pipeline with no model loaded
  WAIFU_ERR_INTERNAL = 7,          // a panic in the library: a bug
  WAIFU_ERR_DEVICE_UNAVAILABLE = 8,  // a pipeline asked for on a device this build or machine cannot use
} WaifuStatusCode;

// What a job is doing. Ranges by task; done/total count what the comment says, 0, 0 otherwise.
typedef enum {
  WAIFU_STAGE_LOADING_WEIGHTS = 1,

  WAIFU_STAGE_IMAGE_GENERATION_ENCODING_PROMPT = 100,
  WAIFU_STAGE_IMAGE_GENERATION_DENOISING = 101,             // steps
  WAIFU_STAGE_IMAGE_GENERATION_DECODING_IMAGE = 102,

  WAIFU_STAGE_TEXT_TO_SPEECH_READING_TEXT = 200,
  WAIFU_STAGE_TEXT_TO_SPEECH_GENERATING_TOKENS = 201,       // tokens; total is an estimate, and can be passed
  WAIFU_STAGE_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO = 202,

  WAIFU_STAGE_VOICE_CONVERSION_READING_RECORDINGS = 300,
  WAIFU_STAGE_VOICE_CONVERSION_GENERATING_TOKENS = 301,     // tokens; total is an estimate
  WAIFU_STAGE_VOICE_CONVERSION_DENOISING = 302,             // steps
  WAIFU_STAGE_VOICE_CONVERSION_SYNTHESIZING_AUDIO = 303,    // vocoder windows

  // The whole picture, then the head, then the depth model that orders the layers.
  WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_IMAGE = 400,
  WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_LAYERS = 401,       // steps
  WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_LAYERS = 402,        // layers
  WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_HEAD = 403,
  WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_HEAD_LAYERS = 404,  // steps
  WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_HEAD_LAYERS = 405,   // layers
  WAIFU_STAGE_LAYER_DECOMPOSITION_ENCODING_FOR_DEPTH = 406,     // layers
  WAIFU_STAGE_LAYER_DECOMPOSITION_DENOISING_DEPTH = 407,        // steps
  WAIFU_STAGE_LAYER_DECOMPOSITION_DECODING_DEPTH = 408,         // layers
  WAIFU_STAGE_LAYER_DECOMPOSITION_CUTTING_INTO_PARTS = 409,

  WAIFU_STAGE_SPEECH_TO_TEXT_READING_RECORDING = 500,       // pieces the recording is cut into
  WAIFU_STAGE_SPEECH_TO_TEXT_GENERATING_TOKENS = 501,       // tokens; total is an estimate, and can be passed

  // Saying and sounding take turns, a hop of tokens at a time.
  WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_READING_TEXT = 600,
  WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_GENERATING_TOKENS = 601,     // tokens; total is an estimate
  WAIFU_STAGE_STREAMING_TEXT_TO_SPEECH_SYNTHESIZING_AUDIO = 602,    // parts handed over
} WaifuStage;

// What can be asked of a model. SUPPORTS_ properties are integers, 1 or 0; where 0, the string
// property of the same number is why, in words to show, and is never NULL. A SUPPORTS_ property
// this library does not know is absent, which is neither 1 nor 0. The rest are an integer or a
// string, as marked. SUPPORTS_<task> has the number of WAIFU_PIPELINE_<task>.
typedef enum {
  WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION = 1,
  WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH = 2,
  WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION = 3,
  WAIFU_MODEL_SUPPORTS_LAYER_DECOMPOSITION = 4,
  WAIFU_MODEL_SUPPORTS_SPEECH_TO_TEXT = 5,
  WAIFU_MODEL_SUPPORTS_STREAMING_TEXT_TO_SPEECH = 6,             // the model streams; the device is checked at load

  WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_FROM_START_IMAGE = 100,  // img2img
  WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_CONTROL_IMAGE = 101,  // a ControlNet
  WAIFU_MODEL_SUPPORTS_IMAGE_GENERATION_WITH_PROMPT_IMAGES = 102,  // images the prompt names as <|key|>

  WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_LIKE_REFERENCE = 200,      // in the voice of a recording
  WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE = 201,   // CosyVoice3 does not
  WAIFU_MODEL_SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE = 202,          // follows an instruction such as "speak happily"

  WAIFU_MODEL_SUPPORTS_VOICE_CONVERSION_OF_STYLE = 300,          // the manner as well as the voice

  WAIFU_MODEL_SIZE_ON_DISK = 1000,                    // integer, bytes; absent until every file is here
  WAIFU_MODEL_SAMPLE_RATE = 1001,                     // integer, of what it says or converts
  WAIFU_MODEL_IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF = 1100,  // integer: 16, or 32 for SDXL and Qwen-Image
  WAIFU_MODEL_IMAGE_GENERATION_SAMPLER = 1101,                   // string: "Euler"
  WAIFU_MODEL_IMAGE_GENERATION_MAX_PROMPT_IMAGES = 1102,         // integer
  WAIFU_MODEL_LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF = 1400, // integer: 64
  WAIFU_MODEL_SPEECH_TO_TEXT_LANGUAGES = 1500,                   // string: "Chinese,English,...", comma separated
} WaifuModelProperty;

// What can be asked of a device. All of it is this process's, not the card as other programs see it.
typedef enum {
  WAIFU_DEVICE_MEMORY_BUDGET = 1,     // the most it may hold there: the memory limit, or the device's memory
  WAIFU_DEVICE_MEMORY_HELD = 2,       // what its allocator holds now, what it keeps for reuse included
  WAIFU_DEVICE_MEMORY_AVAILABLE = 3,  // BUDGET - HELD, never below 0
  WAIFU_DEVICE_MEMORY_LIMIT = 4,      // what waifu_device_set_memory_limit set; absent where nothing did
} WaifuDeviceProperty;

typedef enum {
  WAIFU_LAYER_PART = 0,             // a part as the model drew it
  WAIFU_LAYER_STRATIFIED_NEAR = 1,  // stratify_layers: the near half of a part
  WAIFU_LAYER_STRATIFIED_FAR = 2,   // stratify_layers: the far half of a part
  WAIFU_LAYER_DEPTH = 3,            // the depth of the layer derived_from_id names, pixel by pixel
} WaifuLayerType;

/* -- the library --------------------------------------------------------------------------- */

uint32_t waifu_abi_version(void);
// This thread's last refused call, as a sentence and as a code; both are set together by a refused
// call and left alone by every other, and reading one does not change the other. The string is
// valid until this thread's next refused call. "" and WAIFU_OK before any.
const char *waifu_last_error(void);
WaifuStatusCode waifu_last_error_code(void);
const char *waifu_version(void);      // "0.1.0"; static
const char *waifu_revision(void);     // the full commit, or NULL for a build made without git; static

typedef struct {
  WaifuLogLevel level;
  const char *source;                 // "engine.rs:712"
  const char *message;
  const char *time;                   // "2026-10-23T18:50:41Z"
} WaifuLog;

// Called on whichever thread wrote the line, which can be the caller's own, possibly from several
// threads at once. Must not call into the library.
typedef void (*WaifuLogCallback)(void *user_data, const WaifuLog *log);
// NULL: the console again. Waits for a line being handed to the callback before to finish: once it
// returns, the callback and user_data before are never called again, and may be let go of.
void waifu_set_log_callback(void *user_data, WaifuLogCallback callback);
void waifu_set_log_level(WaifuLogLevel level);                            // INFO unless set

/* -- devices ------------------------------------------------------------------------------- */

bool waifu_device_is_available(WaifuDevice device);

// false for the processor, a device that cannot be used, or a property this library does not
// know. Asking CUDA builds a context on the card, a few hundred megabytes of it.
bool waifu_device_get_int_property(WaifuDevice device, WaifuDeviceProperty property, int64_t *value);

// For the whole process, one per device, shared by every pipeline on it: set it before a model is
// loaded. 0 for no limit but the device's. The processor has none, and is refused.
bool waifu_device_set_memory_limit(WaifuDevice device, uint64_t bytes);

/* -- models -------------------------------------------------------------------------------- */

typedef struct WaifuModel WaifuModel;

// Reads the manifest at `manifest_path` -- a few kilobytes, and none of the weights. Refused --
// NULL -- with WAIFU_ERR_UNKNOWN_MODEL where there is none, WAIFU_ERR_MODEL where it cannot be read.
WaifuModel *waifu_model_open(const char *manifest_path);
void waifu_model_free(WaifuModel *model);

// false where the model has no such property or this library does not know it; *value untouched.
bool waifu_model_get_int_property(const WaifuModel *model, WaifuModelProperty property, int64_t *value);
// NULL where there is none; valid until waifu_model_free.
const char *waifu_model_get_string_property(const WaifuModel *model, WaifuModelProperty property);

/* -- pipelines ----------------------------------------------------------------------------- */

// One task, one thread, at most one model: jobs run on it one at a time, in the order asked.
typedef struct WaifuPipeline WaifuPipeline;

// Starts the pipeline's thread, with no model. Refused -- NULL, WAIFU_ERR_DEVICE_UNAVAILABLE --
// where the device is not available.
WaifuPipeline *waifu_pipeline_new(WaifuPipelineType type, WaifuDevice device);
// Stops what runs, cancels what waits, drops the model and joins the thread.
void waifu_pipeline_free(WaifuPipeline *pipeline);
WaifuPipelineType waifu_pipeline_get_type(const WaifuPipeline *pipeline);
WaifuDevice waifu_pipeline_get_device(const WaifuPipeline *pipeline);   // never AUTO

/* -- jobs ---------------------------------------------------------------------------------- */

// 0: refused at the call, with waifu_last_error and _code, and no callback is ever called.
// Any other id: on_progress any number of times, then on_complete exactly once. Callbacks run on
// the pipeline's thread and may start before the _async call has returned: keep a job's state in
// user_data, not under its id. They must return quickly, and may call nothing but
// waifu_pipeline_cancel.
typedef uint64_t WaifuJob;

void waifu_pipeline_cancel(WaifuPipeline *pipeline, WaifuJob job);  // after the step it is on, or before it starts
void waifu_pipeline_cancel_all(WaifuPipeline *pipeline);

typedef struct {
  WaifuStage stage;
  double fraction;                    // 0..1 of the whole job, or -1 where nothing can say
  uint64_t done;                      // of the stage, done already
  uint64_t total;
  double seconds;                     // since the job started
} WaifuProgress;
typedef void (*WaifuProgressCallback)(void *user_data, const WaifuProgress *progress);

typedef struct {
  WaifuStatusCode code;
  const char *message;                // why, or NULL for WAIFU_OK
} WaifuStatus;

// Every pointer in a result is valid until the callback returns.
typedef struct {
  WaifuStatus status;
} WaifuResult;
typedef void (*WaifuCallback)(void *user_data, const WaifuResult *result);

// Reads the model's weights onto the pipeline's device, dropping the model before first, so two
// are never on it at once. Refused at the call for a model the pipeline's task is not supported
// by (WAIFU_MODEL_SUPPORTS_<task>). The call copies the manifest, not the weights, which the
// pipeline's thread reads from the files: `model` may be freed as soon as it returns. A run may be
// asked for straight after; it waits its turn. "tones", the stand-in voice the tests use, opens
// as a model too.
WaifuJob waifu_pipeline_load_model_async(WaifuPipeline *pipeline, const WaifuModel *model, void *user_data,
                                         WaifuProgressCallback on_progress, WaifuCallback on_complete);
WaifuJob waifu_pipeline_unload_model_async(WaifuPipeline *pipeline, void *user_data, WaifuCallback on_complete);

/* -- image generation ---------------------------------------------------------------------- */

typedef struct {
  uint32_t width;
  uint32_t height;
  const uint8_t *rgb;                 // width * height * 3, rows top to bottom
  size_t len;
} WaifuImage;

typedef struct {
  const char *key;                    // the <|key|> the prompt names it by: letters, digits and _
  const WaifuImage *image;
} WaifuPromptImage;

typedef struct {
  uint32_t struct_size;
  const char *prompt;
  const char *negative;               // NULL or "" for none
  uint32_t width;                     // 64..2048, a multiple of IMAGE_GENERATION_WIDTH_HEIGHT_MULTIPLE_OF
  uint32_t height;
  uint32_t steps;
  float guidance;
  uint64_t seed;
  const WaifuImage *start_image;      // img2img; NULL for none
  float start_image_strength;         // how far it walks away from it, 0..1
  const WaifuImage *control_image;    // a ControlNet's condition; NULL for none
  float control_image_scale;          // how hard it holds to it
  const WaifuPromptImage *prompt_images;
  size_t prompt_image_count;
} WaifuImageGenerationRequest;

typedef struct {
  WaifuStatus status;
  const WaifuImage *image;            // NULL unless WAIFU_OK
} WaifuImageGenerationResult;
typedef void (*WaifuImageGenerationCallback)(void *user_data, const WaifuImageGenerationResult *result);

// Each run function is refused at the call on a pipeline made for another task.
WaifuJob waifu_pipeline_run_image_generation_async(WaifuPipeline *pipeline,
                                                   const WaifuImageGenerationRequest *request,
                                                   void *user_data, WaifuProgressCallback on_progress,
                                                   WaifuImageGenerationCallback on_complete);

/* -- text to speech and voice conversion --------------------------------------------------- */

typedef struct {
  const float *samples;               // mono, -1..1
  size_t count;
  uint32_t sample_rate;
} WaifuAudio;

typedef struct {
  uint32_t struct_size;
  const char *text;
  const WaifuAudio *reference;        // a recording to sound like, or NULL
  const char *reference_transcript;   // what reference says; NULL or "" where it is not known
  float speed;
  float temperature;
  const char *style;                  // any instruction; NULL or "" for the voice's own way
  uint64_t seed;
} WaifuTextToSpeechRequest;

typedef struct {
  uint32_t struct_size;
  const WaifuAudio *source;           // what is said, and how
  const WaifuAudio *reference;        // whose voice to say it in
  uint32_t steps;
  bool convert_style;
  uint64_t seed;
} WaifuVoiceConversionRequest;

// Both speech tasks make audio.
typedef struct {
  WaifuStatus status;
  const WaifuAudio *audio;            // NULL unless WAIFU_OK
} WaifuAudioResult;
typedef void (*WaifuAudioCallback)(void *user_data, const WaifuAudioResult *result);

WaifuJob waifu_pipeline_run_text_to_speech_async(WaifuPipeline *pipeline,
                                                 const WaifuTextToSpeechRequest *request,
                                                 void *user_data, WaifuProgressCallback on_progress,
                                                 WaifuAudioCallback on_complete);
WaifuJob waifu_pipeline_run_voice_conversion_async(WaifuPipeline *pipeline,
                                                   const WaifuVoiceConversionRequest *request,
                                                   void *user_data, WaifuProgressCallback on_progress,
                                                   WaifuAudioCallback on_complete);

/* -- streaming text to speech -------------------------------------------------------------- */

// Text to speech with the audio handed over while the rest is said -- CosyVoice3's a hop of
// tokens at a time, as upstream's stream=True -- for a caller that plays the start at once. Its
// audio is another reading of the same tokens, not text to speech's samples, and it takes longer
// in all. A load is refused at the call for a model that does not stream, and ends with
// WAIFU_ERR_MODEL where the device cannot: anywhere but CUDA at half, on Ampere or later, and Metal.

typedef struct {
  uint32_t struct_size;
  const char *text;
  const WaifuAudio *reference;        // a recording to sound like, or NULL
  const char *reference_transcript;   // as WaifuTextToSpeechRequest's
  float temperature;
  const char *style;                  // any instruction; NULL or "" for the voice's own way
  uint64_t seed;
  // No speed: a streaming reading is at the speed it is said, as upstream's is.
} WaifuStreamingTextToSpeechRequest;

// A part of the reading's audio, in order; the samples are valid until it returns.
typedef void (*WaifuAudioPartCallback)(void *user_data, const WaifuAudio *part);

// on_part for each part as it is made; on_complete with the whole of it, the parts end to end.
WaifuJob waifu_pipeline_run_streaming_text_to_speech_async(WaifuPipeline *pipeline,
                                                           const WaifuStreamingTextToSpeechRequest *request,
                                                           void *user_data, WaifuProgressCallback on_progress,
                                                           WaifuAudioPartCallback on_part,
                                                           WaifuAudioCallback on_complete);

/* -- layer decomposition ------------------------------------------------------------------- */

typedef struct {
  uint32_t width;
  uint32_t height;
  const uint8_t *rgba;                // width * height * 4, not premultiplied
  size_t len;
} WaifuRgbaImage;

typedef struct {
  uint32_t struct_size;
  const WaifuRgbaImage *image;        // any size, centred and scaled by the engine; transparent is fine
  uint32_t steps;                     // 0 for the model's own
  uint32_t depth_steps;               // 0 for the model's own
  uint32_t resolution;                // a multiple of LAYER_DECOMPOSITION_RESOLUTION_MULTIPLE_OF; 0 for 1280
  bool split_pairs;                   // hands, eyes, eyebrows, ears into a -l and a -r layer
  bool stratify_layers;               // hair, hands, top and bottom also cut into a near and a far half
  uint64_t seed;
} WaifuLayerDecompositionRequest;

typedef struct {
  WaifuLayerType type;
  uint32_t id;                        // unique within the result, from 1; it means nothing else
  const char *name;                   // "topwear"; its halves "topwear-0", "topwear-1"; its depth "topwear_depth"
  int32_t left;                       // where its top left corner is on the canvas
  int32_t top;
  // Only as large as what is drawn. A depth layer: its layer's size and place, R, G and B the
  // depth, 0 nearest to 255 farthest, under its layer's alpha.
  WaifuRgbaImage image;
  // How far away it is: 0 nearest, 1 farthest. A near half has its part's depth, not its pixels':
  // that is what keeps it in its part's place in the stratified stack (see the result). A depth
  // layer has its layer's.
  float depth;
  uint32_t derived_from_id;           // a half: its part's id; a depth layer: its layer's; a part: 0
} WaifuLayer;

// `layers` is in one order, which the library never changes: each part, bottom first -- which is
// the stack, unstratified -- then, with stratify_layers, each part's halves, near before far; then
// a depth layer for each of those, in their order. A part a half names in derived_from_id is a
// split part. Links are by id, not by place, so a caller's copy may be reordered or cut.
//
// The stratified stack is every part that was not split and every half, bottom first: from the
// largest depth to the smallest; at one depth, a far half under a near one; then by their order in
// `layers`. No depth layer is in either stack.
typedef struct {
  WaifuStatus status;
  uint32_t canvas;                    // the side of the square canvas; 0 unless WAIFU_OK
  const WaifuLayer *layers;
  size_t layer_count;
} WaifuLayerDecompositionResult;
typedef void (*WaifuLayerDecompositionCallback)(void *user_data, const WaifuLayerDecompositionResult *result);

WaifuJob waifu_pipeline_run_layer_decomposition_async(WaifuPipeline *pipeline,
                                                      const WaifuLayerDecompositionRequest *request,
                                                      void *user_data, WaifuProgressCallback on_progress,
                                                      WaifuLayerDecompositionCallback on_complete);

/* -- speech to text ------------------------------------------------------------------------ */

typedef struct {
  uint32_t struct_size;
  const WaifuAudio *audio;            // any rate, any length; resampled and cut into pieces by the engine
  // The language it is in, one of SPEECH_TO_TEXT_LANGUAGES -- "English", "chinese" -- or NULL or ""
  // for the model to tell.
  const char *language;
  const char *context;                // words that help it hear: names, terms, the subject; NULL or ""
} WaifuSpeechToTextRequest;

typedef struct {
  WaifuStatus status;
  const char *text;                   // what was said; "" for nothing said; NULL unless WAIFU_OK
  // The language it was said in, as the model names it -- "Chinese" -- or several, comma
  // separated, for a recording that changes; "" where none was heard. NULL unless WAIFU_OK.
  const char *language;
} WaifuSpeechToTextResult;
typedef void (*WaifuSpeechToTextCallback)(void *user_data, const WaifuSpeechToTextResult *result);

WaifuJob waifu_pipeline_run_speech_to_text_async(WaifuPipeline *pipeline,
                                                 const WaifuSpeechToTextRequest *request,
                                                 void *user_data, WaifuProgressCallback on_progress,
                                                 WaifuSpeechToTextCallback on_complete);
```

## What changes from ABI 4

| ABI 4 | ABI 5 |
|---|---|
| `WaifuTextToSpeechRequest`, `WaifuStreamingTextToSpeechRequest`: `text`, `reference`, ... | `const char *reference_transcript` after `reference`: the words of the recording, which CosyVoice3 reads zero-shot (see "A reference transcript" under [Why](#why)) |

That is all. A field added at the end could have been read at the older size, as NULL, and no
ABI change; it was not done that way, because that reads a request through a pointer to a struct
larger than the one the caller handed over, and takes on a rule for every size in between. The
field sits beside the recording it is the words of, and a caller built against ABI 4 is turned
away by `waifu_abi_version` before it gets as far as a request.

## What changes from ABI 3

| ABI 3 | ABI 4 |
|---|---|
| `WaifuEngine`, `waifu_engine_new`, `waifu_engine_new_with_options`, `WaifuEngineOptions` | `WaifuPipeline`, `waifu_pipeline_new(type, device)` |
| `waifu_engine_free`, `_cancel`, `_cancel_all` | `waifu_pipeline_free`, `_cancel`, `_cancel_all` |
| `waifu_engine_device` returns `char *` | `waifu_pipeline_get_device` returns `WaifuDevice`; `waifu_pipeline_get_type` |
| `waifu_engine_load_async(engine, "path")`, `_unload_async` | `waifu_pipeline_load_model_async(pipeline, model)`, `_unload_model_async` |
| `waifu_engine_draw_async`, `WaifuDrawRequest` | `waifu_pipeline_run_image_generation_async`, `WaifuImageGenerationRequest` |
| `waifu_engine_speak_async`, `WaifuSpeakRequest` | `waifu_pipeline_run_text_to_speech_async`, `WaifuTextToSpeechRequest` |
| `waifu_engine_speak_stream_async`, `on_audio` | a pipeline of its own: `WAIFU_PIPELINE_STREAMING_TEXT_TO_SPEECH`, `waifu_pipeline_run_streaming_text_to_speech_async`, `WaifuStreamingTextToSpeechRequest` (no `speed`), `on_part` |
| `waifu_engine_transcribe_async`, `WaifuTranscribeRequest`, `WaifuTranscriptCompletionEvent` | `waifu_pipeline_run_speech_to_text_async`, `WaifuSpeechToTextRequest`, `WaifuSpeechToTextResult` |
| a transcriber's `languages` list | `SPEECH_TO_TEXT_LANGUAGES`, one comma-separated string |
| `waifu_engine_voice_conversion_async` | `waifu_pipeline_run_voice_conversion_async` |
| `waifu_engine_layers_async`, `WaifuLayersRequest` | `waifu_pipeline_run_layer_decomposition_async`, `WaifuLayerDecompositionRequest` |
| `…CompletionEvent`, `…CompleteCallback` | `…Result`, `…Callback` |
| `WaifuProgressEvent`, `WaifuLogEvent` | `WaifuProgress`, `WaifuLog` |
| `waifu_describe_json` | `waifu_model_open`, `waifu_model_get_int_property`, `waifu_model_get_string_property`, `waifu_model_free` |
| `waifu_devices_json` | `waifu_device_is_available`, `waifu_device_get_int_property` |
| `waifu_set_memory_limit` | `waifu_device_set_memory_limit` |
| `waifu_version_json` | `waifu_version`, `waifu_revision` |
| `waifu_string_free` | gone: nothing returns a string to free |
| `waifu_last_error` alone | with `waifu_last_error_code`, set by a refused call only; `WAIFU_ERR_DEVICE_UNAVAILABLE` |
| images under the keys `"start_from"` and `"control"` | `start_image`, `control_image`; `WaifuImageInput` is `WaifuPromptImage` |
| `WaifuSpeakRequest.like`, `WaifuAudio.rate` | `reference`, `sample_rate` |
| `WaifuProgressEvent.words`, `part`, `parts`, `file` | gone: one `WaifuStage` for each thing a job does |
| eight `WaifuStage`s | twenty-six, by task |
| `waifu_psd_write`, `waifu_bytes_free`, PSDs in the layers' result | gone: a PSD is the app's to write; each layer's depth map is a `WAIFU_LAYER_DEPTH` layer |
| `waifu_stratify`, `waifu_depth_layers_free` | `stratify_layers` in the request |
| `WAIFU_DEVICE_CUDA_CPU_OFFLOAD`, `_METAL_DISK_OFFLOAD`, `WAIFU_ERR_FETCH` | gone; their numbers are not reused |
| a voice's `styles`, `not_a_voice_because` | gone |

## Why

**A pipeline per task, one type.** The engine ran whatever its model happened to do, so a draw
asked of a voice was accepted and failed at the end, and `NOT_LOADED` meant both "nothing is
loaded" and "the wrong thing is". A pipeline is made for one task: a model that task is not
supported by, or another task's run, is refused at the call, and `NOT_LOADED` is only a run with no
model. It stays one opaque type with one `new` and one set of shared functions -- a type per task,
each with its own copy of all of them, were tried and were twice the API for nothing a caller
needs. It is created empty and loaded on its own queue, as the engine was, so a load and a run can
be asked for back to back, and a new model is only read once the old one is dropped.

CosyVoice3 does three tasks -- text to speech, streaming text to speech and voice conversion; used
for more than one, it is loaded into a pipeline for each and is on the device as many times. A
call that lets a second pipeline share the first's weights can be added later without changing
anything here. Pipelines on one device run side by side, as two engines did, and share its memory
limit.

**Streaming text to speech is a pipeline of its own.** It reads with the same voice, and is
nonetheless another task: its flow draws in blocks and redraws each sentence a hop at a time, its
audio is not text to speech's, it takes longer in all and is heard sooner, it runs on one kind of
device, and it has no speed. Its own pipeline says each of those where it belongs -- a request
without `speed`, a load that fails on a device that cannot stream rather than every run, its own
stages -- and the implementation is to be split along the same line. A caller that offers both
loads the voice twice.

Speech to text is `SPEECH_TO_TEXT`, text to speech's own name turned round and the task's name in
this repository (`speech2text`). The languages a transcriber can be told are a list, and a property
is one value: they are one string, comma separated, as its result already gives several languages.

**Properties, not JSON and not structs.** JSON was written for one screen: a flag beside the
sentence saying why not (`takes_a_recording`, `no_likeness_because`), a map keyed by magic strings
(`why_not`), a `kind` and a `kinds`, the path asked with handed back. Structs in its place would
have been cleaner, and every field added to one an ABI change for every caller. A property is a
number: one more is a number more, and an old library asked about it says so instead of misreading
a layout.

**The description stays in libwaifu.** libwaifu_modelmanager already keeps a table by model type
for a screen's defaults, and describing a model there would have been a second copy of what the
engine itself checks -- which sizes a model draws at, whether it reads a picture -- free to drift
from it.

**A model is a handle, and a load takes one.** The path is checked when it is opened, at the call,
rather than at the end of a job; what a model can do is asked of the same thing that is then
loaded; and a caller that has already looked does not name the file twice.

**Stages, not words.** `words` -- "step 3 of 20", "ordering the layers" -- carried what `WaifuStage`
could not: a decomposition's seven stages, two of them run twice, were folded into a picture's
encoding, drawing and decoding. Each is a stage now, and the words are the app's, in its own
language.

**No style list.** `CosyVoice3::STYLES` is upstream's `instruct_list`
(`cosyvoice/utils/common.py`), which its README calls instructions "such as" these: examples, not
the set. `style` is any instruction, and `SUPPORTS_TEXT_TO_SPEECH_WITH_STYLE` says whether one is
followed; the examples are the app's to offer.

**A reference transcript, and no property for it.** `reference_transcript` is the words of
`reference`. CosyVoice3 then reads zero-shot, as upstream's `inference_zero_shot` does with its
`prompt_text`: it goes on from the recording's own speech tokens rather than only borrowing its
voice. That is upstream's own way of cloning a voice in the same language. Without a transcript it
reads cross-lingual, which upstream keeps for a sentence in another language than the recording.
A model that reads no transcript ignores one, so an app offers the box wherever it offers a
recording, and no property says whether a model reads it. What would be worth a property is the
other way round, a model that cannot read without a transcript, and none here is one; it would be
`SUPPORTS_TEXT_TO_SPEECH_WITHOUT_REFERENCE_TRANSCRIPT`, as `WITHOUT_REFERENCE` is for a recording.

A transcript with no reference ends with `WAIFU_ERR_MODEL`. So, on CosyVoice3, does one given with
a style: upstream's instruct2 puts the instruction where zero-shot puts the transcript and drops
the recording's tokens that the transcript is the words of, and never takes both. So does one
holding a `<|...|>` marker, because the system prompt is put in front of it here. CosyVoice3 says
so before it listens to the recording.

**No PSD.** A Photoshop document is a file format, and files are the app's. The layers' pixels,
places and depths are all in the result.

## Stratification

`stratify_layers` runs See-through's depth-guided stratification (section 4.2.2 of the paper) after
the model: hair, hands, top and bottom are each cut, by k-means on their own pixels' depth, into a
near half (`<tag>-0`) and a far half (`<tag>-1`), appended to `layers` after every part.

The halves' order changes with ABI 4. Until now the near half took its part's place and the far
half sank, from there, past every layer under it that was nearer -- an order that only
`waifu/src/see_through/stratify.rs` could work out, and that a caller would have needed as a list
of indices. Now:

- the near half's `depth` is its part's;
- the far half's is the median of its own pixels, as before;
- the stratified stack is the parts that were not split and every half -- a split part is not in
  it, its halves are -- sorted with one comparison: the larger `depth` lower; at one depth, a far
  half lower than a near one; then the smaller index in `layers` lower.

`layers` itself is never reordered: every part, bottom first, then the halves, near before far,
then a depth layer for each, and a half's `derived_from_id` is its part's `id`. A caller builds
the stack as a list of indices sorted by that comparison. The unstratified stack is the parts in
the order they come.

The depth model draws a depth map for each layer, not one for the picture -- upstream's Marigold
takes every layer as a frame and saves each map as `<tag>_depth.png` -- and each part and each half
has its own, the same size as it is. Each comes back as a layer of its own, `WAIFU_LAYER_DEPTH`,
named `<name>_depth`, whose `derived_from_id` is the id of the layer it is the depth of.

On `waifu-output/waifu-0015.png` the two rules differ in one layer -- the near half of the back hair is under the back of the top rather than over it --
and in 255 of the 1280 x 1280 pixels drawn, thin lines along the shoulders.

A near half's `depth` is therefore its part's and not its own: the strands of back hair that fall
over the chest are at 0.427 and say 0.953. That is what keeps it in its part's place, and it is in
the header.

## What has to change with it

- `waifu/src/c_api.rs`, `waifu/src/engine.rs` (an engine bound to one task),
  `waifu/src/describe.rs`, `waifu/src/progress.rs`, the See-through pipeline and `stratify.rs`,
  and their tests; `waifu.h` is regenerated. A transcription's hearing of its recording, which ABI 3
  shares with a voice conversion as `WAIFU_STAGE_LISTENING`, becomes a stage of its own. Streaming
  text to speech, one `Work::Speak` with an `on_audio` today, becomes a task of its own in the
  engine as well, and its load checks the device the way `CosyVoice3::streams` does now.
- `waifu-cli`: pipelines, `WaifuModel`, its own words for each stage.
- The SwiftUI app: the same; a PSD writer of its own; the style picker offers CosyVoice3's examples
  and free text.
- The WPF app: the same, without the PSD.
- `bindings/rust` is written against ABI 4 afterwards, in a pull request of its own: the crate
  `waifu` (package `libwaifu`), designed in [rust-binding.md](rust-binding.md).

## Not in ABI 5

- libwaifu_modelmanager's API -- its catalogue, defaults and directory JSON, and the words in its
  download progress -- has the same faults and its own ABI; it is a design of its own.

## Not decided

- **`get_` on the library's own getters.** The pipeline's are `waifu_pipeline_get_type` and
  `_get_device`; the library's are still `waifu_abi_version`, `waifu_version`, `waifu_revision`
  and `waifu_last_error` / `_code`, with no `get_`.
