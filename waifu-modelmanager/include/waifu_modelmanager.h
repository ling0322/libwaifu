/* The MIT License (MIT). Copyright (c) 2026 Xiaoyang Chen. See LICENSE. */

#ifndef LIBWAIFU_WAIFU_MODELMANAGER_H
#define LIBWAIFU_WAIFU_MODELMANAGER_H

/* Written by cbindgen from waifu-modelmanager/src/c_api.rs. Change that, not this. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// What a caller compiled against this header expects. Bumped when anything in it changes in a
// way an older caller would misread.
#define WAIFU_MODELMANAGER_ABI_VERSION 1

typedef enum {
  WAIFU_MODELMANAGER_LOG_DEBUG = 0,
  WAIFU_MODELMANAGER_LOG_INFO = 1,
  WAIFU_MODELMANAGER_LOG_WARNING = 2,
  WAIFU_MODELMANAGER_LOG_ERROR = 3,
  // The last line: the process aborts when the callback returns.
  WAIFU_MODELMANAGER_LOG_FATAL = 4,
} WaifuModelManagerLogLevel;

// What a call or a job came to.
typedef enum {
  WAIFU_MODELMANAGER_OK = 0,
  WAIFU_MODELMANAGER_CANCELLED = 1,
  // A NULL, or a string that is not UTF-8.
  WAIFU_MODELMANAGER_ERR_INVALID_ARGUMENT = 2,
  // No published model by that name, or no manifest at that path.
  WAIFU_MODELMANAGER_ERR_UNKNOWN_MODEL = 3,
  // The download failed.
  WAIFU_MODELMANAGER_ERR_FETCH = 4,
  // A panic in the library, or a file that could not be read or written.
  WAIFU_MODELMANAGER_ERR_INTERNAL = 5,
} WaifuModelManagerStatusCode;

// A download running on a thread of its own.
typedef struct WaifuModelFetch WaifuModelFetch;

typedef struct {
  WaifuModelManagerLogLevel level;
  // Where it was written: a module's name.
  const char *source;
  // The line alone, with no level or time in front of it.
  const char *message;
} WaifuModelManagerLogEvent;

// Called on whichever thread wrote the line, which can be the caller's own. It must be safe to
// call from any thread, and must not call into the library.
typedef void (*WaifuModelManagerLogCallback)(void *user_data, const WaifuModelManagerLogEvent *event);

typedef struct {
  // 0..1 of the file, or -1 where the server did not say how long it is.
  double fraction;
  // Bytes of the file; total is 0 where it is not known.
  uint64_t done;
  uint64_t total;
  // Which file of how many; parts is 0 until the first has named the rest.
  uint32_t part;
  uint32_t parts;
  // The file, or NULL before the first.
  const char *file;
  // "fetching sdxl:base from Hugging Face -- ... (1 of 3)", for a status line.
  const char *words;
  // Since the job started.
  double seconds;
  // Which hub the files come from, once that is settled; NULL before.
  const char *hub;
} WaifuModelManagerProgressEvent;

typedef void (*WaifuModelManagerProgressCallback)(void *user_data,
                                                  const WaifuModelManagerProgressEvent *event);

// How a job ended, as on_complete is told it.
typedef struct {
  WaifuModelManagerStatusCode code;
  // Why, when code is not WAIFU_MODELMANAGER_OK; NULL when it is.
  const char *message;
} WaifuModelManagerStatus;

// How a fetch ended.
typedef struct {
  WaifuModelManagerStatus status;
} WaifuModelManagerCompletionEvent;

typedef void (*WaifuModelManagerCompleteCallback)(void *user_data,
                                                  const WaifuModelManagerCompletionEvent *event);

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// The ABI this library was built with: WAIFU_MODELMANAGER_ABI_VERSION.
uint32_t waifu_modelmanager_abi_version(void);

// This thread's last message from this library; valid until this thread's next call into it.
const char *waifu_modelmanager_last_error(void);

// Frees a string a call into this library returned. NULL is fine.
void waifu_modelmanager_string_free(char *s);

// Every line this library would print goes to the callback instead, from whichever thread wrote
// it. NULL puts them back on the console. libwaifu's lines are waifu_set_log_callback's.
void waifu_modelmanager_set_log_callback(void *user_data, WaifuModelManagerLogCallback callback);

// Lines below the level are not made at all. INFO unless set.
void waifu_modelmanager_set_log_level(WaifuModelManagerLogLevel level);

// The published models, aliases only, as JSON. Free with waifu_modelmanager_string_free. NULL on
// failure.
char *waifu_modelmanager_catalog_json(void);

// Where downloads go. Free with waifu_modelmanager_string_free. NULL on failure.
char *waifu_modelmanager_directory(void);

// Where downloads go, who said so and what else could, as JSON:
// `{"directory", "said": "environment"|"config"|"default", "default", "config_file"}`, the last
// two null where they cannot be told. Free with waifu_modelmanager_string_free. NULL on failure.
char *waifu_modelmanager_directory_json(void);

// The manifest of a published model, as a path for libwaifu's load and describe: the one in the
// download directory where it has been fetched, and otherwise one of the bundled manifests. Only
// a model whose catalogue entry says `cached` has its weights beside it. A manifest's own path is
// handed back as it is. Free with waifu_modelmanager_string_free. NULL, with
// waifu_modelmanager_last_error, where there is no manifest here: fetch the model, or every
// manifest, first.
char *waifu_modelmanager_manifest_path(const char *name);

// What a screen should start a model's boxes at, as JSON: the kind of model's defaults with what
// its manifest's `suggested:` block says laid over them. `model` is the path of a manifest -- see
// waifu_modelmanager_manifest_path -- or "tones". A picture model's are `width`, `height`, `steps`,
// `guidance`, `takes_guidance`, `sizes`, `prompt` and `avoid`; a voice's are `speed` and
// `temperature`, and `conversion.steps` for one that also converts. None of it is needed to run a
// model, whose request says every number itself. Free with waifu_modelmanager_string_free. NULL,
// with waifu_modelmanager_last_error, for anything that is not a manifest on the disk.
char *waifu_modelmanager_defaults_json(const char *model);

// Downloads go to `path` from now on, saved to config.toml; NULL or "" for the default.
WaifuModelManagerStatusCode waifu_modelmanager_set_directory(const char *path);

// Reads the manifests of published models out of `directory` where the download directory has
// none: an app built with every model's manifest (waifu-tools' `fetch_manifests` fetches them)
// can describe a model before anything is downloaded: waifu_modelmanager_manifest_path hands
// back the bundled one where nothing has been fetched. Laid out as the download directory is,
// and only read. NULL or "" for none. Not saved.
WaifuModelManagerStatusCode waifu_modelmanager_set_bundled_manifests(const char *directory);

// Deletes a downloaded package.
WaifuModelManagerStatusCode waifu_modelmanager_remove(const char *name);

// Starts fetching a published model on a thread of its own and returns at once. Fetching one
// already here finishes at once. Where it went is waifu_modelmanager_manifest_path's to say.
// NULL, with waifu_modelmanager_last_error and no callback ever, where it cannot start.
WaifuModelFetch *waifu_modelmanager_fetch_async(const char *model,
                                                void *user_data,
                                                WaifuModelManagerProgressCallback on_progress,
                                                WaifuModelManagerCompleteCallback on_complete);

// Starts fetching the manifest of every published model that is not here yet -- a couple of
// kilobytes each, and none of their packages -- on a thread of its own, and returns at once.
// What a model suggests is in its manifest, so a screen that has called this can fill in a
// model's settings the moment it is chosen. Each manifest is a part of the whole in the progress;
// with every one already here it finishes at once. Cancelled and freed as a model's fetch is.
// NULL, with waifu_modelmanager_last_error and no callback ever, where it cannot start.
WaifuModelFetch *waifu_modelmanager_fetch_manifests_async(void *user_data,
                                                          WaifuModelManagerProgressCallback on_progress,
                                                          WaifuModelManagerCompleteCallback on_complete);

void waifu_modelmanager_fetch_cancel(WaifuModelFetch *fetch);

// After on_complete; cancels first if it has not had it, and waits for the thread to end.
void waifu_modelmanager_fetch_free(WaifuModelFetch *fetch);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* LIBWAIFU_WAIFU_MODELMANAGER_H */
