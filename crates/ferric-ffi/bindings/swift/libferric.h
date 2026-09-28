#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* libferric — Ferric's C ABI. The handle is ferric-serve's engine: every registered architecture, the
 * model's own chat template, full sampling, constrained decoding, reasoning split, per-request energy.
 * Requests and responses are OpenAI-shaped JSON. Every returned string is freed with ferric_free_string;
 * an error comes back as {"error": {"message": ...}}, never NULL (ferric_load alone returns NULL). */

typedef struct FerricHandle FerricHandle;

/* Streaming callback: `delta` is valid only during the call; `is_reasoning` is 1 for a thinking
 * model's reasoning, 0 for the answer. */
typedef void (*FerricDeltaCb)(const char *delta, int is_reasoning, void *user);

/** Load a GGUF model. Returns an opaque handle, or NULL on failure (the reason goes to stderr). */
struct FerricHandle *ferric_load(const char *model_path);

/** An OpenAI /v1/chat/completions request (JSON) -> the chat.completion object (JSON). */
char *ferric_chat(struct FerricHandle *h, const char *request_json);

/** ferric_chat, calling `cb` with each piece of the answer as it is generated. */
char *ferric_chat_stream(struct FerricHandle *h, const char *request_json, FerricDeltaCb cb, void *user);

/** An OpenAI /v1/completions request (JSON, a string "prompt") -> the text_completion object (JSON). */
char *ferric_complete(struct FerricHandle *h, const char *request_json);

/** Greedy free-text completion of `prompt` for up to `max_tokens`. */
char *ferric_generate(struct FerricHandle *h, const char *prompt, uint32_t max_tokens);

/** Schema-constrained generation: output is guaranteed-conformant JSON. `schema` is a JSON-Schema
 * string (empty -> any valid JSON object). */
char *ferric_generate_json(struct FerricHandle *h,
                           const char *prompt,
                           const char *schema,
                           uint32_t max_tokens);

/** Free a string returned by any ferric_* call. */
void ferric_free_string(char *s);

/** Free a model handle. */
void ferric_free(struct FerricHandle *h);
