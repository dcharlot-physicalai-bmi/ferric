#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct FerricHandle FerricHandle;

/**
 * Load a GGUF model. Returns an opaque handle, or NULL on failure.
 */
struct FerricHandle *ferric_load(const char *model_path);

/**
 * Greedy free-text completion of `prompt` for up to `max_tokens`. Caller frees the result string.
 */
char *ferric_generate(struct FerricHandle *h, const char *prompt, uint32_t max_tokens);

/**
 * Schema-constrained generation: output is guaranteed-conformant JSON. `schema` is a JSON-Schema
 * string (empty → any valid JSON object). Caller frees the result string.
 */
char *ferric_generate_json(struct FerricHandle *h,
                           const char *prompt,
                           const char *schema,
                           uint32_t max_tokens);

/**
 * Free a string returned by `ferric_generate*`.
 */
void ferric_free_string(char *s);

/**
 * Free a model handle.
 */
void ferric_free(struct FerricHandle *h);
