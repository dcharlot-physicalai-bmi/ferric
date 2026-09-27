# `ferric` — the command line

Run, chat with, pull, list, inspect and benchmark local models, and see what each answer cost in
**joules**. The `ferric` CLI works like Ollama's: it is an HTTP client of a local server
(`ferric-serve`, which speaks Ollama's `/api` dialect) and a process manager for it. When nothing
answers, `ferric run` starts a server in the background.

```bash
cargo build --release -p ferric-serve -p ferric-cli      # puts ferric and ferric-serve side by side
export PATH="$PWD/target/release:$PATH"

ferric pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0          # Hugging Face → ~/.cache/ferric/hub
ferric run qwen2.5-0.5b-instruct-q8_0 "What is the capital of France?" --verbose
ferric run qwen2.5-0.5b-instruct-q8_0                     # chat; /? for commands, /bye to leave
ferric bench qwen2.5-0.5b-instruct-q8_0                   # prefill + decode tok/s, joules per token
```

## Commands

| command | what it does |
|---|---|
| `ferric serve [model...] [flags]` | runs `ferric-serve` in the foreground on `FERRIC_HOST`. Leading arguments are models (resolved, and pulled if they are Hugging Face references); everything from the first flag on is passed to ferric-serve unchanged |
| `ferric run <model> [prompt]` | with a prompt (or piped input): streams one answer and exits. Without: a chat that keeps the conversation and resends it every turn |
| `ferric pull <owner/repo[:tag]>` | downloads a GGUF into `~/.cache/ferric/hub/<owner>_<repo>/`. A `:Q4_K_M` tag picks the matching file, `:file.gguf` names one, no tag means Q4_K_M. Split GGUFs (`-00001-of-00003`) are fetched whole. Interrupted downloads stay `.partial` and resume (`curl -C -`) |
| `ferric list` / `ls` | the hub: name, architecture, parameters, quantisation, size, age (read from each GGUF header). QUANT is the tensor type holding the most weights — the same figure ferric-serve's `/api/tags` reports — so Qwen's own `…-q4_k_m.gguf` lists as `Q5_0` (62% of its elements; 8% are Q4_K). PARAMS counts every tensor in the file, so a tied embedding stored twice counts twice |
| `ferric show <model>` | architecture, parameters, context, quantisation, template, licence, and **Ferric's verdict on the architecture** (verified against the authors' implementation / loads / refused). `--template`, `--license` print those in full |
| `ferric ps` | what the server has loaded (`/api/ps`) |
| `ferric stop <model>` | unloads it (`/api/generate` with `keep_alive: 0`, Ollama's documented unload), and reports failure if the server does not confirm |
| `ferric rm <model>... [-y]` | deletes a model's files from the hub, after asking |
| `ferric bench <model>` | prompt-processing and generation tok/s and J/token over `--reps` runs, reported as ranges |

A **model** is a path to a `.gguf`, a name from `ferric list` (`qwen2.5-0.5b-instruct-q8_0`, or
`<dir>/<name>` where a name is in two directories), or a Hugging Face reference `owner/repo[:tag]`
(also `hf.co/owner/repo:Q4_K_M`), pulled on first use.

### `run` flags and chat commands

`--verbose` prints Ollama's timing block plus the answer's joules and J/token; `--hidethinking` hides a
thinking model's reasoning (shown dimmed otherwise); `--think` / `--think=false`; `--format json`;
`--keepalive 5m`; `--option temperature=0.2` (repeatable); `--system "..."`; `-i` chats even when
stdin is a pipe, which scripts a conversation:

```bash
printf 'What is the capital of France? One word.\nAnd of Germany?\n/bye\n' | ferric run qwen2.5-0.5b-instruct-q8_0 -i
```

In the chat: `"""` opens a multi-line message; `/set parameter <name> <value>`, `/set system`,
`/set verbose|quiet`, `/set think|nothink`, `/set format json|noformat`, `/show info|parameters|system|template|license`,
`/load <model>`, `/clear`, `/bye`. Ctrl-C while an answer streams stops that answer (the connection is
closed, which is how the server learns to stop) and keeps the chat.

## Environment

| variable | default | |
|---|---|---|
| `FERRIC_HOST` | `127.0.0.1:11435` | the server. One port above Ollama's 11434 so both can run; `FERRIC_HOST=127.0.0.1:11434` drives a real Ollama |
| `FERRIC_HOME` | `~/.cache/ferric` | `hub/` (models), `serve.log` (the background server's output), `serve.pid` |
| `FERRIC_SERVE` | beside `ferric`, then `PATH` | the ferric-serve binary |
| `FERRIC_HF_ENDPOINT` | `HF_ENDPOINT`, else `https://huggingface.co` | a mirror |
| `HF_TOKEN` | `~/.cache/huggingface/token` | gated repos. Passed to curl on stdin, never in argv |
| `FERRIC_START_TIMEOUT` | `600` | seconds to wait for a started server to answer |

## Energy

ferric-serve attributes joules to every request from a power meter (macmon on Apple Silicon: GPU + DRAM
rails, idle-subtracted, 100 ms samples — see `crates/ferric-serve/src/energy.rs`). The CLI prints what
the server measured and, when it could not measure, why: a request shorter than three meter samples, or
no idle baseline yet, is reported as *unmeasured* with the gross window energy, never as a number. On a
shared machine the idle baseline can move by more than a light request adds; `bench` says so when a run
comes out negative. A server that reports no energy (Ollama) is said to report none.

## Talking to Ollama

Nothing the CLI sends is private to Ferric. Against a server whose `/api/version` is not ferric-serve's,
model names go through as typed (`llama3.2`), nothing is pulled locally, and the ferric-only checks are
skipped. `list`, `pull` and `rm` always act on Ferric's own hub.

## Tests

`cargo test -p ferric-cli` runs the real binary against an in-test server that speaks Ollama's NDJSON
API (both framings: ferric-serve's close-delimited stream and Ollama's chunked one) and records every
request, and against a mock Hugging Face that `curl` downloads from (a quant-tag pick, a three-part
split, a resumed `.partial`, the token header). The process manager is tested with stand-in server
scripts: one that stays up (start, wait, chat), one that dies (its log is shown), one that echoes argv
(`serve`).
