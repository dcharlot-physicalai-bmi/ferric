"""Ferric from Python — a ctypes binding over libferric (no build step, no dependencies).

    from ferric import Model
    m = Model("qwen2.5-0.5b-instruct-q8_0.gguf")
    r = m.chat([{"role": "user", "content": "Hi"}], max_tokens=32)
    print(r["choices"][0]["message"]["content"], r["energy"])
    m.chat(messages, stream=print)            # called with each piece as it is generated
    m.complete("The capital of France is", max_tokens=8)

Requests and responses are the OpenAI shapes (the same engine as ferric-serve), as dicts. The library is
found at $FERRIC_LIB, else target/release/libferric.{dylib,so,dll} beside this checkout.
"""
import ctypes, json, os, sys
from pathlib import Path

_CB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)


def _lib():
    ext = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
    here = Path(__file__).resolve()
    cands = [os.environ.get("FERRIC_LIB")] + [str(p / "target" / "release" / f"libferric.{ext}") for p in here.parents]
    for c in cands:
        if c and Path(c).exists():
            lib = ctypes.CDLL(c)
            break
    else:
        raise OSError("libferric not found: build it with `cargo build --release -p ferric-ffi` or set FERRIC_LIB")
    lib.ferric_load.restype = ctypes.c_void_p
    lib.ferric_load.argtypes = [ctypes.c_char_p]
    for f in ("ferric_chat", "ferric_complete"):
        getattr(lib, f).restype = ctypes.c_void_p
        getattr(lib, f).argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ferric_chat_stream.restype = ctypes.c_void_p
    lib.ferric_chat_stream.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _CB, ctypes.c_void_p]
    lib.ferric_free_string.argtypes = [ctypes.c_void_p]
    lib.ferric_free.argtypes = [ctypes.c_void_p]
    return lib


_L = None


class FerricError(RuntimeError):
    pass


class Model:
    def __init__(self, path):
        global _L
        _L = _L or _lib()
        self._h = _L.ferric_load(str(path).encode())
        if not self._h:
            raise FerricError(f"could not load {path} (the reason is on stderr)")

    def _take(self, p):
        try:
            v = json.loads(ctypes.string_at(p).decode("utf-8"))
        finally:
            _L.ferric_free_string(p)
        if "error" in v:
            raise FerricError(v["error"].get("message", v["error"]))
        return v

    def chat(self, messages, stream=None, **params):
        """messages + any /v1/chat/completions field -> the chat.completion dict. `stream(text, is_reasoning)`
        (or `stream(text)`) is called with each piece as it is generated."""
        req = json.dumps({"messages": messages, **params}).encode()
        if stream is None:
            return self._take(_L.ferric_chat(self._h, req))
        n = stream.__code__.co_argcount - (1 if hasattr(stream, "__self__") else 0) if hasattr(stream, "__code__") else 1
        cb = _CB(lambda d, r, _u: stream(d.decode("utf-8"), bool(r)) if n >= 2 else stream(d.decode("utf-8")))
        return self._take(_L.ferric_chat_stream(self._h, req, cb, None))

    def complete(self, prompt, **params):
        """A /v1/completions request -> the text_completion dict."""
        return self._take(_L.ferric_complete(self._h, json.dumps({"prompt": prompt, **params}).encode()))

    def close(self):
        if self._h:
            _L.ferric_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
