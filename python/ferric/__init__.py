"""Ferric — a pure-Rust cross-fabric AI runtime, adoptable from Python.

    from ferric import Ferric
    llm = Ferric("Qwen/Qwen2.5-0.5B-Instruct-GGUF")   # HF repo or local .gguf
    print(llm.generate("The capital of France is"))
    person = llm.generate_object("Give a person.",     # schema-constrained (guaranteed valid)
        {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}})

Day-one packaging: this drives the `ferric-serve` binary over its OpenAI-compatible API (no
build step, works with any Python). The in-process pyo3 binding is the follow-up. Deterministic
by default (greedy / fixed-seed sampling); schema-constrained structured output via guided decoding.
"""
import atexit, json, os, socket, subprocess, time, urllib.request

__all__ = ["Ferric", "load"]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Ferric:
    def __init__(self, model, *, bin=None, port=None, name="ferric", mcp=None, mcp_http=None, timeout=300):
        self.bin = bin or os.environ.get("FERRIC_SERVE_BIN", "ferric-serve")
        self.port = port or _free_port()
        args = [self.bin, model, "--port", str(self.port), "--name", name]
        for m in mcp or []:
            args += ["--mcp", m]
        for u in mcp_http or []:
            args += ["--mcp-http", u]
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)
        self.base = f"http://127.0.0.1:{self.port}"
        self._wait(timeout)

    def _wait(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("ferric-serve exited during startup (bad model/binary?)")
            try:
                urllib.request.urlopen(self.base + "/health", timeout=2)
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("ferric-serve did not become ready in time")

    def _post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        return json.load(urllib.request.urlopen(req))

    def chat(self, messages, *, max_tokens=256, temperature=0.0, tools=None, **kw):
        """Return the assistant message dict (content, or tool_calls when tools are used)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, **kw}
        if tools:
            body["tools"] = tools
        return self._post("/v1/chat/completions", body)["choices"][0]["message"]

    def generate(self, messages, **kw):
        """Return just the assistant text."""
        return self.chat(messages, **kw).get("content", "")

    def generate_object(self, messages, schema, **kw):
        """Schema-constrained structured output via guided decoding — guaranteed-conformant JSON."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"messages": messages, "max_tokens": kw.pop("max_tokens", 256),
                "response_format": {"type": "json_schema", "json_schema": {"name": "schema", "schema": schema}}, **kw}
        return json.loads(self._post("/v1/chat/completions", body)["choices"][0]["message"]["content"])

    def close(self):
        p = getattr(self, "proc", None)
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def load(model, **kw):
    return Ferric(model, **kw)
