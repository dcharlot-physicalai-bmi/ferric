#!/usr/bin/env python3
"""OTLP trace export, checked against the OpenTelemetry protocol definitions themselves.

Runs a collector in-process that decodes every export with `opentelemetry-proto` (the generated classes
of the OTLP .proto files): http/protobuf with ExportTraceServiceRequest.ParseFromString, http/json with
google.protobuf.json_format.Parse (strict: an unknown field fails) after the one OTLP/JSON exception —
trace and span ids are hex, not base64. Then drives ferric-serve with real requests and checks each span
against the response the client received: ids (a traceparent is continued), token usage, finish reasons,
joules, the first-token event inside the span, the status code.

usage: otlp_conformance.py <ferric-serve binary> <model.gguf> http/json|http/protobuf
needs: pip install opentelemetry-proto
"""
import base64, http.server, json, os, subprocess, sys, threading, time, urllib.request

from google.protobuf import json_format
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

BIN, MODEL, PROTO = sys.argv[1], sys.argv[2], sys.argv[3]
SERVE_PORT, OTLP_PORT = 18461, 18462
spans, posts, errors = [], [], []


def hex_to_b64(o):
    """OTLP/JSON ids are hex; the protobuf JSON mapping expects base64 for bytes."""
    for rs in o.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                for k in ("traceId", "spanId", "parentSpanId"):
                    if k in s:
                        s[k] = base64.b64encode(bytes.fromhex(s[k])).decode()
    return o


class Collector(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        ctype = self.headers.get("Content-Type", "")
        req = ExportTraceServiceRequest()
        try:
            if ctype == "application/x-protobuf":
                req.ParseFromString(body)
            elif ctype == "application/json":
                json_format.Parse(json.dumps(hex_to_b64(json.loads(body))), req)  # strict: unknown fields raise
            else:
                raise ValueError(f"content-type {ctype!r}")
            posts.append((self.path, ctype, dict(self.headers)))
            for rs in req.resource_spans:
                res = {kv.key: kv.value.string_value for kv in rs.resource.attributes}
                for ss in rs.scope_spans:
                    for s in ss.spans:
                        spans.append((res, ss.scope.name, s))
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except Exception as e:
            errors.append(f"{ctype}: {e}")
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()


def attrs(s):
    out = {}
    for kv in s.attributes:
        v = kv.value
        w = v.WhichOneof("value")
        out[kv.key] = [x.string_value for x in v.array_value.values] if w == "array_value" else getattr(v, w)
    return out


def call(path, body, headers=None, raw=False):
    r = urllib.request.Request(f"http://127.0.0.1:{SERVE_PORT}{path}", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(r, timeout=600) as f:
            d = f.read()
            return f.status, (d.decode() if raw else json.loads(d))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


srv = http.server.ThreadingHTTPServer(("127.0.0.1", OTLP_PORT), Collector)
threading.Thread(target=srv.serve_forever, daemon=True).start()
env = dict(os.environ, OTEL_EXPORTER_OTLP_ENDPOINT=f"http://127.0.0.1:{OTLP_PORT}", OTEL_EXPORTER_OTLP_PROTOCOL=PROTO,
           OTEL_SERVICE_NAME="ferric-conformance", OTEL_EXPORTER_OTLP_HEADERS="x-otlp-check=yes")
proc = subprocess.Popen([BIN, MODEL, "--port", str(SERVE_PORT)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{SERVE_PORT}/health", timeout=1); break
        except Exception:
            time.sleep(0.5)
    time.sleep(3)  # the energy meter's idle baseline (8 quiet samples) before the first request
    msgs = [{"role": "user", "content": "Name three rivers."}]
    tp_trace, tp_parent = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    # A: non-streamed chat continuing the caller's trace.
    # Long enough (~1 s) for the 100 ms energy meter to resolve, so the joules in the span are checked too.
    _, a = call("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a long essay about the rivers of Europe."}],
                                         "max_tokens": 256, "temperature": 0.7, "seed": 3},
                {"traceparent": f"00-{tp_trace}-{tp_parent}-01"})
    # B: streamed chat with usage.
    _, b_raw = call("/v1/chat/completions", {"messages": msgs, "max_tokens": 10, "stream": True,
                                             "stream_options": {"include_usage": True}}, raw=True)
    b_usage = [json.loads(l[6:]) for l in b_raw.splitlines() if l.startswith("data: {")]
    b_usage = [c["usage"] for c in b_usage if c.get("usage")][0]
    # C: two completions at once (the batched path).
    res = {}
    th = [threading.Thread(target=lambda i=i: res.__setitem__(i, call("/v1/completions", {"prompt": f"The history of the number {i + 5}:", "max_tokens": 192 + i})[1])) for i in range(2)]
    [t.start() for t in th]; [t.join() for t in th]
    # D: n = 2 on the serial path; E: a refusal.
    _, d = call("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a long story about a lighthouse."}],
                                         "max_tokens": 160, "n": 2, "temperature": 1.0})
    e_code, _ = call("/v1/chat/completions", {"messages": msgs, "n": 99})
    time.sleep(2.5)  # the exporter batches for up to 1 s
finally:
    proc.terminate(); proc.wait(timeout=30)
srv.shutdown()

fail = []
def check(cond, what):
    (print("  ok  ", what) if cond else (print("  FAIL", what), fail.append(what)))

print(f"{PROTO}: {len(posts)} export(s), {len(spans)} span(s), decode errors {errors}")
check(not errors and posts, "every export decodes with the OTLP definitions")
check(all(p == "/v1/traces" and h.get("x-otlp-check") == "yes" for p, _, h in posts), "POST /v1/traces with OTEL_EXPORTER_OTLP_HEADERS")
check(all(r.get("service.name") == "ferric-conformance" and sc == "ferric-serve" for r, sc, _ in spans), "resource service.name and scope")
gen = [s for _, _, s in spans if "gen_ai.operation.name" in attrs(s)]
by = lambda pred: [s for s in gen if pred(s, attrs(s))]

def usage_ok(s, usage, finishes=None, energy=None):
    at = attrs(s)
    ok = at.get("gen_ai.usage.input_tokens") == usage["prompt_tokens"] and at.get("gen_ai.usage.output_tokens") == usage["completion_tokens"]
    if finishes is not None: ok &= at.get("gen_ai.response.finish_reasons") == finishes
    if energy and energy.get("joules") is not None: ok &= abs(at.get("ferric.energy.joules", -1) - energy["joules"]) < 1e-9
    ev = [e for e in s.events if e.name == "gen_ai.first_token"]
    ok &= s.start_time_unix_nano < s.end_time_unix_nano and s.kind == 2
    ok &= len(ev) == 1 and s.start_time_unix_nano <= ev[0].time_unix_nano <= s.end_time_unix_nano
    return ok

sa = by(lambda s, at: s.trace_id.hex() == tp_trace)
check(len(sa) == 1 and sa[0].parent_span_id.hex() == tp_parent, "A: traceparent continued (trace id, parent span id)")
check(len(sa) == 1 and usage_ok(sa[0], a["usage"], [a["choices"][0]["finish_reason"]], a.get("energy")), "A: usage, finish reason, joules, first-token event")
check(len(sa) == 1 and attrs(sa[0]).get("gen_ai.request.temperature") == 0.7 and attrs(sa[0]).get("gen_ai.request.seed") == 3
      and attrs(sa[0]).get("gen_ai.request.max_tokens") == 256 and sa[0].name.startswith("chat"), "A: request attributes, span name")
check(len(sa) == 1 and attrs(sa[0]).get("http.response.status_code") == 200 and sa[0].status.code == 1, "A: status 200 / OK")
sb = by(lambda s, at: at.get("ferric.request.stream") is True)
check(len(sb) == 1 and usage_ok(sb[0], b_usage), "B: streamed chat usage and first-token event")
for i in range(2):
    sc = by(lambda s, at: at.get("gen_ai.operation.name") == "text_completion" and at.get("gen_ai.request.max_tokens") == 192 + i)
    check(len(sc) == 1 and usage_ok(sc[0], res[i]["usage"], [res[i]["choices"][0]["finish_reason"]], res[i].get("energy")),
          f"C{i}: batched completion usage, finish, joules")
sd = by(lambda s, at: at.get("gen_ai.request.choice.count") == 2)
check(len(sd) == 1 and usage_ok(sd[0], d["usage"], [c["finish_reason"] for c in d["choices"]], d.get("energy")),
      "D: n = 2 — output and joules summed over both choices, two finish reasons")
se = by(lambda s, at: at.get("gen_ai.request.choice.count") == 99)
check(e_code == 400 and len(se) == 1 and attrs(se[0]).get("http.response.status_code") == 400 and se[0].status.code == 1
      and "gen_ai.usage.output_tokens" not in attrs(se[0]), "E: a refusal reports 400, not an error span, no usage")
metered = [x for x in [a, res[0], res[1], d] if (x.get("energy") or {}).get("joules") is not None]
why = {(x.get("energy") or {}).get("why") for x in [a, res[0], res[1], d]} - {None}
print(f"  ({len(metered)} of 4 responses carried joules{'' if metered else ' — so the joules check is vacuous in this run'}{f'; the meter said: {sorted(why)}' if why else ''})")
check(len({s.span_id for s in gen}) == len(gen) and all(len(s.trace_id) == 16 and len(s.span_id) == 8 for s in gen), "ids: unique, 16 + 8 bytes")
print("ALL OK" if not fail else f"{len(fail)} FAILED")
sys.exit(1 if fail else 0)
