#!/usr/bin/env python3
"""The bundled chat page, driven in a real (headless) Chrome over the DevTools protocol.

Checks what a person sees against what the API says: the page is served at / to a browser (text to other
clients), the model list fills, a typed question streams an answer that equals the API's own answer to the
same messages (greedy), the footer's token count equals the API's usage, the energy chip shows joules or the
meter's reason, model output is shown as text (markup escaped), and with --api-key the page loads, a request
without the key surfaces the 401, and one with it answers.

usage: ui_check.py <ferric-serve binary> <model.gguf>      needs: pip install websocket-client
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.request
import websocket

BIN, MODEL = sys.argv[1], sys.argv[2]
PORT, DBG = 18471, 18472
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
fail = []
def check(c, what):
    print(("  ok   " if c else "  FAIL ") + what)
    if not c: fail.append(what)

def wait_http(url, t=120):
    for _ in range(t * 2):
        try: urllib.request.urlopen(url, timeout=1); return
        except Exception: time.sleep(0.5)
    raise SystemExit(f"no answer from {url}")

class Page:
    def __init__(self):
        tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DBG}/json").read())
        tab = [t for t in tabs if t["type"] == "page"][0]
        self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=60, suppress_origin=True)
        self.n = 0
    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n: return m.get("result", {})
    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r: raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")
    def until(self, expr, t=240):
        for _ in range(t * 4):
            v = self.js(expr)
            if v: return v
            time.sleep(0.25)
        raise RuntimeError(f"timed out waiting for {expr}")
    def goto(self, url):
        self.call("Page.navigate", url=url)
        self.until("document.readyState === 'complete' && !!document.getElementById('log')")

def ask(p, text):
    p.js(f"document.getElementById('input').value = {json.dumps(text)}; document.getElementById('form').requestSubmit(); true")
    p.until("(() => { const m = document.querySelectorAll('#log .msg'); const l = m[m.length - 1]; "
            "return !!l && (l.classList.contains('error') || !!l.querySelector('.meta')) && document.getElementById('send').textContent === 'Send'; })()")
    return p.js("(() => { const m = document.querySelectorAll('#log .msg'); const l = m[m.length - 1]; return {"
                "cls: l.className, text: l.querySelector('.body') ? l.querySelector('.body').innerText : l.innerText,"
                "chips: [...l.querySelectorAll('.chip')].map(c => [c.className, c.textContent, c.title])}; })()")

def run(api_key):
    args = [BIN, MODEL, "--port", str(PORT)] + (["--api-key", api_key] if api_key else [])
    srv = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    prof = tempfile.mkdtemp()
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={DBG}", f"--user-data-dir={prof}",
                               "--no-first-run", "--no-default-browser-check", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_http(f"http://127.0.0.1:{PORT}/health")
        wait_http(f"http://127.0.0.1:{DBG}/json/version")
        time.sleep(3)  # the energy meter's idle baseline
        p = Page()
        p.call("Page.enable"); p.call("Runtime.enable")
        p.goto(f"http://127.0.0.1:{PORT}/")
        return p, srv, chrome, prof
    except Exception:
        srv.terminate(); chrome.terminate(); raise

def stop(srv, chrome, prof):
    chrome.terminate(); srv.terminate()
    chrome.wait(timeout=30); srv.wait(timeout=30)
    shutil.rmtree(prof, ignore_errors=True)

# --- no API key ---
p, srv, chrome, prof = run(None)
try:
    plain = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/").read().decode()
    check(plain.startswith("ferric-serve is running"), "GET / without Accept: text/html is still the one-line text")
    check(p.js("document.title") == "Ferric chat", "GET / in a browser serves the chat page")
    models = p.until("(() => { const o = [...document.querySelectorAll('#model option')].map(o => o.value); return o.length && o[0] ? o : null; })()")
    check(len(models) >= 1, f"model list filled: {models}")
    p.js("document.getElementById('temp').value = '0'; document.getElementById('max').value = '48'; true")
    q = "Name the largest planet in the solar system, in one sentence."
    got = ask(p, q)
    api = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({"model": models[0], "messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 48}).encode(),
        headers={"Content-Type": "application/json"})).read())
    want = api["choices"][0]["message"]["content"]
    check(got["text"].strip() == want.strip(), f"streamed answer in the page == the API's answer: {got['text'][:60]!r}")
    chips = {c[1].split(" ")[1] if " " in c[1] else c[1]: c for c in got["chips"]}
    tok = [c for c in got["chips"] if c[1].endswith(" tokens")]
    check(bool(tok) and int(tok[0][1].split()[0]) == api["usage"]["completion_tokens"], f"footer token count == usage.completion_tokens ({api['usage']['completion_tokens']})")
    en = [c for c in got["chips"] if "energy" in c[0]]
    check(len(en) == 1 and (en[0][1].endswith("mJ/token") or (en[0][1] == "energy not attributed" and en[0][2])),
          f"energy chip: {en[0][1] if en else None!r} {('(' + en[0][2][:60] + ')') if en and en[0][2] else ''}")
    esc = p.js("markdown('<img src=x onerror=alert(1)> **b** `c`')")
    check("<img" not in esc and "&lt;img" in esc and "<strong>b</strong>" in esc and "<code>c</code>" in esc, "model output is escaped; bold and code render")
    check(p.js("document.querySelectorAll('#log .msg.user').length") == 1, "the question appears once")
finally:
    stop(srv, chrome, prof)

# --- with an API key ---
p, srv, chrome, prof = run("sesame")
try:
    check(p.js("document.title") == "Ferric chat", "--api-key: the page itself loads without the key")
    p.js("document.getElementById('max').value = '8'; true")
    got = ask(p, "Say hi.")
    check("error" in got["cls"] and "API key" in got["text"], f"--api-key: a request without the key shows the 401: {got['text'][:50]!r}")
    p.js("document.getElementById('key').value = 'sesame'; document.getElementById('key').dispatchEvent(new Event('change')); true")
    got = ask(p, "Say hi.")
    check("error" not in got["cls"] and got["text"].strip() != "", f"--api-key: with the key it answers: {got['text'][:40]!r}")
finally:
    stop(srv, chrome, prof)
print("ALL OK" if not fail else f"{len(fail)} FAILED")
sys.exit(1 if fail else 0)
