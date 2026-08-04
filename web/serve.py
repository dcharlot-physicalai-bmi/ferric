#!/usr/bin/env python3
"""Static server that honours HTTP Range — which `python3 -m http.server` does NOT.

This exists because the obvious choice silently breaks the demo: SimpleHTTPRequestHandler ignores the
Range header and answers 200 with the WHOLE body. For a 675 MB checkpoint that means every "range
request" downloads the entire file, so streaming appears to work while doing the opposite. Verified
before writing this: a 64-byte range came back as 675,710,816 bytes.
"""
import http.server, os, re, socketserver, sys

class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().do_GET()
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if not m:
            self.send_error(400, "malformed Range")
            return
        size = os.path.getsize(path)
        lo, hi = m.group(1), m.group(2)
        if lo == "":                              # suffix form: bytes=-N
            length = min(int(hi or 0), size)
            start, end = size - length, size - 1
        else:
            start = int(lo)
            end = int(hi) if hi else size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        end = min(end, size - 1)
        n = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(n))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = n
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"http://localhost:{port}   (Range-capable; Ctrl-C to stop)")
    Server(("", port), RangeHandler).serve_forever()
