// Install CPython 3.12.3 onto PATH.
// ubuntu-latest is 26.04. actions/setup-python has no 3.12.3 build for that
// image, and that image's PATH has neither python3, curl, nor sudo (exit 127).
// Node is the runner runtime. The standalone build bundles libpython and links
// only glibc, so it runs on 26.04.
const crypto = require("crypto");
const fs = require("fs");
const https = require("https");
const path = require("path");
const zlib = require("zlib");

// The actions/python-versions 24.04 tarball links the system libpython3.12,
// which Ubuntu 26.04 does not ship. This is the standalone 3.12.3 build:
// libpython sits next to the binary, and only glibc is external.
const url = "https://github.com/astral-sh/python-build-standalone/releases/download/20240415/cpython-3.12.3%2B20240415-x86_64-unknown-linux-gnu-install_only.tar.gz";
const expected = "a73ba777b5d55ca89edef709e6b8521e3f3d4289581f174c8699adfb608d09d6";
const tarball = "/tmp/python-3.12.3.tar.gz";

function get(target, dest) {
  return new Promise((resolve, reject) => {
    https.get(target, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        resolve(get(res.headers.location, dest));
        return;
      }
      if (res.statusCode !== 200) {
        reject(new Error("HTTP " + res.statusCode));
        return;
      }
      const file = fs.createWriteStream(dest);
      res.pipe(file);
      file.on("finish", () => file.close(resolve));
    }).on("error", reject);
  });
}

function cstr(buf) {
  const z = buf.indexOf(0);
  return buf.subarray(0, z < 0 ? buf.length : z).toString("utf8");
}

function extract(gz, dest) {
  const buf = zlib.gunzipSync(fs.readFileSync(gz));
  let off = 0;
  while (off + 512 <= buf.length) {
    const header = buf.subarray(off, off + 512);
    if (header.every((b) => b === 0)) break;
    let name = cstr(header.subarray(0, 100));
    const prefix = cstr(header.subarray(345, 500));
    if (prefix) name = prefix + "/" + name;
    const size = parseInt(cstr(header.subarray(124, 136)).trim() || "0", 8);
    const type = String.fromCharCode(header[156] || 48);
    const link = cstr(header.subarray(157, 257));
    const mode = parseInt(cstr(header.subarray(100, 108)).trim() || "644", 8);
    off += 512;
    const data = buf.subarray(off, off + size);
    off += Math.ceil(size / 512) * 512;
    const rel = name.replace(/^\.\//, "").replace(/^python\//, "");
    if (!rel || rel.includes("..")) continue;
    const target = path.join(dest, rel);
    if (type === "5") {
      fs.mkdirSync(target, { recursive: true });
    } else if (type === "2") {
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.symlinkSync(link, target);
    } else if (type === "0" || type === "\0") {
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, data);
      fs.chmodSync(target, mode);
    }
  }
}

(async () => {
  await get(url, tarball);
  const digest = crypto.createHash("sha256").update(fs.readFileSync(tarball)).digest("hex");
  if (digest !== expected) throw new Error("python tarball sha256 " + digest);
  const dest = path.join(process.env.HOME, "python-3.12.3");
  fs.mkdirSync(dest, { recursive: true });
  extract(tarball, dest);
  fs.appendFileSync(process.env.GITHUB_PATH, path.join(dest, "bin") + "\n");
  console.log("extracted", dest);
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
