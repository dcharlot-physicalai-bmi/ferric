// Does speech recognition actually run in a real browser? Headed Chrome, real WebGPU, real model.
// A wasm build that COMPILES proves nothing about whether the kernels dispatch or the transcript is
// right — only loading the checkpoint in a tab and reading the words back does.
//
// ⚠ THIS IS THE WASM *RUN* GATE, AND A BUILD GATE CANNOT REPLACE IT. `std::env::var` returns `Err`
// on wasm32 (silently pinning feature flags) and `Instant::now()` PANICS — both compile clean and
// both killed browser speech during this port while every native test stayed green. CI's wasm step
// builds; only this loads the model in a tab and reads the words back.
//
// Chrome comes from $CHROME so the script is not machine-specific — that hardcoded path is why
// `crates/ferric-web/*_test.mjs` is gitignored, and this file is exempted from that rule.
import puppeteer from 'puppeteer-core';
const CHROME = process.env.CHROME
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const b = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new', userDataDir: '/tmp/ferric-speech', protocolTimeout: 1800000,
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal',
         '--autoplay-policy=no-user-gesture-required'] });
const p = await b.newPage();
const errs = []; p.on('pageerror', e => errs.push(e.message.slice(0, 300)));
p.on('console', m => { const t = m.text(); if (/error|panic|unreachable/i.test(t)) errs.push('console: ' + t.slice(0,300)); });
await p.goto('http://localhost:8770/speech.html', { waitUntil: 'domcontentloaded', timeout: 60000 });
let out;
try { out = await p.evaluate(() => window.runSpeech()); }
catch (e) { out = 'THREW: ' + String(e && e.message || e).slice(0, 400); }
const REF = 'mister quilter is the apostle of the middle classes and we are glad to welcome his gospel';
console.log('TRANSCRIPT: ' + JSON.stringify(out));
console.log('MATCHES_REFERENCE: ' + (String(out).trim().toLowerCase() === REF));
console.log('STATUS: ' + await p.$eval('#status', e => e.textContent).catch(() => 'n/a'));
console.log('pageerrors:', errs.length ? errs.slice(0, 3) : 'none');
await b.close();
