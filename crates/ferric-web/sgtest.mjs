import puppeteer from 'puppeteer-core';
const b = await puppeteer.launch({ executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless:'new', userDataDir:'/tmp/ferric-sg', args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=metal'] });
const p = await b.newPage();
await p.goto('http://localhost:8799/bonsai.html?model=/m.gguf',{waitUntil:'load',timeout:20000});
const out = await p.evaluate(async () => {
  const mod = await import('/pkg/ferric_web.js'); await mod.default();
  const buf = new Uint8Array(await (await fetch('/m.gguf')).arrayBuffer());
  return await mod.bonsai_logits(buf, 'The capital of France is');
});
const s = JSON.parse(out);
console.log('browser sum:', s.sum, 'argmax:', s.argmax);
await b.close();
