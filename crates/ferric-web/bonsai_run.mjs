import puppeteer from 'puppeteer-core';
const b = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new', userDataDir: '/tmp/ferric-bonsai-prof',
  args: ['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=metal']
});
const p = await b.newPage();
const errs = [];
p.on('pageerror', e => errs.push(e.message.slice(0,240)));
p.on('console', m => { const t=m.text(); if(t.includes('panic')||t.includes('Error')) errs.push('console: '+t.slice(0,200)); });
await p.goto('http://localhost:8799/bonsai_test.html', { waitUntil:'load', timeout:30000 });
try {
  const out = await p.evaluate(() => window.__run(), );
  console.log('BONSAI_RESULT: ' + out);
} catch(e) { console.log('EVAL_ERROR: ' + e.message.slice(0,300)); }
console.log('pageerrors:', errs.length ? errs.slice(0,3) : 'none');
await b.close();
