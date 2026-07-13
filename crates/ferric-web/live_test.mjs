import puppeteer from 'puppeteer-core';
const b=await puppeteer.launch({ executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless:'new', userDataDir:'/tmp/ferric-live-prof', args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=metal'] });
const p=await b.newPage(); const errs=[]; p.on('pageerror',e=>errs.push(e.message.slice(0,180)));
await p.goto('https://ferric.pages.dev/',{waitUntil:'networkidle0',timeout:40000});
const out=await p.evaluate(async()=>{ const m=await import('/pkg/ferric_web.js'); await m.default();
  if(!navigator.gpu) return 'NO_WEBGPU'; return await m.ferric_lm_demo('7,2,19,4', 5); });
console.log('LIVE_RESULT: '+out); console.log('pageerrors:', errs.length?errs.slice(0,2):'none');
await b.close();
