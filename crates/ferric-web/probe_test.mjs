import puppeteer from 'puppeteer-core';
const b = await puppeteer.launch({
  executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless:'new', userDataDir:'/tmp/ferric-probe-prof',
  args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=metal']
});
const p = await b.newPage(); const errs=[];
p.on('pageerror',e=>errs.push(e.message.slice(0,200)));
p.on('console',m=>{const t=m.text(); if(/panic|error/i.test(t)) errs.push('con:'+t.slice(0,160));});
await p.goto('http://localhost:8799/probe.html',{waitUntil:'load',timeout:30000});
const ok = await p.waitForFunction(()=>document.getElementById('out').textContent!=='running…',{timeout:60000}).then(()=>true).catch(()=>false);
console.log(await p.evaluate(()=>document.getElementById('out').textContent));
if(!ok) console.log('TIMEOUT');
console.log('pageerrors:', errs.length?errs.slice(0,3):'none');
await b.close();
