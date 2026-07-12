import puppeteer from 'puppeteer-core';
const b=await puppeteer.launch({ executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless:false, userDataDir:'/tmp/ferric-prof', args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU'] });
const p=await b.newPage(); const errs=[];
p.on('pageerror',e=>errs.push(e.message.slice(0,160)));
const done=new Promise(res=>{ p.on('console',m=>{const t=m.text(); if(t.startsWith('FERRIC> ')){ console.log(t.slice(8)); if(/RESULT_OK|RESULT_BAD|ERROR/.test(t)) setTimeout(res,800);} }); });
await p.goto('http://localhost:8900/index.html',{waitUntil:'domcontentloaded',timeout:30000});
await Promise.race([done, new Promise(r=>setTimeout(r,60000))]);
console.log('pageerrors:', errs.length?errs.slice(0,3):'none');
await b.close();
