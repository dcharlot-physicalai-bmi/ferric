import http from 'http'; import { readFile } from 'fs/promises'; import puppeteer from 'puppeteer-core';
const MIME={'.html':'text/html','.js':'text/javascript','.wasm':'application/wasm','.ts':'text/plain'};
const srv=http.createServer(async(req,res)=>{
  try{ let p=req.url.split('?')[0]; if(p==='/')p='/index.html';
    const buf=await readFile('site'+p); const ext=p.slice(p.lastIndexOf('.'));
    res.writeHead(200,{'content-type':MIME[ext]||'application/octet-stream'}); res.end(buf);
  }catch(e){ res.writeHead(404); res.end('nf'); }
});
await new Promise(r=>srv.listen(8901,r));
const b=await puppeteer.launch({ executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless:'new', userDataDir:'/tmp/ferric-lm-prof', args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=metal'] });
const p=await b.newPage(); const errs=[]; p.on('pageerror',e=>errs.push(e.message.slice(0,180)));
await p.goto('http://localhost:8901/index.html',{waitUntil:'networkidle0',timeout:30000});
const out=await p.evaluate(async()=>{
  const m=await import('./pkg/ferric_web.js'); await m.default();
  if(!navigator.gpu) return 'NO_WEBGPU';
  return await m.ferric_lm_demo('3,14,1,15', 6);
});
console.log('LM_RESULT: '+out);
console.log('pageerrors:', errs.length?errs.slice(0,2):'none');
await b.close(); srv.close();
