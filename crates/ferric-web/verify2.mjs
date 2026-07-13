import puppeteer from 'puppeteer-core';
const b=await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:'new',userDataDir:'/tmp/ferric-v2',args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=metal']});
async function check(url){
  const p=await b.newPage(); const errs=[]; p.on('pageerror',e=>errs.push(e.message.slice(0,140)));
  await p.goto(url,{waitUntil:'networkidle0',timeout:40000}); await new Promise(r=>setTimeout(r,2500));
  const d=await p.evaluate(()=>({h1:document.querySelector('h1')?.innerText?.replace(/\n/g,' '),cards:document.querySelectorAll('.card').length,crates:document.querySelectorAll('.crate').length,backend:document.getElementById('backend')?.innerText,diff:(document.getElementById('diff')?.innerText||'').slice(0,40)}));
  console.log(url+' → '+JSON.stringify(d)+' errs:'+(errs.length?errs[0]:'none')); await p.close();
}
await check('https://ef5b74b3.ferric.pages.dev/');
await check('https://ferric.pages.dev/');
await b.close();
