import puppeteer from 'puppeteer-core';
const b = await puppeteer.launch({ executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless:'new', userDataDir:'/tmp/ferric-render', args:['--no-sandbox'] });
const p = await b.newPage(); await p.setViewport({width:1200,height:900});
const errs=[]; p.on('pageerror',e=>errs.push(e.message.slice(0,120)));
await p.goto('http://localhost:8799/index.html',{waitUntil:'networkidle0',timeout:20000});
const info = await p.evaluate(()=>({
  heroCta: document.querySelector('.cta .btn')?.textContent,
  heroHref: document.querySelector('.cta .btn')?.getAttribute('href'),
  nav: [...document.querySelectorAll('nav a')].map(a=>a.textContent).join(' | '),
  featured: document.querySelector('.card[href="/bonsai"] h3')?.textContent,
  featuredNum: document.querySelector('.card[href="/bonsai"] .num')?.textContent,
}));
console.log(JSON.stringify(info,null,1));
await p.screenshot({path:'/tmp/ferric-landing.png',fullPage:false});
console.log('errors:', errs.length?errs:'none');
await b.close();
