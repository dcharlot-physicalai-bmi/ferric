import puppeteer from 'puppeteer-core';
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const b = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
  userDataDir: '/tmp/ferric-embed', protocolTimeout: 900000,
  args: ['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=metal'] });
const p = await b.newPage();
p.on('pageerror', e => console.log('PAGEERROR: ' + e.message.slice(0,200)));
await p.goto('http://localhost:8770/embed.html', { waitUntil: 'domcontentloaded', timeout: 60000 });
let out; try { out = await p.evaluate(() => window.runEmbed()); }
catch (e) { out = 'THREW: ' + String(e && e.message || e).slice(0,200); }
console.log('EMBED: ' + out);
await b.close();
