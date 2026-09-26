// Screenshots for the in-app guide, in both themes.
//
//   cd scripts/guide_screens
//   npm install && npx playwright install chromium
//   node capture.mjs                      # against http://localhost:7860, 1TUP
//   node capture.mjs --url https://pdb2print.org --id 1TUP
//
// Writes raw PNGs and the positions of the controls the guide points at to
// raw/<theme>/. build_images.py turns those into the WebP files in
// frontend/img/guide/ and redraws the boxes and numbers in index.html, so
// after a layout change the two commands are the whole job.
//
// CHROMIUM_PATH points it at a browser other than Playwright's own.

import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const URL = arg('--url', 'http://localhost:7860/');
const ID = arg('--id', '1TUP');

async function openPage(browser, vp, theme) {
  const ctx = await browser.newContext({ viewport: vp, deviceScaleFactor: 2, colorScheme: theme });
  // The welcome card would sit over every shot.
  await ctx.addInitScript(() => { try { localStorage.setItem('pdb2print.welcome', 'screenshots'); } catch (e) {} });
  const p = await ctx.newPage();
  p.on('pageerror', e => console.log('page error:', e.message));
  await p.goto(URL);
  await p.waitForTimeout(800);
  await p.evaluate(t => { document.documentElement.dataset.theme = t; const w = document.querySelector('#welcome'); if (w) w.hidden = true; }, theme);
  return p;
}
const openStage = async (p, id) => { await p.evaluate(id => { document.getElementById(id).open = true; }, id); await p.waitForTimeout(350); };
const away = p => p.mouse.move(2, 790);   // off every control, so no popover shows

async function build(p, id) {
  await openStage(p, 'stage-structure');
  await p.fill('#pdb-id', id);
  await p.press('#pdb-id', 'Enter');
  await p.waitForFunction(() => document.querySelector('#dl-3mf')?.getAttribute('aria-disabled') === 'false', null, { timeout: 900000 });
  await p.waitForTimeout(2500);
}

// Page-coordinate boxes, or boxes relative to one element when `rel` is set.
async function boxes(p, sels, rel) {
  const out = {};
  for (const [k, s] of Object.entries(sels)) {
    out[k] = await p.evaluate(([s, rel]) => {
      const e = document.querySelector(s); if (!e) return null;
      const r = e.getBoundingClientRect();
      const o = rel ? document.querySelector(rel).getBoundingClientRect() : { x: 0, y: 0 };
      return [r.x - o.x, r.y - o.y, r.width, r.height].map(Math.round);
    }, [s, rel]);
    if (!out[k]) console.log('not found:', k, s);
  }
  return out;
}

for (const theme of ['light', 'dark']) {
  const out = path.join(HERE, 'raw', theme);
  fs.mkdirSync(out, { recursive: true });
  const B = {};
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });

  // ---- the page after a build, 1280 x 800 ----
  let p = await openPage(browser, { width: 1280, height: 800 }, theme);
  await openStage(p, 'stage-print');
  await p.click('label.switch:has(#magnets)');
  await p.click('label.switch:has(#basepair_connect)');
  await build(p, ID);
  await away(p);
  await p.screenshot({ path: `${out}/overview.png` });
  B.overview = await boxes(p, {
    pdb: '#stage-structure .search-wrap', examples: '#stage-structure .ex-grid',
    style: '#stage-style > summary', print: '#stage-print > summary', generate: '#generate',
    dl3mf: '#dl-3mf', dlstl: '#dl-stl', share: '#dl-share', stand: '#stand-btn', joints: '#joints-btn',
    legend: '#legend', report: '#report', help: '#help-btn', legal: '#legal-btn', theme: '#theme-btn',
  });
  await p.click('#joints-btn'); await away(p); await p.waitForTimeout(800);
  await p.screenshot({ path: `${out}/joints.png` });
  B.joints = await boxes(p, { chains: '#chain-list', seg: '#joint-list .seg', arrow: '#joint-list button[aria-expanded]', go: '#joints-go' });
  await p.click('#joints-close'); await p.waitForTimeout(300);
  await p.click('#stand-btn'); await away(p); await p.waitForTimeout(600);
  await p.screenshot({ path: `${out}/stand_lock.png` });
  B.stand_lock = await boxes(p, { viewer: '#viewer', sheet: '#stand-sheet', roll: '#stand-roll', lock: '#stand-lock' });
  await p.click('#stand-lock'); await away(p); await p.waitForTimeout(4500);
  await p.screenshot({ path: `${out}/stand_panel.png` });
  B.stand_panel = await boxes(p, {
    sketch: '#sp-sketch', unlock: '#sp-unlock', go: '#stand-go',
    sec1: '#stand-panel .sp-body > details:first-of-type > summary',
    sec3: '#stand-panel .sp-body > details:last-of-type > summary',
  });
  await p.context().close();

  // ---- the settings steps on their own, tall enough to show them whole ----
  p = await openPage(browser, { width: 1280, height: 1500 }, theme);
  await openStage(p, 'stage-style');
  await p.click('label.switch:has(#include_ligands)'); await p.waitForTimeout(300);
  await p.locator('#stage-style').screenshot({ path: `${out}/el_style.png` });
  B.el_style = await boxes(p, {
    protein: '[data-group=protein_rep]', adv: '#advanced-surface > summary', dna: '[data-group=nucleic_rep]',
    dnaparts: '#tubeslab', ligsw: 'label.switch:has(#include_ligands)', ligstyle: '[data-group=ligand_style]',
  }, '#stage-style');
  // One setting with its popover open, for "what the i does".
  await p.locator('#stage-style .hx').first().hover(); await p.waitForTimeout(300);
  const r = await p.evaluate(() => {
    const a = document.querySelector('#stage-style').getBoundingClientRect(), t = document.querySelector('#tip').getBoundingClientRect();
    return { x: a.x, y: a.y, width: t.right - a.x + 12, height: 270 };
  });
  await p.screenshot({ path: `${out}/el_tooltip.png`, clip: r });
  await p.mouse.move(1200, 1400);
  await p.click('label.switch:has(#include_ligands)');
  await openStage(p, 'stage-print');
  await p.click('label.switch:has(#magnets)'); await p.waitForTimeout(300);
  await p.locator('#stage-print').screenshot({ path: `${out}/el_print.png` });
  B.el_print = await boxes(p, {
    scale: '#stage-print [data-slider=scale]', adv: '#advanced-global > summary', assembly: '[data-group=assembly]',
    magsw: 'label.switch:has(#magnets)', size: '#magnet-size', count: '.count-grid',
    socket: 'label.switch:has(#socket)', bp: 'label.switch:has(#basepair_connect)',
  }, '#stage-print');
  await browser.close();
  fs.writeFileSync(`${out}/boxes.json`, JSON.stringify(B, null, 1));
  console.log('captured', theme);
}
