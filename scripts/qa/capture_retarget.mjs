// Headless capture of the 3D-LEX → X Bot retarget preview → PNGs.
// Renders the playing clip at a few frozen times (?t=) and/or live, front+side.
//
// Usage:
//   node scripts/qa/capture_retarget.mjs <sign> [cam] [extraQuery] [waitMs]
//   node scripts/qa/capture_retarget.mjs clean front "mirror=1" 3500
//
// Output: /tmp/avatar-shots/RT_<sign>_<cam>[_<tag>].png
//
// Requires the Next dev server on :3000 and Playwright (run from /tmp/pw which
// has chromium installed): `node /Users/.../scripts/qa/capture_retarget.mjs ...`
// but launch chromium with the swiftshader flags below.

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const [sign = "clean", cam = "front", extra = "", waitMs = "3500"] = process.argv.slice(2);
const OUT = "/tmp/avatar-shots";
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({
  headless: true,
  args: [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
  ],
});
const page = await browser.newPage({ viewport: { width: 560, height: 640 } });
const errors = [];
page.on("console", (m) => {
  const t = m.text();
  if (t.includes("[retarget]")) console.log(`[browser] ${t}`);
  if (m.type() === "error") errors.push(t);
});
page.on("pageerror", (e) => errors.push(e.message));

const qs = `sign=${encodeURIComponent(sign)}&cam=${cam}${extra ? "&" + extra : ""}`;
const url = `http://localhost:3000/dev/retarget-preview?${qs}`;
console.log("GET", url);
await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForSelector("canvas", { timeout: 30000 });
await page.waitForTimeout(parseInt(waitMs, 10)); // model load + warmup + a bit of motion

const tag = extra ? "_" + extra.replace(/[^a-z0-9]/gi, "") : "";
const file = `${OUT}/RT_${sign}_${cam}${tag}.png`;
await page.screenshot({ path: file });
console.log("saved", file, errors.length ? "ERR: " + [...new Set(errors)].slice(0, 4).join(" | ") : "ok");
await browser.close();
