// `next/og` lock (SECURITY VULNERABILITIES F7).
//
// GHSA-vcvr-r3jv-pc5j (Critical, `next >=16.2.0 <16.3.6`) is a remote code
// execution in the Node.js `ImageResponse` from `next/og` when attacker-
// controlled values reach the rendered SVG. The fix is the `next` bump, but
// the SPA never generates images, so — like the image optimizer (F1) — the
// render surface should stay unused rather than merely patched. Two checks:
//   1. No module under `src/` (or `middleware.ts`) imports `next/og` or its
//      upstream `@vercel/og`.
//   2. No code-generated metadata image route exists under `src/app/`
//      (`opengraph-image`, `twitter-image`, `icon`, `apple-icon` as
//      .ts/.tsx/.js/.jsx/.mjs): those files render through `ImageResponse`.
//      A static image file (`icon.png`) is served as-is and stays allowed.

import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const srcRoot = path.resolve(repoRoot, "src");
const appRoot = path.resolve(srcRoot, "app");

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      out.push(...walk(full));
    } else {
      out.push(full);
    }
  }
  return out;
}

const CODE_FILE_RE = /\.(ts|tsx|js|jsx|mjs)$/;

// Any module specifier naming the image generator — static import,
// re-export, dynamic `import()` or `require()` all quote it the same way.
const OG_SPECIFIER_RE = /["'`](?:next\/og|@vercel\/og)["'`]/;

// Metadata image routes Next renders from code (optionally numbered, e.g.
// `opengraph-image2.tsx`).
const METADATA_IMAGE_ROUTE_RE =
  /^(?:opengraph-image|twitter-image|icon|apple-icon)\d*\.(ts|tsx|js|jsx|mjs)$/;

describe("next/og image generation stays unused", () => {
  it("no SPA source imports next/og or @vercel/og", () => {
    const files = [
      ...walk(srcRoot).filter((file) => CODE_FILE_RE.test(file)),
      path.resolve(repoRoot, "middleware.ts"),
    ];
    // Control: the scan reads real sources.
    expect(files.length).toBeGreaterThan(10);
    const offenders = files
      .filter((file) => OG_SPECIFIER_RE.test(readFileSync(file, "utf-8")))
      .map((file) => path.relative(repoRoot, file));
    expect(offenders).toEqual([]);
  });

  it("no code-generated metadata image route exists", () => {
    const offenders = walk(appRoot)
      .filter((file) => METADATA_IMAGE_ROUTE_RE.test(path.basename(file)))
      .map((file) => path.relative(repoRoot, file));
    expect(offenders).toEqual([]);
  });

  it("the patterns recognise what they ban (control: not vacuous)", () => {
    expect(OG_SPECIFIER_RE.test('import { ImageResponse } from "next/og";')).toBe(true);
    expect(OG_SPECIFIER_RE.test("const og = await import('@vercel/og');")).toBe(true);
    expect(METADATA_IMAGE_ROUTE_RE.test("opengraph-image.tsx")).toBe(true);
    expect(METADATA_IMAGE_ROUTE_RE.test("icon.png")).toBe(false);
  });
});
