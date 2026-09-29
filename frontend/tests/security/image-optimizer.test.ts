// Image-optimizer lock (SECURITY VULNERABILITIES F1).
//
// The SPA renders no `next/image`, so Next's `/_next/image` optimizer has
// no caller. Left enabled it is a live, unauthenticated route that fetches
// image bytes and hands them to sharp → libvips / libheif — the decode path
// behind GHSA-2xp9-vwfh-vxw4 (CVSS 9.5). Three checks keep it gone:
//   1. `next.config.ts` sets `images.unoptimized: true` (route unregistered —
//      verified on a standalone build: `/_next/image` answers 404, not 400).
//   2. No file under `src/` imports `next/image` / `next/legacy/image`, so
//      nothing ever needs the optimizer back.
//   3. The middleware matcher does not exclude `/_next/image`: with the
//      optimizer off, that path renders the HTML not-found page, which must
//      carry the CSP + security-header set like every other page.

import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import nextConfig from "../../next.config";
import { config as middlewareConfig } from "../../middleware";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const srcRoot = path.resolve(repoRoot, "src");

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      out.push(...walk(full));
    } else if (/\.(ts|tsx|js|jsx|mjs)$/.test(entry)) {
      out.push(full);
    }
  }
  return out;
}

// Any module specifier naming the image component — static import,
// re-export, dynamic `import()` or `require()` all quote it the same way.
const NEXT_IMAGE_SPECIFIER_RE = /["'`]next\/(?:legacy\/)?image["'`]/;

// Next compiles a single-group matcher like ours to an anchored regexp;
// reading it the same way lets the lock assert which paths it covers.
function matcherCovers(pathname: string): boolean {
  return middlewareConfig.matcher.some((source) =>
    new RegExp(`^${source}$`).test(pathname),
  );
}

describe("image optimizer stays disabled", () => {
  it("next.config.ts sets images.unoptimized", () => {
    expect(nextConfig.images?.unoptimized).toBe(true);
  });

  it("no SPA source imports next/image", () => {
    const offenders = walk(srcRoot)
      .filter((file) => NEXT_IMAGE_SPECIFIER_RE.test(readFileSync(file, "utf-8")))
      .map((file) => path.relative(repoRoot, file));
    expect(offenders).toEqual([]);
  });

  it("the middleware matcher covers /_next/image", () => {
    expect(matcherCovers("/_next/image")).toBe(true);
    // Controls: the reading above is the real one — a page is covered and a
    // hashed static asset is not.
    expect(matcherCovers("/dashboard")).toBe(true);
    expect(matcherCovers("/_next/static/chunks/app.js")).toBe(false);
  });
});
