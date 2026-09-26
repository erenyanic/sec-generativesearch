// External-asset audit.
//
// The SPA is self-hosted by design: no third-party scripts, no CDN
// fonts, no tag managers. Every byte loaded into the page must be
// served from the same origin so the strict CSP (`default-src 'self'`,
// `connect-src 'self'`) and the per-request nonce remain meaningful.
//
// Two cheap static checks pin the contract:
//   1. No `src` / `href` attribute in `src/` points at an off-origin
//      URL (`http://`, `https://`, `//cdn…`).
//   2. No production dependency in `package.json` is a known
//      remote-script loader (analytics, tag manager, CDN font shim).
//      The current dependency set is `next`, `react`, `react-dom` —
//      none of those load external resources at runtime. Update this
//      allow-list with eyes open.

import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const srcRoot = path.resolve(repoRoot, "src");

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      out.push(...walk(full));
    } else if (/\.(ts|tsx|js|mjs|css|html)$/.test(entry)) {
      out.push(full);
    }
  }
  return out;
}

// External-URL attribute pattern: `src=` or `href=` followed by either
// an absolute scheme URL or a protocol-relative URL. We deliberately
// allow internal references like `href="/dashboard"` or
// `src="/local/icon.svg"`.
const EXTERNAL_ASSET_RE =
  /\b(?:src|href)\s*=\s*['"](?:https?:|\/\/)[^'"]+['"]/i;

describe("external-asset audit", () => {
  it("no SPA source loads an off-origin script or stylesheet", () => {
    const offenders: string[] = [];
    for (const file of walk(srcRoot)) {
      const raw = readFileSync(file, "utf-8");
      const match = EXTERNAL_ASSET_RE.exec(raw);
      if (match !== null) {
        offenders.push(`${path.relative(repoRoot, file)}: ${match[0]}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it("package.json's runtime deps do not include known remote-script loaders", () => {
    const pkg = JSON.parse(
      readFileSync(path.resolve(repoRoot, "package.json"), "utf-8"),
    ) as { dependencies?: Record<string, string> };
    const deps = Object.keys(pkg.dependencies ?? {});
    const ALLOW = new Set(["next", "react", "react-dom"]);
    const newcomers = deps.filter((name) => !ALLOW.has(name));
    // A new runtime dep is not necessarily a violation — it is a
    // deliberate, security-reviewed change. Force the reviewer to
    // update this allow-list explicitly.
    expect(newcomers).toEqual([]);
  });
});
