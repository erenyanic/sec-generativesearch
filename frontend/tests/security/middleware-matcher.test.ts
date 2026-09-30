// Middleware-matcher lock (SECURITY VULNERABILITIES F6).
//
// The middleware is the only place the CSP nonce and the static security-
// header set are written. Any path it skips that Next does not serve as a
// file renders the HTML not-found page — nine inline scripts, with the
// request path and query reflected into the RSC flight payload — and, before
// this lock, those responses went out with no CSP, no nonce, no
// X-Frame-Options, no nosniff and no HSTS. The old matcher excluded paths by
// *shape* (`.ext`, `favicon.ico`, `_next/static` without the slash), so
// `/foo.txt`, `/favicon.ico`, `/_next/staticX` and `/_next/data/<id>/x.json`
// all got that page header-less (verified on the standalone build).
//
// The invariant: the only paths the matcher may skip are the hashed build
// assets under `/_next/static/` (trailing slash included). Everything is
// read through Next's own matcher compiler, so the lock tests the regexp the
// build writes to middleware-manifest.json (its optional `/_next/data/<id>`
// prefix and `.json` / `.rsc` transport suffixes included), not a
// hand-simplified copy.

import { createRequire } from "node:module";

import type { NextConfig } from "next";
import { describe, expect, it } from "vitest";

import nextConfig from "../../next.config";
import { config as middlewareConfig } from "../../middleware";

// Next exports its matcher compiler at runtime but not in its typings. If an
// upgrade moves it, the guard below fails loudly — update the path, never
// fall back to a simplified regexp.
type MatcherCompiler = (
  matchers: readonly unknown[],
  config: NextConfig,
) => Array<{ regexp: string }>;
const nodeRequire = createRequire(import.meta.url);
const { getMiddlewareMatchers } = nodeRequire(
  "next/dist/build/analysis/get-page-static-info",
) as { getMiddlewareMatchers?: MatcherCompiler };

function compiledMatchers(): RegExp[] {
  if (typeof getMiddlewareMatchers !== "function") {
    throw new Error(
      "next/dist/build/analysis/get-page-static-info no longer exports getMiddlewareMatchers",
    );
  }
  return getMiddlewareMatchers(middlewareConfig.matcher, nextConfig).map(
    ({ regexp }) => new RegExp(regexp),
  );
}

function covers(pathname: string): boolean {
  return compiledMatchers().some((re) => re.test(pathname));
}

const STATIC_ASSET_PREFIX = "/_next/static/";

describe("middleware matcher skips only /_next/static/ build assets", () => {
  it("covers the not-found shapes the old exclusions sent out header-less", () => {
    const formerlyExcluded = [
      // `.ext` suffix
      "/foo.txt",
      "/dashboard.json",
      "/robots.txt",
      "/index.html",
      "/.well-known/security.txt",
      "/dashboard/x.y",
      "/%3Cimg%20src=x%20onerror=alert(1)%3E.txt",
      // favicon.ico (public/ is empty, so it is a not-found page too)
      "/favicon.ico",
      "/favicon.icoX",
      "/favicon.ico/x",
      // `_next/static` without the trailing slash
      "/_next/static",
      "/_next/staticX",
      "/_next/static%2f..%2fdashboard",
      // Next's data-route prefix + `.json` suffix
      "/_next/data/build-id/dashboard.json",
    ];
    expect(formerlyExcluded.filter((p) => !covers(p))).toEqual([]);
  });

  it("covers pages, the disabled image optimizer and the admin proxy", () => {
    const pages = [
      "/",
      "/dashboard",
      "/nonexistent",
      "/_next/image",
      "/api/admin/session",
      // A page's RSC transport form; the old `.ext` exclusion skipped it.
      "/dashboard.rsc",
    ];
    expect(pages.filter((p) => !covers(p))).toEqual([]);
  });

  it("skips the hashed build assets (control: the reading is not vacuous)", () => {
    const assets = [
      "/_next/static/chunks/app.js",
      "/_next/static/css/app.css",
      "/_next/static/media/font.woff2",
      "/_next/static/build-id/_buildManifest.js",
    ];
    expect(assets.filter((p) => covers(p))).toEqual([]);
  });

  it("skips nothing outside /_next/static/ across enumerated path shapes", () => {
    // Every 1–3 segment path over segments that name each exclusion shape
    // the old matcher used, plus traversal and encoded separators. The
    // invariant must hold for all of them — a new exclusion of any shape
    // (a suffix, a filename, a prefix without its slash) breaks it.
    const segments = [
      "",
      "_next",
      "static",
      "staticX",
      "image",
      "data",
      "build-id",
      "chunks",
      "app.js",
      "favicon.ico",
      "foo.txt",
      "x.y",
      ".well-known",
      "..",
      "%2f",
      "dashboard",
      "dashboard.json",
      "index.rsc",
      "api",
      "admin",
    ];
    const paths = new Set<string>();
    for (const a of segments) {
      paths.add(`/${a}`);
      for (const b of segments) {
        paths.add(`/${a}/${b}`);
        for (const c of segments) {
          paths.add(`/${a}/${b}/${c}`);
        }
      }
    }
    const violations = [...paths].filter(
      (p) => covers(p) === p.startsWith(STATIC_ASSET_PREFIX),
    );
    expect(violations).toEqual([]);
    // Control: the enumeration does reach the one sanctioned exclusion.
    expect([...paths].some((p) => p.startsWith(STATIC_ASSET_PREFIX))).toBe(
      true,
    );
  });

  it("does not make coverage conditional on request headers or cookies", () => {
    // A `has` / `missing` condition would let a request skip the middleware
    // by header (e.g. `Accept`), reopening the header-less page.
    for (const entry of middlewareConfig.matcher as readonly unknown[]) {
      expect(typeof entry).toBe("string");
    }
  });
});
