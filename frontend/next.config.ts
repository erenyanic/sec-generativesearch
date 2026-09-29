import type { NextConfig } from "next";

// Per-request CSP nonces live in middleware.ts, not here — config-level
// `headers()` cannot mint a per-request value. Static security headers that
// never need a nonce are still set in middleware too, so the whole header
// surface stays in one auditable place.
const config: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  productionBrowserSourceMaps: false,
  // Self-contained server bundle for the container image. Next traces the
  // minimal runtime dependency set into `.next/standalone/`, so the runtime
  // image copies that tree + `.next/static` and runs `node server.js` with NO
  // `node_modules` install and NO pnpm at runtime. The standalone server is
  // produced by `next build`, so the Turbopack-compiled middleware (and its
  // per-request CSP headers) ships correctly — unlike `next dev`.
  output: "standalone",
  // The SPA renders no `next/image`, so the `/_next/image` optimizer has no
  // caller — yet left enabled it is a live, unauthenticated route that feeds
  // fetched bytes to sharp → libvips / libheif (the GHSA-2xp9-vwfh-vxw4 decode
  // path). `unoptimized` unregisters the route (404). Locked by
  // tests/security/image-optimizer.test.ts together with a no-`next/image` scan.
  images: { unoptimized: true },
  experimental: {
    // Trusted Types is enabled via the CSP header in middleware; this flag
    // tells Next to opt its hydration sinks into a default policy at build.
    // See src/app/layout.tsx for the client-side default policy installer.
  },
};

export default config;
