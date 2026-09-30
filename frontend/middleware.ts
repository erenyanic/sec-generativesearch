import { NextResponse, type NextRequest } from "next/server";

import {
  buildContentSecurityPolicy,
  generateNonce,
  staticSecurityHeaders,
} from "@/lib/security-headers";

// Edge runtime: lighter, no Node.js APIs, faster cold-start. Web Crypto is
// available in both the Edge and Node runtimes; sticking to Edge keeps the
// security surface minimal.
export const config = {
  // Match every request path except the hashed build assets under
  // `/_next/static/` (trailing slash included). Any path Next does not serve
  // as a file renders the HTML not-found page, which reflects the path and
  // query into an inline script — so it must carry the nonce'd CSP + header
  // set like every other page. Excluding a path by its *shape* (a `.ext`
  // suffix, `favicon.ico`, `_next/static` without the slash) sent that page
  // out header-less; nothing is served from `public/`, so no such exclusion
  // is needed. `/_next/image` is covered too: the optimizer is disabled
  // (next.config.ts), so it renders the same page. Locked by
  // tests/security/middleware-matcher.test.ts.
  matcher: ["/((?!_next/static/).*)"],
};

const IS_DEVELOPMENT = process.env.NODE_ENV === "development";

export function middleware(request: NextRequest): NextResponse {
  const nonce = generateNonce();
  const csp = buildContentSecurityPolicy(nonce, IS_DEVELOPMENT);

  // Forward the nonce to the React tree via a request header so
  // `headers().get("x-nonce")` works in server components / layout.
  // Mutating the request via NextResponse.next({ request }) is the
  // documented seam — but we set response headers on the returned
  // response too, so the browser sees the CSP set.
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);

  const response = NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });

  response.headers.set("Content-Security-Policy", csp);
  for (const [name, value] of Object.entries(staticSecurityHeaders())) {
    response.headers.set(name, value);
  }

  return response;
}
