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
  // Match every request path except file-extension paths, /_next/static and
  // favicon.ico. `/_next/image` is deliberately NOT excluded: the optimizer
  // is disabled (next.config.ts), so that path renders the HTML not-found
  // page, which must carry the CSP + header set like any other page.
  matcher: ["/((?!.*\\.[a-zA-Z0-9]+$|_next/static|favicon.ico).*)"],
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
