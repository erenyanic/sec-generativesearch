// Catch-all server-side proxy for admin-tier backend routes.
//
// Browsers call `/api/admin/<backend-path>` with the `admin_session`
// HttpOnly cookie. The handler looks up the operator's keys server-side,
// injects `X-API-Key` + `X-Admin-Key` into the forwarded request, and
// streams the backend response back. The admin key NEVER reaches the
// browser at any point in this pipeline.

import { NextResponse, type NextRequest } from "next/server";

import { ADMIN_SESSION_COOKIE, lookupSession } from "@/lib/admin-session";
import { buildBackendUrl } from "@/lib/backend";

export const runtime = "nodejs";

// Allow-list of backend path prefixes the proxy will forward to. New admin
// routes must be added here explicitly — accidental over-exposure of a
// destructive surface is the failure mode this list defends against.
//
// Session-tier routes (`session`, `session/edgar`, `session/logout`) are
// reached through the same proxy because session minting and EDGAR
// registration still need the server-held `X-API-Key`.
//
// The allow-list includes `auth/` (user-tier login, enrolment, password
// change, vault re-upload, sign-out) and `admin/users/` (admin mints /
// deletes / unlocks user enrolments). The browser reaches the admin tier
// as `/api/admin/admin/users/...` — the leading segment disambiguates the
// proxy from the backend's own admin namespace.
const ALLOWED_PATH_PREFIXES = [
  "filings/",
  "filings",
  "ingest/",
  "status/",
  "status",
  "resources/",
  "session/",
  "session",
  "providers/",
  "providers",
  "rag/",
  "auth/",
  "admin/users/",
  "admin/users",
] as const;

// Headers that must NEVER be carried verbatim from the browser into the
// backend request — they are either set by the proxy itself or would let a
// hostile caller spoof the auth tier.
const STRIPPED_REQUEST_HEADERS = new Set([
  "host",
  "x-api-key",
  "x-admin-key",
  // The scheduled demo-reset token (F27) is never a browser credential.
  "x-demo-reset-token",
  "x-forwarded-for",
  "x-forwarded-host",
  "x-forwarded-proto",
  "content-length",
  "cookie",
  "connection",
]);

// Headers we strip from the backend response before returning it to the
// browser. The browser receives the SPA's own security headers (set by
// middleware.ts); backend-set CORS/security headers would be confusing or
// permissive and are not needed for same-origin proxy responses.
const STRIPPED_RESPONSE_HEADERS = new Set([
  "access-control-allow-origin",
  "access-control-allow-credentials",
  "access-control-allow-headers",
  "access-control-allow-methods",
  "content-security-policy",
  "strict-transport-security",
  "x-frame-options",
  "transfer-encoding",
  "connection",
]);

const ALLOWED_METHODS = new Set([
  "GET",
  "POST",
  "PUT",
  "PATCH",
  "DELETE",
]);

function isAllowedPath(joined: string): boolean {
  for (const prefix of ALLOWED_PATH_PREFIXES) {
    if (joined === prefix || joined.startsWith(prefix)) {
      return true;
    }
  }
  return false;
}

function envelope(
  error: string,
  message: string,
): { error: string; message: string } {
  return { error, message };
}

async function handle(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
): Promise<NextResponse> {
  if (!ALLOWED_METHODS.has(request.method)) {
    return NextResponse.json(
      envelope("method_not_allowed", `method ${request.method} is not proxied`),
      { status: 405 },
    );
  }

  const sessionCookie = request.cookies.get(ADMIN_SESSION_COOKIE)?.value;
  const keys = lookupSession(sessionCookie);
  if (keys === null) {
    return NextResponse.json(
      envelope("unauthenticated", "admin session is required"),
      { status: 401 },
    );
  }

  const { path } = await context.params;
  if (!Array.isArray(path) || path.length === 0) {
    return NextResponse.json(
      envelope("invalid_path", "missing backend path"),
      { status: 400 },
    );
  }
  // Reject path segments that contain path-traversal or scheme tokens.
  // Next decodes the segments before invoking the handler; an attacker who
  // sneaks a `..` past Next would otherwise reach unrelated backend routes.
  for (const segment of path) {
    if (
      segment === "" ||
      segment === "." ||
      segment === ".." ||
      segment.includes("/") ||
      segment.includes("\\")
    ) {
      return NextResponse.json(
        envelope("invalid_path", "illegal path segment"),
        { status: 400 },
      );
    }
  }

  const joined = path.join("/");
  if (!isAllowedPath(joined)) {
    return NextResponse.json(
      envelope("path_not_allowed", "this backend route is not proxied"),
      { status: 403 },
    );
  }

  // Preserve the trailing slash from the inbound URL — FastAPI's routes are
  // slash-sensitive and a redirect would leak the admin key on the next
  // hop in some clients.
  const inboundPath = request.nextUrl.pathname;
  const trailingSlash = inboundPath.endsWith("/") ? "/" : "";
  const backendPath = `/api/${joined}${trailingSlash}`;
  const backendUrl = buildBackendUrl(backendPath) + request.nextUrl.search;

  const forwardedHeaders = new Headers();
  request.headers.forEach((value, key) => {
    if (!STRIPPED_REQUEST_HEADERS.has(key.toLowerCase())) {
      forwardedHeaders.set(key, value);
    }
  });
  forwardedHeaders.set("X-API-Key", keys.apiKey);
  forwardedHeaders.set("X-Admin-Key", keys.adminKey);

  // Forward the BACKEND `session_id` cookie only — every other cookie in
  // the browser jar (including our own `admin_session`) is irrelevant to
  // the backend and would muddy the request.
  const backendSessionCookie = request.cookies.get("session_id");
  if (backendSessionCookie !== undefined) {
    forwardedHeaders.set(
      "Cookie",
      `session_id=${backendSessionCookie.value}`,
    );
  }

  // Read body for non-GET methods. `request.body` is a `ReadableStream`;
  // node-fetch in Node 22 accepts it directly when `duplex: "half"` is set.
  const hasBody = !(request.method === "GET" || request.method === "HEAD");
  const init: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers: forwardedHeaders,
    cache: "no-store",
    redirect: "manual",
  };
  if (hasBody) {
    init.body = request.body;
    init.duplex = "half";
  }

  let upstream: Response;
  try {
    upstream = await fetch(backendUrl, init);
  } catch {
    return NextResponse.json(
      envelope("backend_unreachable", "backend did not respond"),
      { status: 502 },
    );
  }

  const responseHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    const lower = key.toLowerCase();
    if (lower === "set-cookie") {
      // Set-Cookie is handled below via getSetCookie() so multi-cookie
      // responses (e.g. session_id mint) reach the browser intact.
      return;
    }
    if (!STRIPPED_RESPONSE_HEADERS.has(lower)) {
      responseHeaders.set(key, value);
    }
  });
  responseHeaders.set("Cache-Control", "no-store");

  const proxied = new NextResponse(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders,
  });
  // Append every Set-Cookie individually AFTER construction. The Response
  // constructor in some runtimes (notably the Fetch spec's `fill`
  // algorithm when given a Headers init) drops multi-valued Set-Cookie
  // headers, so we set them on the live Headers object instead.
  for (const cookie of upstream.headers.getSetCookie()) {
    proxied.headers.append("Set-Cookie", cookie);
  }
  return proxied;
}

export const GET = handle;
export const POST = handle;
export const PUT = handle;
export const PATCH = handle;
export const DELETE = handle;
