// Server-side admin proxy.
//
// The proxy is the single seam where the admin key is injected into a
// backend request. Asserts:
//  - unauthenticated callers get 401 with no upstream call
//  - authenticated callers get X-API-Key + X-Admin-Key injected
//  - client-set X-API-Key / X-Admin-Key headers are stripped, not merged
//  - path-traversal segments are rejected
//  - non-allow-listed backend paths are 403'd before reaching the network
//  - backend session_id cookie is forwarded; admin_session cookie is not

import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ADMIN_SESSION_COOKIE,
  _resetForTests,
  createSession,
} from "@/lib/admin-session";

// Import the route handlers AFTER mocking modules they depend on.
import * as proxyRoute from "@/app/api/admin/[...path]/route";

// vi.spyOn does not work cleanly against global fetch without restoreAll.
const originalFetch = globalThis.fetch;

function makeRequest(
  url: string,
  init: RequestInit & { cookies?: Record<string, string> } = {},
): NextRequest {
  // `Cookie` is a forbidden header in browser environments (happy-dom
  // strips it from Headers init). Set via NextRequest.cookies.set() after
  // construction, which writes the canonical cookie store directly.
  const req = new NextRequest(url, {
    method: init.method ?? "GET",
    headers: new Headers(init.headers),
    body: init.body ?? null,
  });
  if (init.cookies !== undefined) {
    for (const [name, value] of Object.entries(init.cookies)) {
      req.cookies.set(name, value);
    }
  }
  return req;
}

type ProxyHandler = (
  req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
) => Promise<Response>;

const PROXY_HANDLERS: Record<"GET" | "POST" | "DELETE", ProxyHandler> = {
  GET: proxyRoute.GET as unknown as ProxyHandler,
  POST: proxyRoute.POST as unknown as ProxyHandler,
  DELETE: proxyRoute.DELETE as unknown as ProxyHandler,
};

async function callHandler(
  method: "GET" | "POST" | "DELETE",
  pathSegments: string[],
  init: RequestInit & { cookies?: Record<string, string>; search?: string } = {},
): Promise<Response> {
  const segs = pathSegments.map((s) => encodeURIComponent(s)).join("/");
  const url = `https://app.test/api/admin/${segs}${init.search ?? ""}`;
  const req = makeRequest(url, { ...init, method });
  return PROXY_HANDLERS[method](req, {
    params: Promise.resolve({ path: pathSegments }),
  });
}

beforeEach(() => {
  _resetForTests();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
  _resetForTests();
});

describe("proxy auth gate", () => {
  it("returns 401 when no admin_session cookie is present", async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const res = await callHandler("GET", ["filings", ""]);
    expect(res.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("returns 401 for a forged cookie that does not match any session", async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const res = await callHandler("GET", ["filings"], {
      cookies: { [ADMIN_SESSION_COOKIE]: "A".repeat(43) },
    });
    expect(res.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("returns 401 for a malformed cookie shape", async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const res = await callHandler("GET", ["filings"], {
      cookies: { [ADMIN_SESSION_COOKIE]: "not-base64!" },
    });
    expect(res.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("path allow-list", () => {
  it("rejects backend paths that are not in the allow-list", async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    // `debug` is a deliberately fictitious backend prefix — it MUST NOT
    // appear in `ALLOWED_PATH_PREFIXES` for this test to remain useful.
    const res = await callHandler("GET", ["debug"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    expect(res.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects path-traversal segments", async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    const res = await callHandler("GET", ["filings", "..", "secret"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("allow-lists session-tier routes (mint, logout, edgar)", async () => {
    const fetchMock = vi.fn(
      async () => new Response("{}", { status: 200 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret

    for (const path of [["session"], ["session", "logout"], ["session", "edgar"]]) {
      const res = await callHandler("POST", path, {
        cookies: { [ADMIN_SESSION_COOKIE]: id },
      });
      expect(res.status).toBe(200);
    }
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("allow-lists rag/ (plan + stream)", async () => {
    const fetchMock = vi.fn(
      async () => new Response("{}", { status: 200 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret

    for (const path of [["rag", "plan"], ["rag", "stream"]]) {
      const res = await callHandler("POST", path, {
        cookies: { [ADMIN_SESSION_COOKIE]: id },
      });
      expect(res.status).toBe(200);
    }
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("rejects rag/ subpaths that contain a path-traversal segment", async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    const res = await callHandler("POST", ["rag", "..", "filings"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("allow-lists providers/ (catalogue) and providers/validate", async () => {
    const fetchMock = vi.fn(
      async () => new Response("{}", { status: 200 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret

    const getRes = await callHandler("GET", ["providers"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    expect(getRes.status).toBe(200);
    const postRes = await callHandler("POST", ["providers", "validate"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    expect(postRes.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("allow-lists auth/ (login-params, login, enrol, password, vault, session)", async () => {
    const fetchMock = vi.fn(
      async () => new Response("{}", { status: 200 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret

    for (const [method, segments] of [
      ["GET", ["auth", "login-params"]],
      ["POST", ["auth", "login"]],
      ["POST", ["auth", "enrol"]],
      ["POST", ["auth", "password"]],
      ["POST", ["auth", "vault"]],
      ["DELETE", ["auth", "session"]],
    ] as const) {
      const res = await callHandler(method, [...segments], {
        cookies: { [ADMIN_SESSION_COOKIE]: id },
      });
      expect(res.status).toBe(200);
    }
    expect(fetchMock).toHaveBeenCalledTimes(6);
  });

  it("allow-lists admin/users/ (mint, delete, unlock)", async () => {
    const fetchMock = vi.fn(
      async () => new Response("{}", { status: 200 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret

    for (const [method, segments] of [
      ["POST", ["admin", "users"]],
      ["DELETE", ["admin", "users", "42"]],
      ["POST", ["admin", "users", "42", "unlock"]],
    ] as const) {
      const res = await callHandler(method, [...segments], {
        cookies: { [ADMIN_SESSION_COOKIE]: id },
      });
      expect(res.status).toBe(200);
    }
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("forwards auth/* routes to the backend /api/auth/* path", async () => {
    const fetchMock = vi.fn(
      async () => new Response("{}", { status: 200 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    await callHandler("POST", ["auth", "login"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    const call = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(String(call[0])).toContain("/api/auth/login");
  });

  it("forwards admin/users/* to the backend /api/admin/users/* path", async () => {
    const fetchMock = vi.fn(
      async () => new Response("{}", { status: 200 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    await callHandler("POST", ["admin", "users"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    const call = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(String(call[0])).toContain("/api/admin/users");
  });
});

describe("header injection", () => {
  function mockBackend(): ReturnType<typeof vi.fn> {
    const fetchMock = vi.fn(async () => {
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    return fetchMock;
  }

  it("injects X-API-Key and X-Admin-Key on the forwarded request", async () => {
    const fetchMock = mockBackend();
    const id = createSession("real-api", "real-admin"); // pragma: allowlist secret
    await callHandler("GET", ["filings"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const firstCall = fetchMock.mock.calls[0];
    expect(firstCall).toBeDefined();
    const [, init] = firstCall as unknown as [string, RequestInit];
    const headers = new Headers(init.headers as HeadersInit);
    expect(headers.get("x-api-key")).toBe("real-api");
    expect(headers.get("x-admin-key")).toBe("real-admin");
  });

  it("strips any client-set X-API-Key / X-Admin-Key", async () => {
    const fetchMock = mockBackend();
    const id = createSession("server-api", "server-admin"); // pragma: allowlist secret
    await callHandler("GET", ["filings"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
      headers: {
        "X-API-Key": "spoofed-api",
        "X-Admin-Key": "spoofed-admin",
      },
    });
    const firstCall = fetchMock.mock.calls[0];
    expect(firstCall).toBeDefined();
    const [, init] = firstCall as unknown as [string, RequestInit];
    const headers = new Headers(init.headers as HeadersInit);
    // The spoofed values must be gone — only the server-side keys remain.
    expect(headers.get("x-api-key")).toBe("server-api");
    expect(headers.get("x-admin-key")).toBe("server-admin");
  });

  it("strips client-set X-Forwarded-* headers (M2: rate-limit key integrity)", async () => {
    // Load-bearing for the Cloud Run half of M2. The API image ships with no
    // baked trusted-proxy set, and the Cloud Run manifest deliberately leaves
    // FORWARDED_ALLOW_IPS unset, on the strength of THIS strip: the proxy is
    // the API's only peer there, so if it forwarded a caller-supplied
    // X-Forwarded-For the backend's per-IP rate-limit key would become
    // client-controlled again. Removing any entry below silently re-opens
    // that path, which is why it is pinned here rather than left implicit.
    const fetchMock = mockBackend();
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    await callHandler("GET", ["filings"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
      headers: {
        "X-Forwarded-For": "198.51.100.9",
        "X-Forwarded-Host": "evil.example",
        "X-Forwarded-Proto": "http",
      },
    });
    const firstCall = fetchMock.mock.calls[0];
    expect(firstCall).toBeDefined();
    const [, init] = firstCall as unknown as [string, RequestInit];
    const headers = new Headers(init.headers as HeadersInit);
    expect(headers.get("x-forwarded-for")).toBeNull();
    expect(headers.get("x-forwarded-host")).toBeNull();
    expect(headers.get("x-forwarded-proto")).toBeNull();
  });

  it("forwards backend session_id cookie but never the admin_session cookie", async () => {
    const fetchMock = mockBackend();
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    await callHandler("GET", ["filings"], {
      cookies: {
        [ADMIN_SESSION_COOKIE]: id,
        session_id: "backend-cookie-xyz",
      },
    });
    const firstCall = fetchMock.mock.calls[0];
    expect(firstCall).toBeDefined();
    const [, init] = firstCall as unknown as [string, RequestInit];
    const headers = new Headers(init.headers as HeadersInit);
    const cookie = headers.get("cookie") ?? "";
    expect(cookie).toContain("session_id=backend-cookie-xyz");
    expect(cookie).not.toContain(ADMIN_SESSION_COOKIE);
  });

  it("preserves the trailing slash on the backend path", async () => {
    const fetchMock = mockBackend();
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    // Next's catch-all router yields `["filings"]` here; the trailing slash
    // is read off `nextUrl.pathname`. Force the slash via `search` is not
    // possible — set it directly on the request URL.
    await callHandler("GET", ["filings"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
      search: "/",
    });
    const firstCall = fetchMock.mock.calls[0];
    expect(firstCall).toBeDefined();
    const [url] = firstCall as unknown as [string, RequestInit];
    expect(url).toMatch(/\/api\/filings\//);
  });

  it("forwards X-Provider-Key-* headers verbatim", async () => {
    const fetchMock = mockBackend();
    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    await callHandler("POST", ["providers", "validate"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
      headers: {
        "X-Provider-Key-openai": "sk-FORWARD-ME",
      },
    });
    const firstCall = fetchMock.mock.calls[0];
    expect(firstCall).toBeDefined();
    const [, init] = firstCall as unknown as [string, RequestInit];
    const headers = new Headers(init.headers as HeadersInit);
    // The per-provider header IS forwarded — only operator auth headers
    // are stripped + replaced.
    expect(headers.get("x-provider-key-openai")).toBe("sk-FORWARD-ME");
    // Sanity: the server-held keys still go up too.
    expect(headers.get("x-api-key")).toBe("api-k");
  });

  it("never echoes either key into the response body", async () => {
    mockBackend();
    const id = createSession("very-secret-api", "very-secret-admin"); // pragma: allowlist secret
    const res = await callHandler("GET", ["filings"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    const text = await res.text();
    expect(text).not.toContain("very-secret-api");
    expect(text).not.toContain("very-secret-admin");
  });

  it("forwards Set-Cookie from the backend without collapsing multi-value cookies", async () => {
    // The backend session_id cookie must reach the browser intact —
    // `Headers.forEach` collapses multi-Set-Cookie into a comma-joined
    // value, so we route through `getSetCookie()` in the proxy.
    const upstream = new Response("{}", {
      status: 201,
      headers: { "content-type": "application/json" },
    });
    upstream.headers.append(
      "Set-Cookie",
      "session_id=abc123; HttpOnly; Secure; SameSite=Strict; Path=/",
    );
    upstream.headers.append(
      "Set-Cookie",
      "other_flag=on; Path=/",
    );
    const fetchMock = vi.fn(async () => upstream);
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const id = createSession("api-k", "admin-k"); // pragma: allowlist secret
    const res = await callHandler("POST", ["session"], {
      cookies: { [ADMIN_SESSION_COOKIE]: id },
    });
    const cookies = res.headers.getSetCookie();
    expect(cookies).toHaveLength(2);
    expect(cookies[0]).toContain("session_id=abc123");
    expect(cookies[0]).toContain("HttpOnly");
    expect(cookies[1]).toContain("other_flag=on");
  });
});
