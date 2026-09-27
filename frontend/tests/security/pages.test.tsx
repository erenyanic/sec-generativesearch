// Page-level smoke + security tests for Dashboard, Filings, Ingest.
//
// We mount each page in isolation (the auth gate is exercised in
// welcome-gate.test.tsx) and assert:
//   - the page issues a request through the admin proxy, never the raw
//     backend
//   - rendered output never contains user-controlled key material
//   - error envelopes surface without echoing the offending input
//
// `next/navigation` is mocked so `usePathname` returns a predictable
// value inside `AppShell` (which is imported transitively by the
// authenticated route layout — but we render pages directly here).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<Record<string, unknown>>(
    "next/navigation",
  );
  return {
    ...actual,
    usePathname: () => "/dashboard",
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      back: vi.fn(),
      refresh: vi.fn(),
      prefetch: vi.fn(),
    }),
  };
});

import DashboardPage from "@/app/(app)/dashboard/page";
import FilingsPage from "@/app/(app)/filings/page";
import IngestPage from "@/app/(app)/ingest/page";

const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("DashboardPage", () => {
  it("loads /api/admin/status/ and renders the snapshot", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          version: "1.0.0",
          deployment_profile: "team",
          embedding_provider: "openai",
          embedding_model: "text-embedding-3-small",
          storage_filings: 1234,
          is_admin: true,
          persist_provider_credentials: true,
        }),
        { status: 200 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<DashboardPage />);
    await waitFor(() => {
      expect(screen.getByText("1,234")).toBeInTheDocument();
    });
    const [url] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/admin/status/");
    expect(screen.getByText("openai")).toBeInTheDocument();
    expect(screen.getByText("text-embedding-3-small")).toBeInTheDocument();
  });

  it("surfaces an error envelope without leaking response internals", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          error: "unauthorised",
          message: "API key was not accepted.",
        }),
        { status: 401 },
      );
    }) as unknown as typeof fetch;

    render(<DashboardPage />);
    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        /API key was not accepted/,
      );
    });
  });

  it("renders the EDGAR identity card", () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response("{}", { status: 200 });
    }) as unknown as typeof fetch;
    render(<DashboardPage />);
    expect(screen.getByText(/EDGAR identity/i)).toBeInTheDocument();
  });
});

describe("FilingsPage", () => {
  it("renders the filings list from /api/admin/filings/", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          total: 1,
          filings: [
            {
              ticker: "AAPL",
              form_type: "10-K",
              filing_date: "2024-09-30",
              accession_number: "0000320193-23-000077",
              chunk_count: 142,
              ingested_at: "2024-10-01T12:34:56Z",
            },
          ],
        }),
        { status: 200 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<FilingsPage />);
    await waitFor(() => {
      expect(screen.getByText("AAPL")).toBeInTheDocument();
    });
    expect(screen.getByText("0000320193-23-000077")).toBeInTheDocument();
    const [url] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    // Always paginated through the proxy — never the unbounded list
    // (OPTIMIZATIONS.md F17; one extra row detects a next page).
    expect(url).toBe("/api/admin/filings/?limit=101&offset=0");
  });

  it("opens a confirmation dialog before deleting", async () => {
    let call = 0;
    const fetchMock = vi.fn(async () => {
      call += 1;
      if (call === 1) {
        return new Response(
          JSON.stringify({
            total: 1,
            filings: [
              {
                ticker: "AAPL",
                form_type: "10-K",
                filing_date: "2024-09-30",
                accession_number: "0000320193-23-000077",
                chunk_count: 142,
                ingested_at: "2024-10-01T12:34:56Z",
              },
            ],
          }),
          { status: 200 },
        );
      }
      // After delete: empty list.
      return new Response(
        JSON.stringify({ total: 0, filings: [] }),
        { status: 200 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<FilingsPage />);
    await screen.findByText("AAPL");

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /^Delete$/i }));
    // Dialog appears.
    expect(
      screen.getByRole("dialog", { name: /confirm filing deletion/i }),
    ).toBeInTheDocument();

    // Cancel: no delete is performed.
    await user.click(screen.getByRole("button", { name: /cancel/i }));
    expect(
      screen.queryByRole("dialog", { name: /confirm filing deletion/i }),
    ).not.toBeInTheDocument();
  });
});

describe("IngestPage", () => {
  it("submits to /api/admin/ingest/add for a single ticker", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          task_id: "t-1",
          status: "pending",
          websocket_url: "/ws/ingest/t-1",
        }),
        { status: 202 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<IngestPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/tickers/i), "AAPL");
    await user.click(screen.getByRole("button", { name: /submit ingest/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("/api/admin/ingest/add");
    const body = JSON.parse(init.body as string) as {
      tickers: string[];
      form_types: string[];
    };
    expect(body.tickers).toEqual(["AAPL"]);
  });

  it("submits to /api/admin/ingest/batch for multiple tickers", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          task_id: "t-2",
          status: "pending",
          websocket_url: "/ws/ingest/t-2",
        }),
        { status: 202 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<IngestPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/tickers/i), "AAPL, MSFT");
    await user.click(screen.getByRole("button", { name: /submit ingest/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });
    const [url] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/admin/ingest/batch");
  });

  it("surfaces submission errors without echoing user input", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          error: "edgar_identity_required",
          message: "Register an EDGAR identity before ingesting.",
        }),
        { status: 401 },
      );
    }) as unknown as typeof fetch;

    render(<IngestPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/tickers/i), "WEIRDVALUE");
    await user.click(screen.getByRole("button", { name: /submit ingest/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Register an EDGAR identity/);
    expect(alert.textContent ?? "").not.toContain("WEIRDVALUE");
  });
});
