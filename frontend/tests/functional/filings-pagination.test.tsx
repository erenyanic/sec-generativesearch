// Filings page pagination (functional, OPTIMIZATIONS.md F17).
//
// The backend returns the whole `DB_MAX_FILINGS`-bounded registry when
// `limit` is omitted (10 000 rows in B/C), so the page must always ask
// for one page. Pins:
//
//   - every list request carries `limit=101` (a 100-row page + one probe
//     row) and a page-aligned `offset`;
//   - Previous / Next walk the pages; Next is disabled once the probe row
//     is absent;
//   - applying filters restarts from the first page;
//   - deleting the last row of a page steps back a page instead of
//     showing an empty one;
//   - the heading never presents the response's `total` (a page size when
//     paginating) as the corpus size.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import FilingsPage from "@/app/(app)/filings/page";

const originalFetch = globalThis.fetch;

function filing(index: number): Record<string, unknown> {
  return {
    ticker: "AAPL",
    form_type: "10-K",
    filing_date: "2024-09-30",
    accession_number: `0000320193-24-${index.toString().padStart(6, "0")}`,
    chunk_count: 10,
    ingested_at: "2024-10-01T12:34:56Z",
  };
}

function page(count: number, start = 0, total = count): Response {
  const filings = Array.from({ length: count }, (_, i) => filing(start + i));
  return new Response(JSON.stringify({ total, filings }), { status: 200 });
}

function listQuery(fetchMock: ReturnType<typeof vi.fn>, call: number): URLSearchParams {
  const [url] = fetchMock.mock.calls[call] as unknown as [string, RequestInit];
  expect(url.startsWith("/api/admin/filings/?")).toBe(true);
  return new URLSearchParams(url.slice(url.indexOf("?") + 1));
}

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("FilingsPage pagination", () => {
  it("shows an exact count when the result set fits one page", async () => {
    // A server `total` that disagrees with the rows must not leak into the heading.
    const fetchMock = vi.fn(async () => page(3, 0, 9999));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<FilingsPage />);
    expect(await screen.findByRole("heading", { name: "3 filings" })).toBeInTheDocument();
    expect(screen.queryByText(/9,?999/)).toBeNull();
    expect(screen.queryByRole("navigation", { name: /filings pages/i })).toBeNull();
    const query = listQuery(fetchMock, 0);
    expect(query.get("limit")).toBe("101");
    expect(query.get("offset")).toBe("0");
  });

  it("walks pages with Next / Previous using page-aligned offsets", async () => {
    const user = userEvent.setup();
    const responses = [page(101, 0), page(5, 100), page(101, 0)];
    const fetchMock = vi.fn(async () => responses.shift() ?? page(0));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<FilingsPage />);
    expect(await screen.findByRole("heading", { name: "Showing 1–100" })).toBeInTheDocument();
    // The probe row is never rendered.
    expect(screen.getAllByRole("button", { name: /^Delete$/i })).toHaveLength(100);
    const nav = screen.getByRole("navigation", { name: /filings pages/i });
    expect(within(nav).getByRole("button", { name: "Previous" })).toBeDisabled();

    await user.click(within(nav).getByRole("button", { name: "Next" }));
    expect(await screen.findByRole("heading", { name: "Showing 101–105" })).toBeInTheDocument();
    expect(listQuery(fetchMock, 1).get("offset")).toBe("100");
    expect(within(nav).getByRole("button", { name: "Next" })).toBeDisabled();

    await user.click(within(nav).getByRole("button", { name: "Previous" }));
    expect(await screen.findByRole("heading", { name: "Showing 1–100" })).toBeInTheDocument();
    expect(listQuery(fetchMock, 2).get("offset")).toBe("0");
  });

  it("restarts from the first page when filters are applied", async () => {
    const user = userEvent.setup();
    const responses = [page(101, 0), page(101, 100), page(2, 0)];
    const fetchMock = vi.fn(async () => responses.shift() ?? page(0));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<FilingsPage />);
    await screen.findByRole("heading", { name: "Showing 1–100" });
    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("heading", { name: "Showing 101–200" });

    await user.type(screen.getByLabelText("Ticker"), "msft");
    await user.click(screen.getByRole("button", { name: /apply filters/i }));
    expect(await screen.findByRole("heading", { name: "2 filings" })).toBeInTheDocument();
    const query = listQuery(fetchMock, 2);
    expect(query.get("ticker")).toBe("MSFT");
    expect(query.get("offset")).toBe("0");
    expect(query.get("limit")).toBe("101");
  });

  it("steps back a page when a delete empties the last page", async () => {
    const user = userEvent.setup();
    const responses: Response[] = [
      page(101, 0),
      page(1, 100),
      new Response(JSON.stringify({ accession_number: "x", chunks_deleted: 10 }), {
        status: 200,
      }),
      page(0),
      page(100, 0),
    ];
    const fetchMock = vi.fn(async () => responses.shift() ?? page(0));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<FilingsPage />);
    await screen.findByRole("heading", { name: "Showing 1–100" });
    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("heading", { name: "Showing 101–101" });

    await user.click(screen.getByRole("button", { name: /^Delete$/i }));
    const dialog = screen.getByRole("dialog", { name: /confirm filing deletion/i });
    await user.click(within(dialog).getByRole("button", { name: /^Delete$/i }));

    // The emptied page 2 is replaced by page 1, never rendered empty.
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(5);
    });
    expect(listQuery(fetchMock, 3).get("offset")).toBe("100");
    expect(listQuery(fetchMock, 4).get("offset")).toBe("0");
    expect(await screen.findByRole("heading", { name: "100 filings" })).toBeInTheDocument();
    expect(screen.queryByText(/no filings match/i)).toBeNull();
  });
});
