"use client";

// Filings — list, filter, delete.
//
// Read tier: `GET /api/filings/` (auth-gated; optional `ticker` /
// `form_type` query). Destructive tier: `DELETE /api/filings/{accession}`
// (admin-gated; needs the admin key the operator submitted at login).
//
// Paginated (OPTIMIZATIONS.md F17): the backend returns the whole
// `DB_MAX_FILINGS`-bounded registry when `limit` is omitted (10 000 rows
// in B/C), so this page always sends `limit` + `offset`. The server
// resolves sort + page in SQL with a stable tie-breaker, so pages never
// skip or repeat a row.
//
// We refresh after every deletion so the displayed page stays
// consistent with the registry. Delete is confirmation-gated client-side
// because a one-click destructive surface is a footgun.

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useState,
  type FormEvent,
  type JSX,
} from "react";

import { Skeleton } from "@/components/skeleton";
import { ApiError, deleteFiling, listFilings } from "@/lib/api";
import type { FilingListResponse, FilingSchema } from "@/lib/api-types";

// Rows per page. Each request asks for one extra row so "is there a next
// page?" is known exactly: the response's `total` counts the rows
// *returned* (a page size when paginating), never the registry, so it
// cannot answer that — and must never be shown as the corpus size.
const FILINGS_PAGE_SIZE = 100;

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; rows: FilingSchema[]; hasNext: boolean }
  | { kind: "error"; message: string };

interface Filters {
  ticker: string;
  form_type: string;
}

export default function FilingsPage(): JSX.Element {
  const [filters, setFilters] = useState<Filters>({ ticker: "", form_type: "" });
  const [applied, setApplied] = useState<Filters>({ ticker: "", form_type: "" });
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  // Zero-based page index into the applied filters' result set.
  const [page, setPage] = useState(0);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const tickerId = useId();
  const formTypeId = useId();

  // Bump on every action (delete) that needs a refresh. The effect below
  // refetches whenever `refreshTick` or the applied filters change.
  const [refreshTick, setRefreshTick] = useState(0);
  const reload = useCallback(() => {
    setRefreshTick((n) => n + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const data: FilingListResponse = await listFilings({
          ticker: applied.ticker !== "" ? applied.ticker : undefined,
          form_type: applied.form_type !== "" ? applied.form_type : undefined,
          limit: FILINGS_PAGE_SIZE + 1,
          offset: page * FILINGS_PAGE_SIZE,
        });
        if (cancelled) {
          return;
        }
        if (data.filings.length === 0 && page > 0) {
          // A delete emptied the last page — step back one page.
          setPage(page - 1);
          return;
        }
        setState({
          kind: "ready",
          rows: data.filings.slice(0, FILINGS_PAGE_SIZE),
          hasNext: data.filings.length > FILINGS_PAGE_SIZE,
        });
      } catch (exc) {
        if (cancelled) {
          return;
        }
        const message =
          exc instanceof ApiError ? exc.message : "Could not load filings.";
        setState({ kind: "error", message });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [applied.form_type, applied.ticker, page, refreshTick]);

  const handleFilterSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      setApplied({
        ticker: filters.ticker.trim().toUpperCase(),
        form_type: filters.form_type.trim().toUpperCase(),
      });
      // New filters → new result set; start from its first page.
      setPage(0);
    },
    [filters.form_type, filters.ticker],
  );

  const handleConfirmDelete = useCallback(async () => {
    if (pendingDelete === null || deleting) {
      return;
    }
    setDeleting(true);
    setActionError(null);
    try {
      await deleteFiling(pendingDelete);
      setPendingDelete(null);
      reload();
    } catch (exc) {
      const message =
        exc instanceof ApiError
          ? `${exc.message}${exc.hint !== undefined ? ` ${exc.hint}` : ""}`
          : "Delete failed.";
      setActionError(message);
    } finally {
      setDeleting(false);
    }
  }, [deleting, pendingDelete, reload]);

  const rows = useMemo<FilingSchema[]>(
    () => (state.kind === "ready" ? state.rows : []),
    [state],
  );
  const hasNext = state.kind === "ready" && state.hasNext;
  const firstRow = page * FILINGS_PAGE_SIZE + 1;
  const lastRow = page * FILINGS_PAGE_SIZE + rows.length;

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Filings</h1>
        <p className="text-sm text-slate-600">
          Read the dual-store registry and remove filings that no longer
          need indexing.
        </p>
      </header>

      <form
        onSubmit={handleFilterSubmit}
        className="flex flex-wrap items-end gap-4 rounded-lg border border-slate-200 bg-white p-4 shadow-sm"
        aria-label="Filing filters"
      >
        <div className="flex-1 min-w-[12rem] space-y-1">
          <label
            htmlFor={tickerId}
            className="block text-xs font-medium uppercase tracking-wide text-slate-500"
          >
            Ticker
          </label>
          <input
            id={tickerId}
            type="text"
            value={filters.ticker}
            onChange={(event) => {
              setFilters({ ...filters, ticker: event.target.value });
            }}
            maxLength={16}
            placeholder="AAPL"
            className="w-full rounded border border-slate-300 px-3 py-1.5 text-sm uppercase focus:border-slate-500 focus:outline-none"
          />
        </div>
        <div className="flex-1 min-w-[12rem] space-y-1">
          <label
            htmlFor={formTypeId}
            className="block text-xs font-medium uppercase tracking-wide text-slate-500"
          >
            Form type
          </label>
          <input
            id={formTypeId}
            type="text"
            value={filters.form_type}
            onChange={(event) => {
              setFilters({ ...filters, form_type: event.target.value });
            }}
            maxLength={16}
            placeholder="10-K"
            className="w-full rounded border border-slate-300 px-3 py-1.5 text-sm uppercase focus:border-slate-500 focus:outline-none"
          />
        </div>
        <button
          type="submit"
          className="rounded bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800"
        >
          Apply filters
        </button>
      </form>

      {actionError !== null ? (
        <p
          className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
          role="alert"
        >
          {actionError}
        </p>
      ) : null}

      <section
        aria-labelledby="filings-table-heading"
        className="rounded-lg border border-slate-200 bg-white shadow"
      >
        <h2
          id="filings-table-heading"
          className="border-b border-slate-200 px-4 py-3 text-sm font-medium text-slate-700"
        >
          {state.kind !== "ready" || rows.length === 0
            ? "Filings"
            : page === 0 && !hasNext
              ? // The whole result set fits on one page: an exact count.
                `${rows.length.toLocaleString()} filing${rows.length === 1 ? "" : "s"}`
              : `Showing ${firstRow.toLocaleString()}–${lastRow.toLocaleString()}`}
        </h2>
        {state.kind === "loading" ? (
          <div className="px-4 py-6">
            <Skeleton rows={5} label="Loading filings…" />
          </div>
        ) : null}
        {state.kind === "error" ? (
          <p
            className="px-4 py-6 text-sm text-red-700"
            role="alert"
          >
            {state.message}
          </p>
        ) : null}
        {state.kind === "ready" && rows.length === 0 ? (
          <p className="px-4 py-6 text-sm text-slate-500">
            No filings match the current filters.
          </p>
        ) : null}
        {state.kind === "ready" && rows.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-2 text-left font-medium text-slate-600">
                    Ticker
                  </th>
                  <th className="px-4 py-2 text-left font-medium text-slate-600">
                    Form
                  </th>
                  <th className="px-4 py-2 text-left font-medium text-slate-600">
                    Filed
                  </th>
                  <th className="px-4 py-2 text-left font-medium text-slate-600">
                    Accession
                  </th>
                  <th className="px-4 py-2 text-right font-medium text-slate-600">
                    Chunks
                  </th>
                  <th className="px-4 py-2 text-right font-medium text-slate-600">
                    {/* delete column */}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {rows.map((row) => (
                  <tr key={row.accession_number}>
                    <td className="px-4 py-2 font-mono text-slate-900">
                      {row.ticker}
                    </td>
                    <td className="px-4 py-2 text-slate-700">{row.form_type}</td>
                    <td className="px-4 py-2 text-slate-700">
                      {row.filing_date}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-slate-700">
                      {row.accession_number}
                    </td>
                    <td className="px-4 py-2 text-right text-slate-700">
                      {row.chunk_count.toLocaleString()}
                    </td>
                    <td className="px-4 py-2 text-right">
                      <button
                        type="button"
                        onClick={() => {
                          setPendingDelete(row.accession_number);
                          setActionError(null);
                        }}
                        className="rounded border border-red-300 px-2 py-1 text-xs text-red-700 hover:bg-red-50"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        {state.kind === "ready" && (page > 0 || hasNext) ? (
          <nav
            aria-label="Filings pages"
            className="flex items-center justify-between border-t border-slate-200 px-4 py-3 text-sm"
          >
            <button
              type="button"
              onClick={() => {
                setPage((n) => Math.max(0, n - 1));
              }}
              disabled={page === 0}
              className="rounded border border-slate-300 px-3 py-1 text-slate-700 hover:bg-slate-100 disabled:opacity-50"
            >
              Previous
            </button>
            <span className="text-slate-600">Page {(page + 1).toLocaleString()}</span>
            <button
              type="button"
              onClick={() => {
                setPage((n) => n + 1);
              }}
              disabled={!hasNext}
              className="rounded border border-slate-300 px-3 py-1 text-slate-700 hover:bg-slate-100 disabled:opacity-50"
            >
              Next
            </button>
          </nav>
        ) : null}
      </section>

      {pendingDelete !== null ? (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Confirm filing deletion"
          className="fixed inset-0 z-10 flex items-center justify-center bg-slate-900/60 px-4"
        >
          <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
            <h2 className="text-lg font-semibold tracking-tight">
              Delete filing?
            </h2>
            <p className="mt-2 text-sm text-slate-600">
              The filing{" "}
              <span className="font-mono text-slate-900">{pendingDelete}</span>{" "}
              will be removed from both ChromaDB and the metadata registry.
              This action cannot be undone.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setPendingDelete(null)}
                disabled={deleting}
                className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  void handleConfirmDelete();
                }}
                disabled={deleting}
                className="rounded bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-500 disabled:opacity-50"
              >
                {deleting ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
