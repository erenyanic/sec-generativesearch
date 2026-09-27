// Chat page render cost while a turn streams (functional, OPTIMIZATIONS.md F30).
//
// Every batched `STREAM_DELTA` flush re-renders the page. Pins:
//
//   - a committed turn's card is memoised — its answer does not re-render
//     while a later turn streams (per-frame work stays O(1) in committed
//     turns instead of O(turns));
//   - the pending turn's citation map keeps its identity across deltas
//     (memo keyed on `state.citations`, not the per-delta `state`).
//
// Observed through the `./answer-parts` module boundary: `AnswerBody` is
// wrapped with a recorder, the real component still renders.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

type AnswerProps = { answer: string; citations: Map<number, unknown>; turnId: string };

const answerRenders = vi.hoisted(() => [] as AnswerProps[]);

vi.mock("@/app/(app)/chat/answer-parts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/app/(app)/chat/answer-parts")>();
  return {
    ...actual,
    AnswerBody: (props: Parameters<typeof actual.AnswerBody>[0]) => {
      answerRenders.push(props as AnswerProps);
      return actual.AnswerBody(props);
    },
  };
});

vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<Record<string, unknown>>("next/navigation");
  return {
    ...actual,
    usePathname: () => "/chat",
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      back: vi.fn(),
      refresh: vi.fn(),
      prefetch: vi.fn(),
    }),
  };
});

import ChatPage from "@/app/(app)/chat/page";
import {
  emptyProvidersResponse,
  finalFrame,
  pendingStreamingResponse,
  sseFrame,
  streamingResponse,
} from "../security/_sse-harness";

const originalFetch = globalThis.fetch;

const SAMPLE_PLAN = {
  raw_query: "Apple AI risk",
  detected_language: "en",
  query_en: "Apple AI risk",
  tickers: ["AAPL"],
  form_types: ["10-K"],
  date_range: null,
  intent: "lookup",
  suggested_answer_mode: "concise",
};

beforeEach(() => {
  answerRenders.length = 0;
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("ChatPage — render cost while streaming (F30)", () => {
  it("does not re-render committed turns, and keeps the pending citation map, per delta", async () => {
    const first = "First committed answer";
    const pending = pendingStreamingResponse();
    let streams = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      if (url === "/api/admin/rag/plan") {
        return new Response(
          JSON.stringify({ plan: SAMPLE_PLAN, provider: "openai", model: "gpt-test" }),
          { status: 200 },
        );
      }
      streams += 1;
      return streams === 1
        ? streamingResponse([sseFrame("delta", { text: first }), finalFrame(first)])
        : pending.response;
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ChatPage />);
    const user = userEvent.setup();

    // Turn 1 commits.
    await user.type(screen.getByLabelText(/your message/i), "first question");
    await user.click(screen.getByRole("button", { name: /^Send$/i }));
    await screen.findByRole("listitem", { name: "Turn 1" });
    const committedRenders = () => answerRenders.filter((r) => r.turnId !== "pending").length;
    await waitFor(() => {
      expect(committedRenders()).toBeGreaterThan(0);
    });
    const baseline = committedRenders();

    // Turn 2 streams five separately flushed deltas.
    await user.type(screen.getByLabelText(/your message/i), "second question");
    await user.click(screen.getByRole("button", { name: /^Send$/i }));
    let streamed = "";
    for (const word of ["Second ", "answer ", "arrives ", "in ", "pieces"]) {
      streamed += word;
      pending.release([sseFrame("delta", { text: word })]);
      await waitFor(() => {
        expect(screen.getByText(streamed.trimEnd())).toBeInTheDocument();
      });
    }

    // Committed card: zero extra renders across the whole stream.
    expect(committedRenders()).toBe(baseline);

    // Pending card rendered once per flush, always with the same map.
    const pendingRenders = answerRenders.filter((r) => r.turnId === "pending");
    expect(pendingRenders.length).toBeGreaterThanOrEqual(5);
    expect(new Set(pendingRenders.map((r) => r.citations)).size).toBe(1);

    // Let the second stream finish so React can unmount cleanly.
    pending.release([finalFrame(streamed)]);
  });
});
