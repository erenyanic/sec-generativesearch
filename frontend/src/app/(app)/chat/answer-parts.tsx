// Answer rendering parts for the chat page: the plain-text answer with
// inline `[N]` citation chips, and the turn-scoped source panel.
//
// A sibling module (like `./reducer.ts`) rather than local functions in
// `page.tsx`: a Next page file may only export its route surface, and this
// boundary lets tests observe how often a committed turn's answer
// re-renders while a later turn streams (OPTIMIZATIONS.md F30).

import { useMemo, type JSX } from "react";

import type { CitationSchema } from "@/lib/api-types";

// Inline `[N]` markers turn into clickable chips anchored at the turn-
// scoped source panel below. Plain-text rendering only — same posture
// as the Ask page (no Markdown / HTML sink).
export function AnswerBody({
  answer,
  citations,
  turnId,
}: {
  answer: string;
  citations: Map<number, CitationSchema>;
  turnId: string;
}): JSX.Element {
  const segments = useMemo(() => splitAnswerByCitation(answer), [answer]);
  return (
    <p className="mt-1 whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
      {segments.map((segment, idx) => {
        if (segment.kind === "text") {
          return <span key={idx}>{segment.text}</span>;
        }
        const citation = citations.get(segment.index);
        const label = `[${segment.index.toString()}]`;
        const title =
          citation !== undefined
            ? `${citation.ticker} ${citation.form_type} ${citation.filing_date}`
            : "unmatched citation";
        return (
          <a
            key={idx}
            href={`#citation-${turnId}-${segment.index.toString()}`}
            title={title}
            className="mx-0.5 rounded bg-slate-100 px-1 py-0.5 font-mono text-xs text-slate-700 hover:bg-slate-200"
          >
            {label}
          </a>
        );
      })}
    </p>
  );
}

export function SourcePanel({
  citations,
  turnId,
}: {
  citations: CitationSchema[];
  turnId: string;
}): JSX.Element {
  return (
    <details className="mt-3 rounded border border-slate-200 bg-slate-50">
      <summary className="cursor-pointer px-3 py-2 text-xs font-medium text-slate-700">
        Sources ({citations.length.toString()})
      </summary>
      <ol className="space-y-2 px-3 pb-3">
        {citations.map((citation, idx) => (
          <li
            key={`${citation.chunk_id}-${idx.toString()}`}
            id={`citation-${turnId}-${citation.display_index.toString()}`}
            className="rounded border border-slate-200 bg-white p-2 text-xs text-slate-700"
          >
            <p className="font-mono">
              [{citation.display_index.toString()}] {citation.ticker}{" "}
              {citation.form_type} {citation.filing_date}{" "}
              <span className="text-slate-500">
                {citation.accession_number}
              </span>
            </p>
            <p className="mt-1 text-slate-500">{citation.section_path}</p>
            <p className="mt-1 whitespace-pre-wrap text-slate-800">
              {citation.text_span}
            </p>
          </li>
        ))}
      </ol>
    </details>
  );
}

type Segment =
  | { kind: "text"; text: string }
  | { kind: "citation"; index: number };

const CITATION_RE = /\[(\d+)\]/g;

function splitAnswerByCitation(answer: string): Segment[] {
  const out: Segment[] = [];
  let cursor = 0;
  CITATION_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = CITATION_RE.exec(answer)) !== null) {
    if (match.index > cursor) {
      out.push({ kind: "text", text: answer.slice(cursor, match.index) });
    }
    const indexNum = Number.parseInt(match[1] ?? "0", 10);
    if (Number.isFinite(indexNum) && indexNum > 0) {
      out.push({ kind: "citation", index: indexNum });
    } else {
      out.push({ kind: "text", text: match[0] });
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < answer.length) {
    out.push({ kind: "text", text: answer.slice(cursor) });
  }
  return out;
}
