# Embedding Models — Reference

> **Purpose.** A single place to look when choosing an embedding model **at initial
> system setup**. The embedding model is an *administrative* decision: it is chosen
> once, stamped onto the ChromaDB collection at creation, and is **not** a user-facing
> toggle (unlike the LLM, which a user may switch per query). Changing it later means a
> full re-index. This file is **informational only** — it is **not** read by any code.
> The load-bearing source of truth in the codebase is each adapter's
> `MODEL_DIMENSIONS` ClassVar (and `ProviderRegistry.get_dimension`).
>
> **Why a doc and not a third-party-sourced catalogue (like the LLM list).** The LLM
> catalogue is auto-refreshable because it is a large (~105 rows), frequently-changing,
> multi-field, *hosted* dataset whose cost is cosmetic. Embedding metadata is the
> opposite: ~15 stable single-integer dimensions, half of them **local/on-device** or
> **Chinese-native** models that no third-party framework lists, and the one field that
> matters — the **dimension** — is *correctness-critical* (it locks the collection and
> the embedder stamp). The third-party lists are also demonstrably unreliable for it:
> at the time of writing models.dev reported `mistral-embed` as **3072** (real: 1024)
> and `gemini-embedding-001` as **1**. Sourcing the dimension from such a list would be
> strictly worse than hardcoding it. So this stays hand-maintained, and this doc is the
> human-readable companion.

## How to read these tables

- **Dim** is the *native default* output dimension — authoritative, matching the code.
  Models marked *(MRL)* additionally support requesting a *smaller* dimension at call
  time (Matryoshka); v1 always uses the native default for collection stability.
- **~Price** is an *approximate* hosted cost in **USD per 1M input tokens** (embeddings
  bill input only). Informational — verify against the vendor's pricing page before
  relying on it. `—` = not published by the third-party lists.
- Last reviewed: **2026-06-27** (dims cross-checked against models.dev / LiteLLM and the
  in-repo adapters).

## Hosted providers

| Provider | Model | Dim | Max input | ~Price /1M tok | Notes |
| -------- | ----- | --: | --------: | -------------: | ----- |
| **openai** | `text-embedding-3-small` | 1536 *(MRL)* | 8191 | 0.02 | Default for the provider. |
| **openai** | `text-embedding-3-large` | 3072 *(MRL)* | 8191 | 0.13 | Highest quality. |
| **openai** | `text-embedding-ada-002` | 1536 | 8191 | 0.10 | Legacy; kept for back-compat. |
| **gemini** | `gemini-embedding-2` | 3072 *(MRL)* | 8192 | ~0.20 | Provider default (GA). |
| **gemini** | `gemini-embedding-2-preview` | 3072 *(MRL)* | 8192 | ~0.20 | Superseded preview; kept so a collection already stamped against it still opens. |
| **gemini** | `gemini-embedding-001` | 3072 *(MRL)* | 2048 | 0.15 | GA. |
| **mistral** | `mistral-embed` | 1024 | 8192 | 0.10 | Previous provider default; still selectable. |
| **mistral** | `codestral-embed` | 1536 *(MRL)* | 8192 | 0.15 | Provider default. Code-tuned; supports larger dims via MRL. |
| **qwen** (DashScope) | `qwen3.7-text-embedding` | 1024 | — | — | Provider default. **Not in models.dev / LiteLLM — price and max input from the Alibaba Cloud DashScope console; the 1024 dimension is not yet verified against the live API.** |
| **qwen** (DashScope) | `text-embedding-v4` | 1024 *(MRL)* | ~8192 | — | Previous provider default; still selectable. See note above; confirm v4's default dimension. |
| **qwen** (DashScope) | `text-embedding-v3` | 1024 *(MRL)* | ~8192 | — | See note above. |

## Local / on-device (`local` provider — `sentence-transformers`)

> Free (no API cost); runs on the host, so **no query or filing text leaves the
> machine** — this is the project default and the privacy-preserving choice. Dimensions
> are fixed properties of each model and never change. Not present in any hosted-pricing
> list (and should not be).

| Model | Dim | Notes |
| ----- | --: | ----- |
| `google/embeddinggemma-300m` | 768 *(MRL)* | **Project default embedder.** |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Small, fast. |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Higher quality than MiniLM. |
| `BAAI/bge-small-en-v1.5` | 384 | |
| `BAAI/bge-base-en-v1.5` | 768 | |
| `BAAI/bge-large-en-v1.5` | 1024 | Highest quality of the BGE set. |

## Selecting the embedder

Set at setup via environment:

- `EMBEDDING_PROVIDER` — one of the providers above (`local` by default).
- `EMBEDDING_MODEL_NAME` — a model from that provider's row set.

The choice is stamped onto the `sec_filings` collection at creation. To change it on an
existing store, re-index (`sec-rag manage reindex` / `ReindexService`) — the dimension
is part of the collection seal, so a mismatched model is refused rather than silently
mixed.

## Sources & caveats

- **models.dev** (`https://models.dev/api.json`) and **LiteLLM**
  (`model_prices_and_context_window.json`) — the project's pinned LLM-refresh upstreams;
  used here only as a cross-check for the Western providers.
- Neither lists the **native** DashScope (`qwen`) embedding models, nor any
  Zhipu/MiniMax/Moonshot/DeepSeek embedding API — only open-weights `qwen3-embedding`
  re-hosted by aggregators, which is **not** what the `qwen` adapter calls.
- Both lists carried **wrong dimensions** for some models (see the note at the top) — do
  not treat them as authoritative for the dimension. The adapters' `MODEL_DIMENSIONS`
  are the source of truth.
