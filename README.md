# SEC-GenerativeSearch

A **security-first Retrieval-Augmented Generation (RAG) system** for SEC filings (10-K, 10-Q, 8-K, and their amended variants). It fetches filings from EDGAR, embeds them locally, retrieves the passages most relevant to your question, and then asks a large language model to write a grounded answer with inline citations back to the source text.

This is a **generative** system: a language model writes the answer. That answer is assembled from the actual filing excerpts the retriever found, and every claim carries a `[N]` citation you can trace back to a specific chunk — but the wording is generated, not quoted verbatim.

> Looking for pure vector similarity search with **no** language model in the loop? That is a different project — [SEC-SemanticSearch](https://github.com/ErenYanic/SEC-SemanticSearch) — which returns raw filing excerpts and never calls an LLM.

---

## Where your data goes (read this first)

Because an LLM generates the answer, **this system is not fully local by default.** Where each step sends your data:

- **Generation goes to an LLM — remote by default.** To answer a question, the system sends your question text **and the retrieved filing excerpts** to whichever LLM provider you configure. By default that is a remote bring-your-own-key vendor (OpenAI, Anthropic, Gemini, Mistral, DeepSeek, Grok, Qwen, Kimi, MiniMax, MiMo, Z.ai, or OpenRouter), and **that provider sees your prompts**. A **self-hosted `local_llm`** option (Ollama / llama.cpp-server / vLLM / LM Studio) lets you point generation at a model server **you** run — loopback by default — so the prompt need not reach any third party. Even then the prompt is still transmitted over the wire to that endpoint: *"local" means a server you control, not that nothing ever leaves the process.*
- **Retrieval is local by default.** The default embedder (`google/embeddinggemma-300m`, 768-dim) runs on your own machine, so the *retrieval* step — embedding your query and the filing chunks — does not call out to a third party. That keeps the LLM provider as the only external service in the default configuration. It does not by itself make the pipeline private: with a remote LLM the prompt still leaves your machine at generation time. Pairing local retrieval with a loopback `local_llm` endpoint keeps the prompt on your own host.
- **A hosted embedder is opt-in and widens the exposure.** If you set `EMBEDDING_PROVIDER=openai|gemini|mistral|qwen`, every query and every filing chunk is also sent to that embedding API. Only do this when that exposure is acceptable.
- **Filings are public.** The filing text itself is public SEC data. What is sensitive is the *pattern* of your activity — which companies, when, and what you asked — which is exactly what reaches the LLM provider.

What the project protects: your search queries and chat history are never persisted to disk, provider API keys are encrypted client-side (the server never sees your password or key-encryption key), and the database can be encrypted at rest. See [Security design](#security-design) below.

---

## How it works

```text
Fetch (edgartools) → Parse (doc2dict) → Chunk (sentence splitter)
        ↓                    ↓                    ↓
  FilingIdentifier      list[Segment]        list[Chunk]
        ↓
Embed (local model)  →  Store (ChromaDB + SQLite)  →  Retrieve  →  Generate (LLM provider)
        ↓                        ↓                        ↓              ↓
 np.ndarray(768)        sec_filings collection      RetrievalResult   GenerationResult
                        + metadata registry         + Citation[]      + Citation[]
```

Filings are fetched from SEC EDGAR, parsed into structured sections, split into chunks at sentence boundaries (chunks never cross section boundaries, which keeps citations accurate), and embedded locally into 768-dimensional vectors stored in ChromaDB. A SQLite registry tracks metadata, duplicates, and retention. At query time the question is embedded with the same model, the most relevant chunks are retrieved, and those chunks — wrapped in untrusted-content delimiters — are handed to your chosen LLM, which writes the answer and cites the chunks it used.

---

## Features

- **Grounded answers with citations** — answers are generated only from retrieved filing chunks; every `[N]` marker resolves to a specific chunk, and citations pointing at chunks that were not retrieved are dropped, never fabricated.
- **Bring-your-own-key (BYOK)** — twelve hosted providers via their first-party SDKs (no aggregator middleware), plus a keyless self-hosted **`local_llm`** endpoint (Ollama / llama.cpp-server / vLLM / LM Studio); supply your own key — or none, for `local_llm` — per session, per user, or per operator.
- **Local-by-default retrieval** — the on-device embedder keeps the retrieval step off the network, so the LLM provider is the *only* third party in the default configuration (and a loopback `local_llm` endpoint removes even that).
- **Streaming chat** — interactive RAG chat over a corpus with cancellable, citation-aware streaming in both CLI and web UI; prior-turn filing chunks never re-enter a later prompt.
- **Three deployment profiles** — Local (single user), Team (shared server), Cloud (GCP Cloud Run, internet-facing), each with hardened defaults.
- **Full web UI** — Next.js 16 + React 19 SPA with strict per-request CSP nonce, Trusted Types, and a per-user encrypted vault for provider keys.
- **Security-first throughout** — two-tier API keys, per-session EDGAR identity, SQLCipher at rest, sliding-TTL rate limits, content-free logs and metrics, and a large dedicated security-regression suite.

---

## Requirements

- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/) package manager (recommended)
- Node 22 LTS + pnpm 11.2.2 (web frontend only)
- `libsqlcipher-dev` system package (only for the optional `[encryption]` extra)
- A CUDA-capable NVIDIA GPU is recommended for the local embedder (CPU works, slower; BF16 is applied automatically on CUDA)

---

## Installation

```bash
# Clone
git clone https://github.com/ErenYanic/SEC-GenerativeSearch.git
cd SEC-GenerativeSearch

# Create and activate a virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install with the extras you need (local embedder is in [local-embeddings])
uv pip install -e ".[dev,test,local-embeddings]"
```

### Optional extras

| Extra                | Installs                         | When to use                                                          |
| -------------------- | -------------------------------- | -------------------------------------------------------------------- |
| `[local-embeddings]` | `sentence-transformers`, `torch` | The default on-device embedder. Needed for the local retrieval path. |
| `[encryption]`       | `pysqlcipher3`                   | SQLCipher encryption at rest (Scenarios B/C; needs the system lib).  |
| `[metrics]`          | `prometheus-client`              | The admin-gated `GET /api/metrics` OpenMetrics endpoint.             |

```bash
# Everything
uv pip install -e ".[dev,test,encryption,local-embeddings,metrics]"
```

---

## Configuration

Copy the template and fill in the essentials:

```bash
cp .env.example .env
```

The SEC requires an identifying name and email in the User-Agent of every EDGAR request, and you must point the system at an LLM provider for generation to work.

**Minimum for local CLI use:**

| Variable               | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `EDGAR_IDENTITY_NAME`  | Your name (sent to SEC EDGAR)                           |
| `EDGAR_IDENTITY_EMAIL` | Your email (sent to SEC EDGAR)                          |
| `LLM_PROVIDER`         | The provider used to generate answers, e.g. `anthropic` |
| `<PROVIDER>_API_KEY`   | The matching key, e.g. `ANTHROPIC_API_KEY`              |

**Commonly adjusted optional variables:**

| Variable                      | Default | Description                                                                                             |
| ----------------------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `EMBEDDING_PROVIDER`          | `local` | `local` keeps retrieval on-device; `openai`/`gemini`/`mistral`/`qwen` send queries + chunks to that API |
| `DB_DEPLOYMENT_PROFILE`       | `local` | `local`, `team`, or `cloud` — fills in hardened per-profile defaults                                    |
| `DB_ENCRYPTION_KEY` / `_FILE` | unset   | SQLCipher key (or a path to a secret file); unset = plain SQLite                                        |
| `DB_MAX_FILINGS`              | `2500`  | Corpus ceiling                                                                                          |
| `API_KEY` / `API_ADMIN_KEY`   | unset   | Read-tier and admin-tier API keys; unset = auth disabled (local only)                                   |
| `API_EDGAR_SESSION_REQUIRED`  | `false` | Require per-session EDGAR identity (recommended for shared servers)                                     |
| `LOG_REDACT_QUERIES`          | `false` | Hash query text, tickers and accession numbers before they reach the logs                               |

A `.env` for local use is as short as:

```env
EDGAR_IDENTITY_NAME=Your Name
EDGAR_IDENTITY_EMAIL=your@email.com
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
DB_DEPLOYMENT_PROFILE=local
```

See [`.env.example`](.env.example) for every variable with descriptions.

---

## Quick start

### 1. Start the API

```bash
uvicorn sec_generative_search.api.app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

### 2. Ingest a filing and ask a question (CLI)

```bash
sec-rag ingest add AAPL --form 10-K --count 1
sec-rag rag query "What were Apple's main risk factors?"
```

### 3. (Optional) Start the web UI

```bash
cd frontend
corepack enable && corepack prepare --activate   # one-off
pnpm install
pnpm dev                                          # webpack dev server on :3000
```

Open `http://localhost:3000`. The SPA prompts for your API/admin keys via `WelcomeGate`.

---

## CLI

The `sec-rag` command covers every operator workflow.

```bash
# Ingest filings
sec-rag ingest add AAPL --form 10-K --count 3
sec-rag ingest batch tickers.txt --form 10-Q

# Manage the corpus
sec-rag manage list
sec-rag manage status
sec-rag manage remove AAPL
sec-rag manage clear -y          # always proceeds; not subject to API_DEMO_MODE

# Semantic search (retrieval only, no LLM call)
sec-rag search "capital expenditure trends" --ticker AAPL

# RAG — one-shot generation
sec-rag rag query "What were Apple's main risk factors in 2024?"

# RAG — interactive chat (cancellable streaming; cancelled turns are not kept)
sec-rag rag chat

# Provider management (keys are prompted, never passed as a flag)
sec-rag provider list
sec-rag provider validate anthropic
sec-rag provider set anthropic

# Evaluation, reindex, backup
sec-rag evaluate retrieval --cases tests/fixtures/retrieval_eval_cases.json
sec-rag manage reindex            # re-embed the corpus under a new model
sec-rag backup create             # tarball SQLite + ChromaDB
```

Add `--output json` to `search`, `rag query`, `manage list`, `manage status`, `provider list`, `provider validate`, and `evaluate retrieval` for machine-readable output. Evaluation and search JSON are content-free: case IDs and numeric metrics only, never query, chunk, or answer text.

> **The CLI is an operator-on-host tool.** It bypasses every API control (auth keys, rate limits, per-session EDGAR identity, log redaction). Do not hand CLI access to team users on a shared or public deployment.

---

## Web application

A full SPA built with FastAPI (backend) and Next.js 16 + React 19 (frontend).

- **Dashboard** — corpus overview: filing count, chunk count, form-type breakdown, ticker table.
- **Search** — retrieval-only semantic search with filters and expandable, copyable results.
- **Ingest** — tag-style ticker input, real-time WebSocket progress per filing, cancel and recovery.
- **RAG / Chat** — two-step *plan → answer* flow with citation chips and cancellable streaming.
- **Provider settings** — manage your own LLM keys, encrypted in a per-user vault the server cannot read.
- **Filings** — sortable, filterable, paginated table with single and bulk delete.

The admin key never reaches the browser: a server-side Next.js proxy injects it, and the browser only ever holds an `HttpOnly`/`Secure`/`SameSite=Strict` session cookie. Full interactive API docs are at `http://localhost:8000/docs` (exposed in the local profile only).

---

## LLM providers

Twelve hosted providers, each via its first-party SDK, plus a self-hosted `local_llm` endpoint. Supply a key and the system uses that provider's default model unless you pick another. Current defaults, as defined in `src/sec_generative_search/providers/`:

| Provider   | Default chat model            | Embedding (if used)          |
| ---------- | ----------------------------- | ---------------------------- |
| OpenAI     | `gpt-5.4-mini`                | `text-embedding-3-small`     |
| Anthropic  | `claude-haiku-4-5`            | —                            |
| Gemini     | `gemini-3-flash-preview`      | `gemini-embedding-2-preview` |
| Mistral    | `mistral-small-2603`          | `mistral-embed`              |
| DeepSeek   | `deepseek-v4-flash`           | —                            |
| Grok       | `grok-4-1-fast-non-reasoning` | —                            |
| Qwen       | `qwen3.6-plus`                | `text-embedding-v4`          |
| Kimi       | `kimi-k2.5`                   | —                            |
| MiniMax    | `minimax-m2.7`                | —                            |
| MiMo       | `mimo-v2.5`                   | —                            |
| Z.ai       | `glm-5`                       | —                            |
| OpenRouter | `qwen/qwen3.6-plus`           | — (accepts any model slug)   |
| Local LLM  | `llama3.2` (any slug)         | — (self-hosted)              |
| Local      | — (no LLM; embedding only)    | `google/embeddinggemma-300m` |

`local_llm` targets a self-hosted OpenAI-wire server (`LOCAL_LLM_BASE_URL`, loopback by default; see [Configuration](#configuration)). Keys are never accepted via URL parameters or shell flags. Provide them through an environment variable, the `sec-rag provider set` prompt, or the SPA Provider Settings page (encrypted in the per-user vault).

---

## Deployment scenarios

| Scenario      | Who it is for             | Auth                          | At rest            | EDGAR identity              |
| ------------- | ------------------------- | ----------------------------- | ------------------ | --------------------------- |
| **A — Local** | Single user, own machine  | disabled                      | optional SQLCipher | env var                     |
| **B — Team**  | Small team, shared server | API + admin key               | SQLCipher required | per-session, per-user vault |
| **C — Cloud** | Internet-facing           | API + admin key + rate limits | SQLCipher required | per-session, per-user vault |

The repo ships a digest-pinned API image (`deploy/Dockerfile.api`), a stateless frontend image (`deploy/Dockerfile.frontend`), a portable Compose + nginx stack, and GCP Cloud Run manifests with a CI-gated keyless (Workload Identity Federation) deploy workflow.

> **HTTPS is mandatory for Scenarios B and C.** Without TLS, API keys, per-session EDGAR credentials, and queries cross the network in plaintext. The `api` service must run as **exactly one replica** — task IDs are minted in process.

Local quick run:

```bash
docker build -f deploy/Dockerfile.api -t sec-gs-api:local .
docker run --rm -p 8000:8000 \
  -e EDGAR_IDENTITY_NAME="Your Name" \
  -e EDGAR_IDENTITY_EMAIL="your@email.com" \
  -v sec_gs_data:/app/data \
  sec-gs-api:local
```

All deployment artefacts live under [`deploy/`](deploy/): the two Dockerfiles, the Compose + nginx stack (`docker-compose.yml`, `nginx/`), and the GCP Cloud Run manifests (`cloud/`) with their Cloud Build config and Grafana dashboard.

---

## Security design

| Control                                                                       | What it closes                                                                                       |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Two-tier API key (`API_KEY` + `API_ADMIN_KEY`)                                | Destructive routes require both keys; a leaked read key cannot probe the delete surface.             |
| Per-user vault (PBKDF2 → HKDF → AES-GCM)                                      | The server never holds your password or key-encryption key; provider keys are encrypted client-side. |
| SQLCipher at rest (`[encryption]`)                                            | Database-file theft without the key yields no usable data.                                           |
| Queries and chat never persisted                                              | No on-disk record of which companies you researched or what you asked.                               |
| `sanitize_retrieved_context()` + `<UNTRUSTED_FILING_CONTEXT>` delimiters      | Defence-in-depth against prompt injection carried inside filing text.                                |
| Citation integrity                                                            | Every citation must resolve to a retrieved chunk; unknown indices are dropped, never invented.       |
| Per-request CSP nonce + Trusted Types; `localStorage`/`sessionStorage` banned | XSS-to-key-exfiltration in the browser tier.                                                         |
| Per-session EDGAR identity                                                    | Prevents shared-identity rate-limit collapse in multi-tenant deployments.                            |
| Content-free correlation IDs and metric labels                                | Ticker, query, and `session_id` never reach a log aggregator or metrics axis.                        |

**Limitations.** None of this protects the prompt once it reaches the LLM provider — that is inherent to remote generation. SQLCipher protects a stolen database file, not a compromised running process (the key is in memory at runtime). A malicious browser extension can read the in-memory vault. HTTPS and host security are prerequisites, not features.

---

## Testing

```bash
# Backend — default suite (load tests deselected via addopts)
.venv/bin/python -m pytest

# Backend — opt-in load / throughput suite
.venv/bin/python -m pytest -m load

# Lint and format
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .

# Frontend
cd frontend
pnpm test        # Vitest
pnpm lint
pnpm typecheck
pnpm build
pnpm audit:ci
```

**Backend:** 2685 tests, of which 978 are security-regression locks (`@pytest.mark.security`). **Frontend:** 294 tests across security and functional trees. A security-tree failure is a release blocker.

---

## Project structure

```text
SEC-GenerativeSearch/
├── pyproject.toml                       # Package config, dependencies, CLI entry point (sec-rag)
├── .env.example                         # Environment variable template
├── deploy/                              # Dockerfiles, Compose + nginx, Cloud Run manifests, Grafana dashboard
├── src/sec_generative_search/
│   ├── config/                          # Pydantic settings, deployment profiles, constants
│   ├── core/                            # Types, exceptions, security primitives, logging, metrics, resilience
│   ├── pipeline/                        # Fetch, parse, chunk, embed, orchestrate
│   ├── database/                        # ChromaDB client, SQLite/SQLCipher registry, encrypted vault, backup
│   ├── providers/                       # Embedder + 13 LLM providers (incl. self-hosted local_llm), factory, registry
│   ├── retrieval/                       # RetrievalService
│   ├── rag/                             # RAG orchestrator, prompts, citations, evaluation
│   ├── search/                          # Search facade + evaluation
│   ├── cli/                             # Typer CLI (ingest, search, manage, rag, provider, evaluate)
│   └── api/                             # FastAPI backend (routes, schemas, middleware, tasks, WebSocket)
├── frontend/                            # Next.js 16 + React 19 SPA
└── tests/                               # unit / integration / api / load (mirrors src/ layout)
```

---

## Technology stack

| Component        | Library                                                                           | Purpose                                   |
| ---------------- | --------------------------------------------------------------------------------- | ----------------------------------------- |
| Filing retrieval | [edgartools](https://github.com/dgunning/edgartools)                              | SEC EDGAR API wrapper                     |
| HTML parsing     | [doc2dict](https://github.com/john-friedman/doc2dict)                             | Structured document extraction            |
| Local embeddings | [sentence-transformers](https://sbert.net/)                                       | `google/embeddinggemma-300m` (768-dim)    |
| Vector database  | [ChromaDB](https://www.trychroma.com/)                                            | Persistent local store, cosine similarity |
| Metadata store   | SQLite / [SQLCipher](https://www.zetetic.net/sqlcipher/)                          | Registry, retention, encrypted vault      |
| LLM providers    | `openai`, `anthropic`, `google-genai` (+ OpenAI-compatible vendors)               | Answer generation                         |
| REST API         | [FastAPI](https://fastapi.tiangolo.com/)                                          | Backend with WebSocket + SSE streaming    |
| Frontend         | [Next.js](https://nextjs.org/) 16 + [React](https://react.dev/) 19                | App Router, Tailwind CSS, React Query     |
| CLI              | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/)       | Formatted CLI output                      |
| Configuration    | [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Environment-based config                  |

---

## Licensing & usage

**SEC-GenerativeSearch** is licensed under the **Business Source License 1.1 (BSL 1.1)**. The goal is to keep the code open for developers, students, and researchers to study the RAG architecture, chunking, and security design, while protecting it against unauthorised commercial exploitation and competing hosted offerings.

**Free & open** — personal and academic projects, non-commercial research, and temporary internal evaluation (including running the full stack locally to assess fit).

**Requires a commercial licence** — production deployment in a company's operations or data pipelines, and offering the software to third parties as a hosted service, API, or search platform.

**Future open-source conversion:** every release automatically transitions to the **Apache License 2.0** four years after its release date.

For commercial licensing enquiries, contact me directly.
