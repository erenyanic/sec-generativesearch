"""Application-wide constants for SEC-GenerativeSearch."""

# Supported SEC filing forms (alphabetical order).
# Amendments (e.g. 10-K/A) are separate form types because edgartools
# marks them with a distinct ``form`` attribute and supports querying
# them directly via ``Company.get_filings(form='10-K/A')``.
SUPPORTED_FORMS = ("8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A")

# Base (non-amendment) form types — used by the amendment filter to
# decide whether to skip ``/A`` filings returned by edgartools.
BASE_FORMS = ("8-K", "10-K", "10-Q")

# Amendment form types — the ``/A`` variants.
AMENDMENT_FORMS = ("8-K/A", "10-K/A", "10-Q/A")

# Default form types for ingestion (both annual and quarterly)
DEFAULT_FORM_TYPES = "10-K,10-Q"


def parse_form_types(form_input: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated form type string.

    Accepts ``"10-K"``, ``"10-Q"``, ``"10-K,10-Q"``, ``"10-Q, 10-K"``, etc.
    Returns a sorted tuple of unique, validated form types.  Input order
    does not matter — ``"10-Q,10-K"`` produces the same result as
    ``"10-K,10-Q"``.

    Args:
        form_input: Comma-separated form type string.

    Returns:
        Sorted tuple of validated form types.

    Raises:
        ValueError: If any form type is not in ``SUPPORTED_FORMS``.
    """
    raw = [part.strip().upper() for part in form_input.split(",") if part.strip()]

    if not raw:
        raise ValueError(f"Empty form type. Supported: {', '.join(SUPPORTED_FORMS)}")

    invalid = [f for f in raw if f not in SUPPORTED_FORMS]
    if invalid:
        raise ValueError(
            f"Unsupported form type(s): {', '.join(invalid)}. "
            f"Supported: {', '.join(SUPPORTED_FORMS)}"
        )

    return tuple(sorted(set(raw)))


# Embedding model parameters
EMBEDDING_DIMENSION = 768  # Dimension of google/embeddinggemma-300m
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"

# Chunking parameters
DEFAULT_CHUNK_TOKEN_LIMIT = 500
DEFAULT_CHUNK_TOLERANCE = 50

# Database
DEFAULT_CHROMADB_PATH = "./data/chroma_db"
DEFAULT_METADATA_DB_PATH = "./data/metadata.sqlite"
DEFAULT_MAX_FILINGS = 500

# Collection naming
COLLECTION_NAME = "sec_filings"

# Search defaults
DEFAULT_SEARCH_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.0

# ---------------------------------------------------------------------------
# RAG pipeline constants
# ---------------------------------------------------------------------------

# Default context-window token budget for retrieved chunks.
DEFAULT_CONTEXT_TOKEN_BUDGET = 6000

# Default chunk overlap in tokens (context continuity between adjacent chunks).
DEFAULT_CHUNK_OVERLAP_TOKENS = 50

# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------

# Default timeout for external LLM/embedding API calls (seconds).
DEFAULT_PROVIDER_TIMEOUT = 60

# Default maximum retries for transient provider failures.
DEFAULT_PROVIDER_MAX_RETRIES = 3

# Default exponential backoff base (seconds).
DEFAULT_PROVIDER_RETRY_BACKOFF_BASE = 2.0
