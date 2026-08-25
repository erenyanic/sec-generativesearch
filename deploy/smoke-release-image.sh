#!/usr/bin/env bash
# ===========================================================================
# smoke-release-image.sh — release-image security smoke (Phase 19.2, item 6)
# ===========================================================================
# Four assertions that a static/test-suite pass structurally cannot make,
# because each one depends on the *built image* rather than the source tree:
#
#   1. SQLCipher at rest      — with DB_ENCRYPTION_KEY set, the metadata
#                               database on the volume is genuinely
#                               ciphertext, the runtime stage's compiled
#                               pysqlcipher3 opens it with the key, and a
#                               plain sqlite3 open of the same file fails.
#   2. Boot refusal           — DB_PERSIST_PROVIDER_CREDENTIALS=true with no
#                               key aborts at settings load, before uvicorn
#                               binds a port.
#   3. Docs 404 under API_KEY — /docs, /redoc and /openapi.json are 404 (in
#                               the unified error envelope) when API_KEY is
#                               set, and 200 when it is not.
#   4. 429 header set         — a rate-limited response still carries the
#                               full SecurityHeadersMiddleware header set
#                               plus Retry-After (SecurityHeaders sits
#                               OUTSIDE RateLimit in the middleware stack).
#
# The image is NOT built here — build it first, then point the script at it:
#
#   docker build -f deploy/Dockerfile.api -t sec-gs-api:smoke .
#   deploy/smoke-release-image.sh sec-gs-api:smoke
#
# Requires: a container engine (docker or podman) and curl.  Everything the
# script needs *inside* the container is already in the release image.
#
# Exit code is 0 only when all four probes pass.
# ===========================================================================
set -euo pipefail

IMAGE="${1:-${SMOKE_IMAGE:-sec-gs-api:smoke}}"
PORT="${SMOKE_PORT:-18000}"
BASE_URL="http://127.0.0.1:${PORT}"
PREFIX="sec-gs-smoke-$$"

# Container engine: honour $CONTAINER_ENGINE, else prefer a reachable docker
# daemon, else podman.  Both drive the identical OCI image.
ENGINE="${CONTAINER_ENGINE:-}"
if [ -z "${ENGINE}" ]; then
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        ENGINE=docker
    elif command -v podman >/dev/null 2>&1; then
        ENGINE=podman
    else
        echo "FATAL: no usable container engine (docker daemon down, podman absent)." >&2
        exit 2
    fi
fi

# Ephemeral secrets, generated per run — never literals in this file, and
# never reused across runs.  The SQLCipher key is asserted absent from the
# container's own logs in probe 1.
DB_KEY="$(openssl rand -hex 32)"
API_KEY_VALUE="$(openssl rand -hex 24)"

PASSES=0
FAILURES=0

# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
banner() { printf '\n=== %s ===\n' "$1"; }
pass()   { PASSES=$((PASSES + 1)); printf '  PASS  %s\n' "$1"; }
fail()   { FAILURES=$((FAILURES + 1)); printf '  FAIL  %s\n' "$1"; }

# check <description> <actual> <expected>
check() {
    if [ "$2" = "$3" ]; then
        pass "$1 (= $3)"
    else
        fail "$1 (expected '$3', got '$2')"
    fi
}

# check_contains <description> <haystack> <needle>
check_contains() {
    case "$2" in
        *"$3"*) pass "$1" ;;
        *)      fail "$1 (missing '$3')" ;;
    esac
}

# check_absent <description> <haystack> <needle>
check_absent() {
    case "$2" in
        *"$3"*) fail "$1 (found '$3')" ;;
        *)      pass "$1" ;;
    esac
}

# --------------------------------------------------------------------------
# Container helpers
# --------------------------------------------------------------------------
cleanup() {
    "${ENGINE}" rm -f -v "${PREFIX}-encrypted" "${PREFIX}-refuse" \
        "${PREFIX}-docs-on" "${PREFIX}-docs-off" "${PREFIX}-rate" \
        >/dev/null 2>&1 || true
    "${ENGINE}" volume rm -f "${PREFIX}-data" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# wait_for_health <container name> — poll the unauthenticated /api/health
# until it answers 200 or the budget runs out.
wait_for_health() {
    local name="$1" attempt=0
    while [ "${attempt}" -lt 60 ]; do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL}/api/health" || true)" = "200" ]; then
            return 0
        fi
        if [ "$("${ENGINE}" inspect -f '{{.State.Running}}' "${name}" 2>/dev/null || echo false)" != "true" ]; then
            echo "FATAL: ${name} exited before becoming healthy. Logs:" >&2
            "${ENGINE}" logs "${name}" >&2 2>&1 || true
            return 1
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    echo "FATAL: ${name} never became healthy within 60s. Logs:" >&2
    "${ENGINE}" logs "${name}" >&2 2>&1 || true
    return 1
}

# status_of <path> [extra curl args...]
status_of() {
    local path="$1"; shift
    curl -s -o /dev/null -w '%{http_code}' "$@" "${BASE_URL}${path}"
}

printf 'Release-image security smoke\n'
printf '  image  : %s\n' "${IMAGE}"
printf '  engine : %s (%s)\n' "${ENGINE}" "$("${ENGINE}" --version 2>/dev/null | head -1)"
printf '  port   : %s\n' "${PORT}"

if ! "${ENGINE}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "FATAL: image '${IMAGE}' not found. Build it first:" >&2
    echo "  ${ENGINE} build -f deploy/Dockerfile.api -t ${IMAGE} ." >&2
    exit 2
fi

# ==========================================================================
# Probe 1 — SQLCipher at rest
# ==========================================================================
banner "1. SQLCipher at rest (DB_ENCRYPTION_KEY set)"

"${ENGINE}" volume create "${PREFIX}-data" >/dev/null
"${ENGINE}" run -d --name "${PREFIX}-encrypted" \
    -p "127.0.0.1:${PORT}:8000" \
    -v "${PREFIX}-data:/app/data" \
    -e DB_ENCRYPTION_KEY="${DB_KEY}" \
    -e DB_PERSIST_PROVIDER_CREDENTIALS=true \
    "${IMAGE}" >/dev/null

wait_for_health "${PREFIX}-encrypted"

# (a) The file on the volume must NOT be a readable SQLite database.  A
#     plain sqlite3 file starts with the 16-byte "SQLite format 3\0" magic;
#     SQLCipher encrypts from byte 0, salt included.
HEADER="$("${ENGINE}" exec "${PREFIX}-encrypted" python -c \
    "print(open('/app/data/metadata.sqlite','rb').read(16))" 2>&1 || true)"
check_absent "metadata.sqlite carries no plain-SQLite magic header" \
    "${HEADER}" "SQLite format 3"

# (b) The compiled pysqlcipher3 extension in the runtime stage opens it with
#     the key.  This is what proves libsqlcipher0 is actually linked — a
#     header check alone would also pass on a corrupt file.
OPEN_WITH_KEY="$("${ENGINE}" exec -e SMOKE_DB_KEY="${DB_KEY}" "${PREFIX}-encrypted" python -c '
import os
from pysqlcipher3 import dbapi2 as sqlcipher

conn = sqlcipher.connect("/app/data/metadata.sqlite")
conn.execute("PRAGMA key = \"x\x27%s\x27\"" % os.environ["SMOKE_DB_KEY"].encode().hex())
names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
conn.close()
print("OPENED" if {"schema_version", "filings"} <= names else "MISSING_TABLES:%s" % sorted(names))
' 2>&1 || true)"
check "pysqlcipher3 opens the database with the key" "${OPEN_WITH_KEY}" "OPENED"

# (c) The same file must NOT open with the stdlib sqlite3 driver (no key).
OPEN_NO_KEY="$("${ENGINE}" exec "${PREFIX}-encrypted" python -c '
import sqlite3

try:
    conn = sqlite3.connect("/app/data/metadata.sqlite")
    conn.execute("SELECT name FROM sqlite_master").fetchall()
    print("READABLE")
except sqlite3.Error:
    print("REFUSED")
' 2>&1 || true)"
check "plain sqlite3 cannot read the encrypted database" "${OPEN_NO_KEY}" "REFUSED"

ENCRYPTED_LOGS="$("${ENGINE}" logs "${PREFIX}-encrypted" 2>&1 || true)"

# (d) The registry must not have taken the silent plain-sqlite3 fallback that
#     fires when the [encryption] extra is missing from the image.
check_absent "SQLCipher driver did not fall back to plain sqlite3" \
    "${ENCRYPTED_LOGS}" "pysqlcipher3 is not installed"

# (e) The key must never reach the container's own log stream.
check_absent "boot logs never echo the SQLCipher key" "${ENCRYPTED_LOGS}" "${DB_KEY}"

"${ENGINE}" rm -f -v "${PREFIX}-encrypted" >/dev/null 2>&1 || true

# ==========================================================================
# Probe 2 — boot refusal without a key
# ==========================================================================
banner "2. Boot refusal: DB_PERSIST_PROVIDER_CREDENTIALS=true, no key"

set +e
REFUSE_OUTPUT="$("${ENGINE}" run --name "${PREFIX}-refuse" \
    -e DB_PERSIST_PROVIDER_CREDENTIALS=true \
    "${IMAGE}" 2>&1)"
REFUSE_EXIT=$?
set -e

if [ "${REFUSE_EXIT}" -ne 0 ]; then
    pass "container exits non-zero (exit ${REFUSE_EXIT})"
else
    fail "container exited 0 — it must refuse to boot"
fi

# The refusal must land at settings load, BEFORE the server binds a port.
check_absent "uvicorn never reaches the listening state" \
    "${REFUSE_OUTPUT}" "Uvicorn running on"

# The message must name both knobs the operator can act on.
check_contains "refusal names DB_PERSIST_PROVIDER_CREDENTIALS" \
    "${REFUSE_OUTPUT}" "DB_PERSIST_PROVIDER_CREDENTIALS"
check_contains "refusal names DB_ENCRYPTION_KEY" \
    "${REFUSE_OUTPUT}" "DB_ENCRYPTION_KEY"

"${ENGINE}" rm -f -v "${PREFIX}-refuse" >/dev/null 2>&1 || true

# ==========================================================================
# Probe 3 — docs surfaces 404 under API_KEY (and 200 without it)
# ==========================================================================
banner "3. Docs endpoints 404 with API_KEY set"

"${ENGINE}" run -d --name "${PREFIX}-docs-on" \
    -p "127.0.0.1:${PORT}:8000" \
    -e API_KEY="${API_KEY_VALUE}" \
    "${IMAGE}" >/dev/null
wait_for_health "${PREFIX}-docs-on"

for path in /docs /redoc /openapi.json; do
    check "GET ${path} is 404 (unauthenticated)" "$(status_of "${path}")" "404"
    check "GET ${path} is 404 (with a valid API key)" \
        "$(status_of "${path}" -H "X-API-Key: ${API_KEY_VALUE}")" "404"
done

# The 404 must use the unified {error,message,details,hint} envelope, not
# Starlette's raw {"detail": ...} — that is the surface the Starlette
# exception handler exists to cover.
DOCS_BODY="$(curl -s "${BASE_URL}/openapi.json")"
check_contains "404 body uses the unified error envelope" "${DOCS_BODY}" '"error"'
check_absent "404 body does not leak Starlette's raw detail shape" \
    "${DOCS_BODY}" '"detail"'

"${ENGINE}" rm -f -v "${PREFIX}-docs-on" >/dev/null 2>&1 || true

# Control run: the same image with API_KEY unset must SERVE the docs.  Without
# this differential the 404s above would also pass on a broken server.
banner "3b. Control: docs served when API_KEY is unset"

"${ENGINE}" run -d --name "${PREFIX}-docs-off" \
    -p "127.0.0.1:${PORT}:8000" \
    "${IMAGE}" >/dev/null
wait_for_health "${PREFIX}-docs-off"

for path in /docs /redoc /openapi.json; do
    check "GET ${path} is 200 without API_KEY" "$(status_of "${path}")" "200"
done

"${ENGINE}" rm -f -v "${PREFIX}-docs-off" >/dev/null 2>&1 || true

# ==========================================================================
# Probe 4 — a 429 carries the full security-header set
# ==========================================================================
banner "4. 429 carries the security-header set + Retry-After"

# API_RATE_LIMIT_GENERAL=1 makes the breach deterministic in two requests.
# The assertion is header presence on a 429, not the numeric limit, so
# lowering the bucket is faithful to what is under test.
"${ENGINE}" run -d --name "${PREFIX}-rate" \
    -p "127.0.0.1:${PORT}:8000" \
    -e API_KEY="${API_KEY_VALUE}" \
    -e API_RATE_LIMIT_GENERAL=1 \
    "${IMAGE}" >/dev/null
wait_for_health "${PREFIX}-rate"

# /api/providers/ rides the `general` bucket.  The rate limiter sits OUTSIDE
# the route's auth dependency, so no key is needed to provoke the 429.
curl -s -o /dev/null "${BASE_URL}/api/providers/"
RATE_HEADERS="$(curl -s -D - -o /dev/null "${BASE_URL}/api/providers/")"
RATE_STATUS="$(printf '%s' "${RATE_HEADERS}" | head -1 | tr -d '\r')"

# The header assertions below are only meaningful ON a 429 — every response
# carries the security-header set, so running them against a 401 would print
# passes for a property they never tested.  Gate the whole loop on the status
# line rather than letting it run past a failed status check.
RATE_HEADERS_LC="$(printf '%s' "${RATE_HEADERS}" | tr '[:upper:]' '[:lower:]')"
case "${RATE_STATUS}" in
    *429*)
        pass "second request is rate-limited (${RATE_STATUS})"
        # Header names are case-insensitive on the wire; folded above.
        for header in \
            "x-content-type-options: nosniff" \
            "x-frame-options: deny" \
            "x-xss-protection: 1; mode=block" \
            "referrer-policy: strict-origin-when-cross-origin" \
            "content-security-policy: default-src 'self'" \
            "permissions-policy: camera=()" \
            "retry-after:"
        do
            check_contains "429 carries '${header%%:*}'" "${RATE_HEADERS_LC}" "${header}"
        done
        ;;
    *)
        fail "second request is rate-limited (got '${RATE_STATUS}')"
        fail "429 header set NOT evaluated — the response was not a 429"
        ;;
esac

# The 429 body must stay the unified envelope and must not echo the API key.
RATE_BODY="$(curl -s "${BASE_URL}/api/providers/" -H "X-API-Key: ${API_KEY_VALUE}")"
check_contains "429 body uses the unified error envelope" "${RATE_BODY}" '"rate_limited"'
check_absent "429 body never echoes the API key" "${RATE_BODY}" "${API_KEY_VALUE}"

"${ENGINE}" rm -f -v "${PREFIX}-rate" >/dev/null 2>&1 || true

# ==========================================================================
# Verdict
# ==========================================================================
banner "Result"
printf '  passed : %d\n  failed : %d\n' "${PASSES}" "${FAILURES}"

if [ "${FAILURES}" -ne 0 ]; then
    echo "SMOKE FAILED — do not push this image." >&2
    exit 1
fi
echo "SMOKE PASSED"
