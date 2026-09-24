"""Security + correctness lockers for the deployment artefacts.

``deploy/Dockerfile.api``, ``deploy/Dockerfile.frontend``,
``deploy/docker-entrypoint.sh``, the repo-root ``.dockerignore`` and
``frontend/.dockerignore`` are operator artefacts, not code — so nothing stops
them silently regressing into an insecure image. These static tests are the
load-bearing control for both deployment images:

    - **Supply chain.** Every base image MUST be digest-pinned
      (``@sha256:<64 hex>``) — mirrors the SHA-pinned GitHub Actions in
      ``ci.yml``. A floating ``:tag`` is a silent-substitution vector.
    - **Least privilege.** The long-lived server MUST run non-root. The image
      uses the gosu pattern: start as root only to chown the volume, then
      ``exec gosu`` to an unprivileged account. Both halves are asserted.
    - **No baked secret.** Neither the Dockerfile nor the entrypoint may carry
      secret-shaped material, and the build MUST NOT slurp the whole context
      (which would drag ``.env`` into a layer). ``.dockerignore`` MUST exclude
      every secret-/state-bearing path.
        - **Operability.** A ``HEALTHCHECK`` probes the unauth ``/api/health``; the
            server runs exactly one worker with ``--proxy-headers`` (the in-process
            TaskManager single-replica contract).

All assertions are on tracked, CI-visible files; nothing here requires Docker
or a network, so the lockers run in the normal pytest job.
"""

from __future__ import annotations

import ast
import ipaddress
import json
import re
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src" / "sec_generative_search"
_DOCKERFILE_API = _REPO_ROOT / "deploy" / "Dockerfile.api"
_DOCKERFILE_FRONTEND = _REPO_ROOT / "deploy" / "Dockerfile.frontend"
_ENTRYPOINT = _REPO_ROOT / "deploy" / "docker-entrypoint.sh"
_DOCKERIGNORE = _REPO_ROOT / ".dockerignore"
_FRONTEND_DOCKERIGNORE = _REPO_ROOT / "frontend" / ".dockerignore"
_NEXT_CONFIG = _REPO_ROOT / "frontend" / "next.config.ts"
_COMPOSE = _REPO_ROOT / "deploy" / "docker-compose.yml"
_NGINX_CONF = _REPO_ROOT / "deploy" / "nginx" / "nginx.conf"
_GITIGNORE = _REPO_ROOT / ".gitignore"
_CLOUD_API = _REPO_ROOT / "deploy" / "cloud" / "api-service.yaml"
_CLOUD_FRONTEND = _REPO_ROOT / "deploy" / "cloud" / "frontend-service.yaml"
_CLOUD_JOB = _REPO_ROOT / "deploy" / "cloud" / "demo-reset-job.yaml"
_DEPLOY_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "deploy.yml"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_CLOUDBUILD = _REPO_ROOT / "deploy" / "cloudbuild.yaml"
_FRONTEND_PACKAGE_JSON = _REPO_ROOT / "frontend" / "package.json"

_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}\b")
_ARG_RE = re.compile(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(.+)$")
_FROM_RE = re.compile(r"^FROM\s+(\S+)")
_VAR_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")


def _collect_arg_defaults(dockerfile: str) -> dict[str, str]:
    """Map ``ARG NAME=default`` declarations to their default value."""
    defaults: dict[str, str] = {}
    for line in dockerfile.splitlines():
        match = _ARG_RE.match(line.strip())
        if match:
            defaults[match.group(1)] = match.group(2).strip()
    return defaults


def _collect_base_images(dockerfile: str) -> list[str]:
    """Every base reference on a ``FROM`` line, resolved through ARG defaults."""
    arg_defaults = _collect_arg_defaults(dockerfile)
    resolved: list[str] = []
    for line in dockerfile.splitlines():
        match = _FROM_RE.match(line.strip())
        if not match:
            continue
        token = match.group(1)
        var = _VAR_RE.match(token)
        if var:
            assert var.group(1) in arg_defaults, (
                f"FROM references ${var.group(1)} but no ARG default defines it"
            )
            token = arg_defaults[var.group(1)]
        resolved.append(token)
    return resolved


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return _DOCKERFILE_API.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def entrypoint() -> str:
    return _ENTRYPOINT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore() -> str:
    return _DOCKERIGNORE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontend_dockerfile() -> str:
    return _DOCKERFILE_FRONTEND.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontend_dockerignore() -> str:
    return _FRONTEND_DOCKERIGNORE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def next_config() -> str:
    return _NEXT_CONFIG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def arg_defaults(dockerfile: str) -> dict[str, str]:
    return _collect_arg_defaults(dockerfile)


@pytest.fixture(scope="module")
def base_images(dockerfile: str) -> list[str]:
    return _collect_base_images(dockerfile)


# ---------------------------------------------------------------------------
# Files exist
# ---------------------------------------------------------------------------


def test_deployment_artifacts_exist() -> None:
    assert _DOCKERFILE_API.is_file(), "deploy/Dockerfile.api is missing"
    assert _ENTRYPOINT.is_file(), "deploy/docker-entrypoint.sh is missing"
    assert _DOCKERIGNORE.is_file(), ".dockerignore is missing"


# ---------------------------------------------------------------------------
# Supply chain: digest-pinned bases
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_every_base_image_is_digest_pinned(base_images: list[str]) -> None:
    assert base_images, "no FROM instruction found in deploy/Dockerfile.api"
    floating = [image for image in base_images if not _DIGEST_RE.search(image)]
    assert not floating, (
        f"base image(s) not digest-pinned (@sha256:...): {floating}. "
        "A floating :tag is a silent-substitution / supply-chain vector."
    )


# ---------------------------------------------------------------------------
# Least privilege: non-root server via the gosu drop pattern
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_image_creates_unprivileged_user(dockerfile: str) -> None:
    assert re.search(r"^RUN\s+useradd\b", dockerfile, re.MULTILINE) or re.search(
        r"^RUN\s+adduser\b", dockerfile, re.MULTILINE
    ), "Dockerfile does not create an unprivileged service account"


@pytest.mark.security
def test_entrypoint_drops_privileges_via_gosu(entrypoint: str) -> None:
    # The final, long-lived process must not be root: the entrypoint execs
    # the server through gosu after the privileged chown.
    assert "gosu" in entrypoint, "entrypoint never drops privileges via gosu"
    assert re.search(r"\bexec\s+gosu\b", entrypoint), (
        "entrypoint must 'exec gosu <user> \"$@\"' so the non-root server is PID 1"
    )
    # And the fallback path (already non-root) must still exec, not fork.
    assert re.search(r"\bexec\s+\"\$@\"", entrypoint), (
        "entrypoint must exec the command so signals reach the server directly"
    )


@pytest.mark.security
def test_dockerfile_does_not_pin_a_root_runtime_user(dockerfile: str) -> None:
    # We intentionally do NOT set `USER appuser` (gosu drops at runtime), but a
    # stray `USER root` as the last USER directive would defeat the pattern.
    user_lines = re.findall(r"^USER\s+(\S+)", dockerfile, re.MULTILINE)
    assert "root" not in user_lines, "Dockerfile pins USER root — defeats the gosu drop"


# ---------------------------------------------------------------------------
# No baked secret
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_no_baked_secret_in_image_definition(dockerfile: str, entrypoint: str) -> None:
    blob = f"{dockerfile}\n{entrypoint}"
    lowered = blob.lower()
    # Obvious secret-shaped tokens. The sk- check is anchored on a word
    # boundary + key-length tail so it flags `sk-proj-...` keys, not the
    # `sk-` buried inside an English word (e.g. "ta[sk-]404s").
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), (
        "Dockerfile/entrypoint contains an sk- API-key-shaped token"
    )
    for needle in ("bearer ", "authorization:", "private key"):
        assert needle not in lowered, (
            f"image definition contains secret-shaped material: {needle!r}"
        )
    # Secret-bearing env knobs must never be ASSIGNED a literal value in the
    # image (referencing them at runtime is fine; only `NAME=<value>` is the
    # leak). Allow `NAME=` empty and `NAME=${VAR}` indirections.
    secret_env = (
        "API_KEY",
        "API_ADMIN_KEY",
        "API_AUTH_PEPPER",
        "DB_ENCRYPTION_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGING_FACE_TOKEN",
        # The variable the factory actually reads for the (gated) local
        # embedder — ``HUGGING_FACE_TOKEN`` above is a dead knob (F35).
        "HF_TOKEN",
    )
    for name in secret_env:
        bad = re.search(rf"\b{name}=(?!\s|$|\$)\S", blob)
        assert not bad, f"image definition assigns a literal value to {name} — never bake a secret"
    # A build ARG is recorded in the image history even without a default, so
    # a token must never arrive that way (only a BuildKit secret mount would do).
    for name in ("HF_TOKEN", "HUGGING_FACE_TOKEN"):
        assert not re.search(rf"^\s*ARG\s+{name}\b", dockerfile, re.MULTILINE), (
            f"Dockerfile declares ARG {name} — build args persist in the image "
            "history; use RUN --mount=type=secret instead"
        )


@pytest.mark.security
def test_dockerfile_does_not_copy_whole_context(dockerfile: str) -> None:
    # `COPY . .` / `ADD . .` would drag the entire context (incl. a stray
    # `.env`) into a layer. Selective COPYs only.
    for line in dockerfile.splitlines():
        stripped = line.strip()
        assert not re.match(r"^(COPY|ADD)\s+\.(\s|/|$)", stripped), (
            f"image copies the whole build context: {stripped!r}. Copy explicit paths only."
        )


# ---------------------------------------------------------------------------
# .dockerignore excludes every secret-/state-bearing path
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_dockerignore_excludes_secrets_and_state(dockerignore: str) -> None:
    entries = {line.strip() for line in dockerignore.splitlines() if line.strip()}
    required = {
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "edgar-identity.txt",
        "data/",
        ".git/",
        ".venv/",
    }
    missing = sorted(required - entries)
    assert not missing, f".dockerignore is missing secret/state exclusions: {missing}"


# ---------------------------------------------------------------------------
# Operability: healthcheck + single-worker proxy-aware server
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_healthcheck_probes_health_endpoint(dockerfile: str) -> None:
    assert "HEALTHCHECK" in dockerfile, "Dockerfile has no HEALTHCHECK"
    assert "/api/health" in dockerfile, "HEALTHCHECK does not probe /api/health"


def test_server_runs_single_worker_behind_proxy(dockerfile: str) -> None:
    # Collapse the CMD continuation lines so flag/value pairs are adjacent.
    flat = re.sub(r"\\\s*\n", " ", dockerfile)
    cmd = next((ln for ln in flat.splitlines() if ln.strip().startswith("CMD")), "")
    assert "uvicorn" in cmd, "CMD does not launch uvicorn"
    assert "create_app" in cmd and "--factory" in cmd, "CMD does not use the create_app factory"
    assert re.search(r'"--workers",\s*"1"', cmd), (
        "uvicorn must run exactly one worker (in-process TaskManager contract)"
    )
    assert "--proxy-headers" in cmd, "uvicorn must run with --proxy-headers behind nginx/GFE"


# ---------------------------------------------------------------------------
# Boot-time egress: tokenizer bake + model-cache placement (F25)
#
#   - The tiktoken BPE files are fetched in the BUILDER stage into a cache
#     under /opt/venv — the only tree the runtime stage inherits — and the
#     runtime ENV names the same path. Without this the API lifespan downloads
#     cl100k_base from Azure blob storage on every fresh container.
#   - The image's HF_HOME sits on the /app/data volume, and the entrypoint's
#     fallback default names the same path (the entrypoint re-owns it).
# ---------------------------------------------------------------------------

_TIKTOKEN_ENCODINGS = ("cl100k_base", "o200k_base")


def _dockerfile_stages(dockerfile: str) -> dict[str, str]:
    """Map each ``FROM … AS <name>`` stage to its instructions.

    Comment lines are dropped and ``\\``-continuations joined, so a path
    quoted in a comment can never satisfy an assertion.
    """
    lines = [ln for ln in dockerfile.splitlines() if not ln.lstrip().startswith("#")]
    flat = re.sub(r"\\\s*\n", " ", "\n".join(lines))
    stages: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in flat.splitlines():
        match = re.match(r"^FROM\s+\S+\s+AS\s+(\S+)", line.strip(), re.IGNORECASE)
        if match:
            current = stages.setdefault(match.group(1), [])
            continue
        if current is not None:
            current.append(line)
    return {name: "\n".join(body) for name, body in stages.items()}


def _env_value(stage: str, name: str) -> str | None:
    """Last value an ``ENV`` instruction in *stage* assigns to *name*."""
    values = [
        m.group(1)
        for line in stage.splitlines()
        if line.strip().startswith("ENV")
        for m in re.finditer(rf"\b{name}=(\"[^\"]*\"|\S+)", line)
    ]
    return values[-1].strip('"') if values else None


def test_tiktoken_encodings_are_baked_into_the_inherited_venv(dockerfile: str) -> None:
    stages = _dockerfile_stages(dockerfile)
    assert {"builder", "runtime"} <= set(stages), f"expected builder + runtime stages: {stages}"
    builder, runtime = stages["builder"], stages["runtime"]

    cache_dir = _env_value(builder, "TIKTOKEN_CACHE_DIR")
    assert cache_dir, "builder stage never sets TIKTOKEN_CACHE_DIR before the prefetch"
    assert cache_dir.startswith("/opt/venv/"), (
        f"TIKTOKEN_CACHE_DIR={cache_dir!r} is outside /opt/venv — the runtime stage "
        "inherits the builder only through COPY --from=builder /opt/venv, so the "
        "baked BPE files would be lost"
    )
    assert re.search(r"^COPY\s+--from=builder\s+/opt/venv\s+/opt/venv", runtime, re.M), (
        "runtime stage no longer inherits /opt/venv from the builder"
    )
    assert _env_value(runtime, "TIKTOKEN_CACHE_DIR") == cache_dir, (
        "runtime TIKTOKEN_CACHE_DIR must name the builder's baked cache — otherwise "
        "tiktoken falls back to <tmp>/data-gym-cache and downloads at boot"
    )

    # The prefetch must run AFTER the cache ENV (else it fills /tmp) and name
    # every encoding the codebase can reach.
    env_at = builder.find("TIKTOKEN_CACHE_DIR=")
    prefetch = [
        (builder.find(line), line)
        for line in builder.splitlines()
        if line.strip().startswith("RUN") and "get_encoding" in line
    ]
    assert prefetch, "builder stage never prefetches the tiktoken encodings"
    position, command = prefetch[-1]
    assert position > env_at, "tiktoken prefetch runs before TIKTOKEN_CACHE_DIR is set"
    for encoding in _TIKTOKEN_ENCODINGS:
        assert f"get_encoding('{encoding}')" in command or (
            f'get_encoding("{encoding}")' in command
        ), f"tiktoken prefetch does not fetch {encoding}"


def test_tiktoken_prefetch_covers_every_encoding_the_catalogue_reaches() -> None:
    # The baked cache is root-owned and read-only to the server; with a
    # user-specified TIKTOKEN_CACHE_DIR tiktoken RAISES on a failed cache
    # write instead of downloading. So an encoding outside the baked set is a
    # hard failure at runtime, not a slow path — keep the set exhaustive.
    tiktoken = pytest.importorskip("tiktoken")
    from sec_generative_search.providers.catalogue import ModelCatalogue
    from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface

    baseline = ModelCatalogue.load_baseline()
    llm_providers = [
        entry.name
        for entry in ProviderRegistry.all_entries(ProviderSurface.LLM, include_unavailable=True)
    ]
    assert any(baseline.list_llm_models(name) for name in llm_providers), "empty catalogue"
    reached = {"cl100k_base"}  # retrieval / orchestrator / fallback counter
    for provider in llm_providers:
        for slug in baseline.list_llm_models(provider):
            try:
                reached.add(tiktoken.model.encoding_name_for_model(slug))
            except KeyError:
                reached.add("cl100k_base")  # openai_compat.count_tokens fallback
    missing = reached - set(_TIKTOKEN_ENCODINGS)
    assert not missing, (
        f"catalogued models reach tiktoken encodings the image does not bake: "
        f"{sorted(missing)} — add them to the Dockerfile prefetch and this locker"
    )


def test_hf_home_is_on_the_volume_and_matches_the_entrypoint(
    dockerfile: str, entrypoint: str
) -> None:
    runtime = _dockerfile_stages(dockerfile)["runtime"]
    hf_home = _env_value(runtime, "HF_HOME")
    assert hf_home, "runtime stage never sets HF_HOME"
    volumes = re.findall(r'^VOLUME\s+\[\s*"([^"]+)"', runtime, re.MULTILINE)
    assert volumes, "runtime stage declares no VOLUME"
    assert any(hf_home.startswith(v.rstrip("/") + "/") for v in volumes), (
        f"HF_HOME={hf_home!r} is not under a declared VOLUME {volumes} — the gated "
        "embedder weights would be re-downloaded on every new container"
    )
    fallback = re.search(r'HF_HOME="\$\{HF_HOME:-([^}]+)\}"', entrypoint)
    assert fallback, "entrypoint no longer defaults HF_HOME"
    assert fallback.group(1) == hf_home, (
        f"entrypoint HF_HOME default {fallback.group(1)!r} != image HF_HOME "
        f"{hf_home!r} — the entrypoint would create/re-own the wrong directory"
    )


@pytest.mark.security
def test_dockerfile_never_bakes_a_wildcard_trusted_proxy_set(dockerfile: str) -> None:
    # M2 regression lock. `--forwarded-allow-ips *` puts uvicorn's
    # _TrustedHosts into always_trust mode, where get_trusted_client_host()
    # returns the LEFTMOST X-Forwarded-For entry. nginx forwards
    # `$proxy_add_x_forwarded_for`, which APPENDS the real peer to the RIGHT of
    # whatever the client sent — so the leftmost value is fully
    # client-controlled and `scope["client"]` (the per-IP rate-limit key)
    # becomes rotatable per request. The trusted set must be
    # deployment-supplied via $FORWARDED_ALLOW_IPS and always bounded.
    flat = re.sub(r"\\\s*\n", " ", dockerfile)
    assert "--forwarded-allow-ips" not in flat, (
        "Dockerfile bakes --forwarded-allow-ips; the trusted-proxy set must come "
        "from the deployment ($FORWARDED_ALLOW_IPS), never the image"
    )
    assert not re.search(r'FORWARDED_ALLOW_IPS[=:]\s*["\']?\*', flat), (
        "Dockerfile sets a wildcard FORWARDED_ALLOW_IPS — every per-IP rate-limit "
        "window becomes spoofable via X-Forwarded-For"
    )


# ---------------------------------------------------------------------------
# Dependency layers independent of src/ + the hash-pinned image lock (F26)
#
#   - LAYER CACHE. The builder stage — whose /opt/venv the runtime inherits —
#     takes exactly one file from the build context: the compiled lock. A
#     source edit then never re-runs the dependency install and never changes
#     the multi-GB venv layer; the application arrives as a wheel from its own
#     stage and is installed last.
#   - SUPPLY CHAIN. Every third-party package except torch comes from that
#     lock under --require-hashes --no-deps, and the application wheel installs
#     with --no-deps --no-index — nothing resolves outside the lock.
#   - LOCK INTEGRITY. The lock is ==-pinned and sha256-hashed, PyPI-only, omits
#     torch (its wheel index is a build-arg), and still satisfies every
#     dependency pyproject.toml declares for the baked extras.
# ---------------------------------------------------------------------------

_API_LOCK = _REPO_ROOT / "deploy" / "requirements.txt"
_API_LOCK_IN_CONTEXT = "deploy/requirements.txt"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
# The extras Dockerfile.api bakes — the lock is compiled from exactly these.
_IMAGE_EXTRAS = ("encryption", "metrics", "local-embeddings")
# Installed from the TORCH_INDEX_URL build-arg (CPU default, CUDA opt-in), so
# no single set of lock hashes can fit it.
_INDEX_RESOLVED_PACKAGES = frozenset({"torch"})
# The image's marker environment (python:3.12-slim-bookworm, x86_64).
_IMAGE_MARKER_ENV = {
    "python_version": "3.12",
    "python_full_version": "3.12.13",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "os_name": "posix",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}
_LOCK_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")
_LOCK_HASH_RE = re.compile(r"--hash=sha256:[0-9a-f]{64}(?:\s|$)")
# pip options that consume the next token as their value.
_PIP_VALUE_OPTIONS = frozenset(
    {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "--trusted-host",
        "-w",
        "--wheel-dir",
    }
)
# pip options that point at a package source other than the default index.
_PIP_SOURCE_OPTIONS = frozenset(
    {"-i", "--index-url", "--extra-index-url", "-f", "--find-links", "--trusted-host"}
)


@pytest.fixture(scope="module")
def api_lock() -> str:
    return _API_LOCK.read_text(encoding="utf-8")


def _lock_requirement_lines(lock: str) -> list[str]:
    """Logical lines of a requirements file: comments dropped, ``\\`` joined."""
    kept = [line for line in lock.splitlines() if not line.lstrip().startswith("#")]
    joined = re.sub(r"\\\s*\n", " ", "\n".join(kept))
    return [line.strip() for line in joined.splitlines() if line.strip()]


def _lock_pins(lock: str) -> dict[str, str]:
    """Canonical package name → pinned version for every ``==`` entry."""
    pins: dict[str, str] = {}
    for line in _lock_requirement_lines(lock):
        match = _LOCK_PIN_RE.match(line)
        if match:
            pins[canonicalize_name(match.group(1))] = match.group(2)
    return pins


def _context_copy_sources(line: str) -> list[str]:
    """Build-context sources of a ``COPY`` line (``[]`` for ``COPY --from=``)."""
    tokens = line.split()
    if not tokens or tokens[0] != "COPY" or any(t.startswith("--from=") for t in tokens):
        return []
    operands = [t for t in tokens[1:] if not t.startswith("--")]
    return operands[:-1]


def _pip_commands(stage: str) -> list[str]:
    """Every ``pip …`` command a stage runs, in order (``RUN`` split on ``&&``)."""
    commands: list[str] = []
    for line in stage.splitlines():
        stripped = line.strip()
        if stripped.startswith("RUN "):
            commands += [
                part.strip()
                for part in stripped[len("RUN ") :].split("&&")
                if part.strip().startswith("pip ")
            ]
    return commands


def _parse_pip(command: str) -> tuple[str, dict[str, str | None], list[str]]:
    """Split a ``pip`` command into (sub-command, options, positionals)."""
    tokens = shlex.split(command)
    options: dict[str, str | None] = {}
    positionals: list[str] = []
    rest = iter(tokens[2:])
    for token in rest:
        if not token.startswith("-"):
            positionals.append(token)
            continue
        name, has_value, value = token.partition("=")
        if has_value:
            options[name] = value
        elif name in _PIP_VALUE_OPTIONS:
            options[name] = next(rest, "")
        else:
            options[name] = None
    return tokens[1], options, positionals


def test_api_dependency_layers_are_independent_of_the_source_tree(dockerfile: str) -> None:
    stages = _dockerfile_stages(dockerfile)
    builder, runtime = stages["builder"], stages["runtime"]

    builder_sources = [
        source for line in builder.splitlines() for source in _context_copy_sources(line.strip())
    ]
    assert builder_sources == [_API_LOCK_IN_CONTEXT], (
        f"the builder stage copies {builder_sources} from the build context — it must "
        f"take only {_API_LOCK_IN_CONTEXT}, or a source/metadata edit busts the "
        "dependency install and the multi-GB venv layer the runtime inherits"
    )

    source_stages = sorted(
        name
        for name, body in stages.items()
        for line in body.splitlines()
        for source in _context_copy_sources(line.strip())
        if source.split("/")[0] == "src"
    )
    assert len(source_stages) == 1, f"expected exactly one stage to copy src/: {source_stages}"
    (project_stage,) = source_stages
    assert project_stage not in {"builder", "runtime"}, (
        f"src/ is copied into the {project_stage!r} stage — it must reach the image "
        "only as a wheel built in its own stage"
    )

    lines = [line.strip() for line in runtime.splitlines()]
    venv_at = next(
        (i for i, ln in enumerate(lines) if re.match(r"^COPY\s+--from=builder\s+/opt/venv\s", ln)),
        None,
    )
    wheel_at = next(
        (i for i, ln in enumerate(lines) if ln.startswith(f"COPY --from={project_stage} ")),
        None,
    )
    install_at = next(
        (i for i, ln in enumerate(lines) if ln.startswith("RUN ") and ".whl" in ln), None
    )
    assert None not in (venv_at, wheel_at, install_at), (
        "runtime must COPY the builder venv, then COPY + install the application wheel"
    )
    assert venv_at < wheel_at < install_at, (
        "the application wheel must be installed AFTER the venv COPY — the small "
        "per-edit layer goes last so the venv layer above stays cached"
    )


@pytest.mark.security
def test_api_image_installs_third_party_code_only_from_the_hashed_lock(dockerfile: str) -> None:
    # Supply-chain lock: the ONLY sanctioned pip invocations are the installer
    # self-upgrade, torch from the TORCH_INDEX_URL build-arg, the hash-pinned
    # lock, and our own wheel. Any other `pip install <pkg>` would resolve an
    # unpinned, unhashed package straight from an index.
    stages = _dockerfile_stages(dockerfile)
    lock_dest = next(
        line.split()[-1]
        for line in stages["builder"].splitlines()
        if _context_copy_sources(line.strip()) == [_API_LOCK_IN_CONTEXT]
    )
    sequence: dict[str, list[str]] = {name: [] for name in stages}
    for name, body in stages.items():
        for command in _pip_commands(body):
            sub, options, positionals = _parse_pip(command)
            where = f"[{name}] {command!r}"
            if sub == "check":
                sequence[name].append("check")
                continue
            if sub == "wheel":
                assert "--no-deps" in options, (
                    f"{where}: `pip wheel` without --no-deps also collects dependency "
                    "wheels, which the runtime would then install unhashed"
                )
                assert not _PIP_SOURCE_OPTIONS & options.keys(), f"{where}: extra index"
                sequence[name].append("wheel")
                continue
            assert sub == "install", f"{where}: unexpected pip sub-command"
            if positionals and all(p.endswith(".whl") for p in positionals):
                assert "--no-deps" in options and "--no-index" in options, (
                    f"{where}: the application wheel must install with --no-deps "
                    "--no-index — every dependency is already in the venv from the lock"
                )
                sequence[name].append("wheel-install")
            elif "-r" in options or "--requirement" in options:
                assert (options.get("-r") or options.get("--requirement")) == lock_dest, (
                    f"{where}: installs a requirements file other than the copied lock"
                )
                assert "--require-hashes" in options and "--no-deps" in options, (
                    f"{where}: the lock must install with --require-hashes (every "
                    "artefact matches a pinned sha256) and --no-deps (nothing outside it)"
                )
                assert not positionals, f"{where}: extra packages beside the lock"
                assert not _PIP_SOURCE_OPTIONS & options.keys(), (
                    f"{where}: the lock must resolve from the default index only"
                )
                sequence[name].append("lock")
            elif positionals == ["pip"] and set(options) == {"--upgrade"}:
                sequence[name].append("self-upgrade")
            elif len(positionals) == 1 and re.match(r"^torch\b", positionals[0]):
                assert options == {"--index-url": "${TORCH_INDEX_URL}"}, (
                    f"{where}: torch must come from the TORCH_INDEX_URL build-arg alone"
                )
                sequence[name].append("torch")
            else:
                raise AssertionError(
                    f"{where}: unsanctioned `pip install` — third-party code must come "
                    f"from the hash-pinned {_API_LOCK_IN_CONTEXT}, never an ad-hoc resolve"
                )

    builder_steps = [s for s in sequence["builder"] if s != "self-upgrade"]
    assert builder_steps == ["torch", "lock", "check"], (
        f"builder must install torch, then the hashed lock, then `pip check`: {builder_steps}"
    )
    assert sequence["runtime"] == ["wheel-install", "check"], (
        "runtime must install only the application wheel and then `pip check` "
        f"(fails the build if the lock no longer satisfies pyproject.toml): {sequence['runtime']}"
    )
    assert sum(steps.count("wheel") for steps in sequence.values()) == 1, (
        "exactly one stage must build the application wheel"
    )


@pytest.mark.security
def test_api_lock_is_fully_pinned_hashed_and_pypi_only(api_lock: str) -> None:
    lines = _lock_requirement_lines(api_lock)
    assert lines, f"{_API_LOCK_IN_CONTEXT} is empty"
    for line in lines:
        assert not line.startswith("-"), (
            f"the lock carries a pip option line {line.split()[0]!r}: index / "
            "find-links / trusted-host / editable lines let it pull from somewhere "
            "other than PyPI (dependency confusion) or bypass the hash check"
        )
        assert "://" not in line and " @ " not in line, (
            f"the lock carries a URL / VCS / path requirement: {line[:80]!r}"
        )
        match = _LOCK_PIN_RE.match(line)
        assert match, f"lock entry is not an exact `==` pin: {line[:80]!r}"
        assert _LOCK_HASH_RE.search(line), (
            f"lock entry {match.group(1)}=={match.group(2)} carries no sha256 hash — "
            "pip --require-hashes would refuse the whole install"
        )
    leaked = sorted(
        name
        for name in _lock_pins(api_lock)
        if name in _INDEX_RESOLVED_PACKAGES or name.startswith("nvidia-") or name == "triton"
    )
    assert not leaked, (
        f"the lock pins {leaked}: torch comes from the TORCH_INDEX_URL build-arg, and "
        "nvidia-*/triton mean it was compiled against the CUDA torch (GBs of GPU "
        "runtime in the CPU image). Recompile with `--torch-backend cpu "
        "--no-emit-package torch` (see the lock header)"
    )


@pytest.mark.security
def test_api_lock_satisfies_every_dependency_the_image_declares(
    api_lock: str, dockerfile: str
) -> None:
    # A stale lock silently overrides pyproject.toml: raising a lower bound, or
    # excluding a bad release, does nothing to the image until the lock is
    # recompiled. The motivating case is `fastapi>=0.115,!=0.136.3` — 0.136.3
    # is the compromised release (MAL-2026-4750); a lock pinning it would ship
    # the malware no matter what pyproject.toml says.
    project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    declared = [Requirement(spec) for spec in project["dependencies"]]
    for extra in _IMAGE_EXTRAS:
        declared += [Requirement(spec) for spec in project["optional-dependencies"][extra]]

    pins = _lock_pins(api_lock)
    problems: list[str] = []
    for requirement in declared:
        name = canonicalize_name(requirement.name)
        if name in _INDEX_RESOLVED_PACKAGES:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate(_IMAGE_MARKER_ENV):
            continue
        pinned = pins.get(name)
        if pinned is None:
            problems.append(f"{requirement} — not in the lock")
        elif not requirement.specifier.contains(Version(pinned), prereleases=True):
            problems.append(f"{requirement} — the lock pins {pinned}")
    assert not problems, (
        f"{_API_LOCK_IN_CONTEXT} no longer satisfies pyproject.toml: {problems}. "
        "Recompile it with the command in its header, in the same commit"
    )

    # torch is the one declared dependency outside the lock: the Dockerfile's
    # index-resolved install must carry pyproject.toml's specifier verbatim.
    declared_torch = {r.specifier for r in declared if canonicalize_name(r.name) == "torch"}
    installed_torch = {
        Requirement(positional).specifier
        for command in _pip_commands(_dockerfile_stages(dockerfile)["builder"])
        for positional in _parse_pip(command)[2]
        if re.match(r"^torch\b", positional)
    }
    assert declared_torch and installed_torch == declared_torch, (
        f"Dockerfile installs torch{sorted(map(str, installed_torch))} but pyproject.toml "
        f"declares torch{sorted(map(str, declared_torch))} — keep them in lockstep"
    )


def test_api_lock_is_not_git_ignored() -> None:
    # `*.txt` is ignored repo-wide, and `gcloud builds submit` derives its upload
    # set from .gitignore (the repo has no .gcloudignore): without the
    # `!deploy/requirements.txt` carve-out Cloud Build never receives the lock —
    # even if the file was force-added to git. --no-index tests the rules, not
    # the index.
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [git, "check-ignore", "-q", "--no-index", _API_LOCK_IN_CONTEXT],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode == 128:
        pytest.skip("not a git work tree")
    assert result.returncode == 1, (
        f"{_API_LOCK_IN_CONTEXT} matches a .gitignore rule — neither git nor "
        "`gcloud builds submit` would carry it; restore the carve-out"
    )


def test_dockerignore_keeps_nested_bytecode_out_of_the_context(dockerignore: str) -> None:
    # A bare `__pycache__/` matches the context root only; src/**/__pycache__
    # would ride along and every local test run would rewrite the `COPY src`
    # input, busting its layer cache without a source edit.
    entries = {line.strip().rstrip("/") for line in dockerignore.splitlines()}
    for pattern in ("**/__pycache__", "**/*.py[cod]"):
        assert pattern in entries, f".dockerignore does not exclude nested {pattern!r}"


# ==========================================================================
# Frontend image (deploy/Dockerfile.frontend) — the frontend-image-keyless
# half of the deployment lockers.
# ===========================================================================


def test_frontend_artifacts_exist() -> None:
    assert _DOCKERFILE_FRONTEND.is_file(), "deploy/Dockerfile.frontend is missing"
    assert _FRONTEND_DOCKERIGNORE.is_file(), "frontend/.dockerignore is missing"


@pytest.mark.security
def test_frontend_every_base_image_is_digest_pinned(frontend_dockerfile: str) -> None:
    base_images = _collect_base_images(frontend_dockerfile)
    assert base_images, "no FROM instruction found in deploy/Dockerfile.frontend"
    floating = [image for image in base_images if not _DIGEST_RE.search(image)]
    assert not floating, (
        f"frontend base image(s) not digest-pinned (@sha256:...): {floating}. "
        "A floating :tag is a silent-substitution / supply-chain vector."
    )


@pytest.mark.security
def test_frontend_runs_non_root(frontend_dockerfile: str) -> None:
    # The frontend is stateless (no mounted data volume), so there is no gosu
    # chown dance — the image must instead drop to a non-root user with a plain
    # `USER` directive, and the LAST such directive must not be root.
    user_lines = re.findall(r"^USER\s+(\S+)", frontend_dockerfile, re.MULTILINE)
    assert user_lines, "frontend Dockerfile never sets a non-root USER"
    assert user_lines[-1] != "root", "frontend Dockerfile's final USER is root"
    assert user_lines[-1] not in {"0", "0:0"}, "frontend Dockerfile's final USER is uid 0"


@pytest.mark.security
def test_frontend_image_is_keyless(frontend_dockerfile: str) -> None:
    # The frontend bakes NO secret. The operator/admin keys live only in the
    # Next server-side admin-session Map (minted at runtime via WelcomeGate);
    # the only runtime env knob is SEC_API_BASE_URL, and even that is supplied
    # at run time, never assigned a literal in the image.
    lowered = frontend_dockerfile.lower()
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), (
        "frontend Dockerfile contains an sk- API-key-shaped token"
    )
    for needle in ("bearer ", "authorization:", "private key"):
        assert needle not in lowered, (
            f"frontend image definition contains secret-shaped material: {needle!r}"
        )
    # No secret-bearing env knob may be ASSIGNED a literal value. A bare
    # `NAME=` or a `NAME=${VAR}` indirection is fine; `NAME=<value>` is a leak.
    # NEXT_PUBLIC_* is doubly forbidden a key — it would ship to the browser.
    secret_env = (
        "API_KEY",
        "API_ADMIN_KEY",
        "API_AUTH_PEPPER",
        "DB_ENCRYPTION_KEY",
        "SEC_API_BASE_URL",
        "NEXT_PUBLIC_API_KEY",
        "NEXT_PUBLIC_ADMIN_KEY",
        "NEXT_PUBLIC_SEC_API_BASE_URL",
    )
    for name in secret_env:
        bad = re.search(rf"\b{name}=(?!\s|$|\$)\S", frontend_dockerfile)
        assert not bad, f"frontend image assigns a literal value to {name} — never bake a key / URL"
    # Belt-and-braces: the admin key must NEVER be shipped to the browser via a
    # NEXT_PUBLIC_ env var of any name.
    assert not re.search(r"\bNEXT_PUBLIC_\w*(?:ADMIN|API_KEY|PEPPER)", frontend_dockerfile), (
        "frontend image exposes a key via a NEXT_PUBLIC_* env var — reaches the browser"
    )


@pytest.mark.security
def test_frontend_does_not_copy_whole_context(frontend_dockerfile: str) -> None:
    # `COPY . .` / `ADD . .` would drag the entire context (incl. a stray
    # `.env`) into a build layer. Selective COPYs only; `.dockerignore` is a
    # backstop, not the only line of defence.
    for line in frontend_dockerfile.splitlines():
        stripped = line.strip()
        assert not re.match(r"^(COPY|ADD)\s+\.(\s|/|$)", stripped), (
            f"frontend image copies the whole build context: {stripped!r}. "
            "Copy explicit paths only."
        )


@pytest.mark.security
def test_frontend_install_is_frozen_lockfile(frontend_dockerfile: str) -> None:
    # A non-frozen `pnpm install` would let the image drift from the committed
    # lockfile — a supply-chain surface. Corepack provisions the pinned pnpm.
    assert "corepack" in frontend_dockerfile.lower(), (
        "frontend image must provision pnpm via Corepack (pinned packageManager)"
    )
    assert re.search(r"pnpm install\s+--frozen-lockfile", frontend_dockerfile), (
        "frontend image must run `pnpm install --frozen-lockfile`"
    )


@pytest.mark.security
def test_frontend_uses_standalone_output(frontend_dockerfile: str, next_config: str) -> None:
    # The runtime stage must ship Next's standalone server (no source tree, no
    # dev node_modules, no pnpm at runtime). That requires both the build-time
    # config opt-in AND the runtime COPY of the traced tree.
    assert re.search(r'output:\s*["\']standalone["\']', next_config), (
        "next.config.ts must set output: 'standalone' for the container image"
    )
    assert ".next/standalone" in frontend_dockerfile, (
        "frontend runtime stage does not copy Next's .next/standalone output"
    )
    assert ".next/static" in frontend_dockerfile, (
        "frontend runtime stage does not copy .next/static (the tracer omits it)"
    )


def test_frontend_cmd_runs_standalone_server(frontend_dockerfile: str) -> None:
    flat = re.sub(r"\\\s*\n", " ", frontend_dockerfile)
    cmd = next((ln for ln in flat.splitlines() if ln.strip().startswith("CMD")), "")
    assert "server.js" in cmd, "frontend CMD does not launch the standalone server.js"
    assert "HEALTHCHECK" in frontend_dockerfile, "frontend Dockerfile has no HEALTHCHECK"


@pytest.mark.security
def test_frontend_dockerignore_excludes_secrets_and_state(frontend_dockerignore: str) -> None:
    entries = {line.strip() for line in frontend_dockerignore.splitlines() if line.strip()}
    required = {
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "node_modules/",
        ".next/",
        ".git/",
    }
    missing = sorted(required - entries)
    assert not missing, f"frontend/.dockerignore is missing secret/state exclusions: {missing}"


# ==========================================================================
# Compose + nginx portable stack (deploy/docker-compose.yml, deploy/nginx/
# nginx.conf) — the supply-chain + runtime contract for the whole deployment, including
# the in-process TaskManager contract and the edge-proxy contract. The compose file is
# the single source of truth for the deployment's supply chain (digest-pinned bases)
# and runtime configuration (replica count, env knobs, volume mounts); the nginx
# conf is the single source of truth for the edge proxy's routing and TLS contract.
# ===========================================================================


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compose_text() -> str:
    return _COMPOSE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def nginx_conf() -> str:
    return _NGINX_CONF.read_text(encoding="utf-8")


# A secret-bearing env knob assigned a literal value in the compose file is a
# leak. A `${VAR}` interpolation or a `*_FILE` path indirection is fine.
_SECRET_ENV = (
    "API_KEY",
    "API_ADMIN_KEY",
    "API_AUTH_PEPPER",
    "DB_ENCRYPTION_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HUGGING_FACE_TOKEN",
    "HF_TOKEN",
)


def test_compose_artifacts_exist() -> None:
    assert _COMPOSE.is_file(), "deploy/docker-compose.yml is missing"
    assert _NGINX_CONF.is_file(), "deploy/nginx/nginx.conf is missing"


def _services(compose: dict[str, Any]) -> dict[str, Any]:
    services = compose.get("services", {})
    assert {"api", "frontend", "nginx"} <= set(services), (
        f"compose must define api + frontend + nginx services; got {sorted(services)}"
    )
    return services


# ---------------------------------------------------------------------------
# Supply chain: the third-party (nginx) base image is digest-pinned.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_compose_nginx_image_is_digest_pinned(compose: dict[str, Any]) -> None:
    image = _services(compose)["nginx"].get("image", "")
    assert _DIGEST_RE.search(image), (
        f"nginx image is not digest-pinned (@sha256:...): {image!r}. "
        "A floating :tag is a silent-substitution / supply-chain vector."
    )


# ---------------------------------------------------------------------------
# In-process TaskManager: exactly one api replica.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_compose_api_is_single_replica(compose: dict[str, Any]) -> None:
    api = _services(compose)["api"]
    replicas = api.get("deploy", {}).get("replicas", 1)
    assert replicas == 1, (
        f"api must run exactly one replica (in-process TaskManager mints task "
        f"IDs in-process; a second replica orphans them); got replicas={replicas}"
    )
    # `scale:` is the deprecated knob for the same thing — pin it too.
    assert "scale" not in api or api["scale"] == 1, "api must not scale beyond one instance"


# ---------------------------------------------------------------------------
# Trusted-proxy set: bounded, never "*", and in lockstep with the pinned
# network subnet (M2).
# ---------------------------------------------------------------------------


def _parse_networks(value: str, label: str) -> set[Any]:
    """Parse a comma-separated CIDR list, failing with a readable message.

    A bare ``ipaddress.ip_network`` raises ``ValueError`` on a wildcard or a
    typo, which surfaces as an unreadable traceback rather than a locker
    message naming the offending artefact.
    """
    out = set()
    for part in value.split(","):
        part = part.strip()
        try:
            out.add(ipaddress.ip_network(part))
        except ValueError as exc:
            raise AssertionError(
                f"{label} entry {part!r} is not a valid CIDR ({exc}). It must be a "
                "bounded network — uvicorn silently files an unparseable value under "
                "`trusted_literals`, where it matches no peer at all."
            ) from None
    return out


def _api_forwarded_allow_ips(compose: dict[str, Any]) -> str:
    env = _services(compose)["api"].get("environment", {})
    # compose accepts both the mapping and the "KEY=value" list form.
    if isinstance(env, list):
        env = dict(item.split("=", 1) for item in env if isinstance(item, str) and "=" in item)
    value = env.get("FORWARDED_ALLOW_IPS")
    assert value, (
        "api must set FORWARDED_ALLOW_IPS. Unset, uvicorn falls back to 127.0.0.1 "
        "and every client collapses onto nginx's address — safe, but per-IP rate "
        "limiting stops discriminating between tenants (DEPLOYMENT.md 4.20.1)"
    )
    return str(value).strip()


@pytest.mark.security
def test_compose_trusted_proxy_set_is_bounded(compose: dict[str, Any]) -> None:
    # M2. With "*" uvicorn reads the LEFTMOST X-Forwarded-For entry, which
    # nginx leaves client-controlled (it appends the real peer to the right),
    # so any caller rotates the per-IP rate-limit key at will. Runtime proof of
    # both halves: tests/api/test_search.py::TestForwardedForRateLimitKeying.
    value = _api_forwarded_allow_ips(compose)
    assert value != "*" and "*" not in value, (
        f"FORWARDED_ALLOW_IPS must be a bounded CIDR, never a wildcard; got {value!r}"
    )
    # A bounded value must parse as real network(s) — a hostname or a typo
    # would silently land in uvicorn's `trusted_literals` and match nothing.
    _parse_networks(value, "FORWARDED_ALLOW_IPS")


@pytest.mark.security
def test_compose_trusted_proxy_set_matches_the_pinned_subnet(
    compose: dict[str, Any],
) -> None:
    # The two values are a pair: FORWARDED_ALLOW_IPS names exactly the peers
    # that can reach the api service, which is the pinned `sec_gs` subnet
    # (nginx + the Next frontend proxy). Drift between them either over-trusts
    # (a wider CIDR re-opens spoofing from any co-located peer) or breaks the
    # real-client lookup entirely. Pinning the subnet is what makes the
    # bounded set expressible at all — a Docker-assigned range is unpredictable.
    network = compose.get("networks", {}).get("sec_gs") or {}
    configs = (network.get("ipam") or {}).get("config") or []
    subnets = [c.get("subnet") for c in configs if isinstance(c, dict) and c.get("subnet")]
    assert subnets, (
        "the sec_gs network must pin an ipam subnet so FORWARDED_ALLOW_IPS can "
        "name exactly the trusted peers"
    )
    declared = _parse_networks(",".join(subnets), "sec_gs ipam subnet")
    trusted = _parse_networks(_api_forwarded_allow_ips(compose), "FORWARDED_ALLOW_IPS")
    assert trusted == declared, (
        f"FORWARDED_ALLOW_IPS {sorted(map(str, trusted))} has drifted from the pinned "
        f"sec_gs subnet {sorted(map(str, declared))} — change the two in lockstep"
    )


# ---------------------------------------------------------------------------
# Only the edge publishes host ports; api + frontend stay internal.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_compose_only_nginx_publishes_ports(compose: dict[str, Any]) -> None:
    services = _services(compose)
    for name in ("api", "frontend"):
        assert not services[name].get("ports"), (
            f"{name} publishes host ports — the browser must reach the backend only "
            "through the Next proxy; only nginx should publish ports."
        )
    assert services["nginx"].get("ports"), "nginx publishes no host ports — nothing is reachable"


# ---------------------------------------------------------------------------
# No baked secret; secret-bearing knobs only via *_FILE / interpolation.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_compose_bakes_no_secret(compose_text: str) -> None:
    lowered = compose_text.lower()
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), (
        "docker-compose.yml contains an sk- API-key-shaped token"
    )
    for needle in ("bearer ", "authorization:", "private key"):
        assert needle not in lowered, f"compose contains secret-shaped material: {needle!r}"
    # No secret-bearing env knob may be ASSIGNED a literal value. Allow the
    # `*_FILE` path indirection (DB_ENCRYPTION_KEY_FILE: /run/secrets/...) and a
    # `${VAR}` interpolation; a bare `NAME: <value>` / `NAME=<value>` is a leak.
    for name in _SECRET_ENV:
        bad = re.search(rf"^\s*-?\s*{name}\s*[:=]\s*(?!\s|$|\$|/run/secrets)\S", compose_text, re.M)
        assert not bad, (
            f"compose assigns a literal value to {name} — supply it at runtime via "
            "env_file / a mounted *_FILE secret, never inline."
        )


@pytest.mark.security
def test_compose_api_uses_secret_file_indirection(compose: dict[str, Any]) -> None:
    api = _services(compose)["api"]
    env = api.get("environment", {}) or {}
    # The encryption key + auth pepper must arrive via a mounted file, not a
    # literal env value — so they live in /run/secrets, never in the image or
    # the process env table as cleartext.
    for knob in ("DB_ENCRYPTION_KEY_FILE", "API_AUTH_PEPPER_FILE"):
        value = env.get(knob, "")
        assert value.startswith("/run/secrets/"), (
            f"api {knob} must point at a mounted Docker-secret under /run/secrets/; got {value!r}"
        )
    # And those files must be wired as Docker secrets (read-only /run/secrets
    # mount that fails loudly when the source file is absent).
    secret_refs = {s if isinstance(s, str) else s.get("source") for s in (api.get("secrets") or [])}
    assert {"db_encryption_key", "api_auth_pepper"} <= secret_refs, (
        f"api must mount db_encryption_key + api_auth_pepper as Docker secrets; got {secret_refs}"
    )


@pytest.mark.security
def test_compose_top_level_secrets_are_file_sourced(compose: dict[str, Any]) -> None:
    secrets = compose.get("secrets", {})
    for name in ("db_encryption_key", "api_auth_pepper"):
        spec = secrets.get(name, {})
        assert "file" in spec, f"secret {name} must be file-sourced (operator-supplied), not inline"
        assert "environment" not in spec, (
            f"secret {name} must not be sourced from an env var (would bake into the process table)"
        )


@pytest.mark.security
def test_compose_frontend_is_keyless(compose: dict[str, Any]) -> None:
    fe = _services(compose)["frontend"]
    env = fe.get("environment", {}) or {}
    # The only runtime knob is the server-side backend base URL. No key, and
    # crucially no NEXT_PUBLIC_* (that would ship to the browser).
    for key in env:
        assert not key.startswith("NEXT_PUBLIC_"), (
            f"frontend sets {key} — a NEXT_PUBLIC_* var reaches the browser bundle"
        )
    assert "SEC_API_BASE_URL" in env, "frontend must point at the backend via SEC_API_BASE_URL"
    # The base URL is server-side; it must NOT be advertised through a public var.
    assert "NEXT_PUBLIC_SEC_API_BASE_URL" not in env, (
        "SEC_API_BASE_URL must stay server-side, never NEXT_PUBLIC_*"
    )


@pytest.mark.security
def test_compose_api_persists_data_on_a_volume(compose: dict[str, Any]) -> None:
    api = _services(compose)["api"]
    targets = []
    for vol in api.get("volumes", []) or []:
        if isinstance(vol, str):
            targets.append(vol.split(":")[1] if ":" in vol else vol)
        else:
            targets.append(vol.get("target"))
    assert "/app/data" in targets, (
        "api must mount a durable volume at /app/data — task_history is the only "
        "crash-durable ingest record (§4.7.quinquies)"
    )


# ---------------------------------------------------------------------------
# .gitignore re-ignores the operator secret / TLS material under deploy/.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_gitignore_reignores_deploy_secrets() -> None:
    gitignore = _GITIGNORE.read_text(encoding="utf-8")
    carve_out = gitignore.index("!deploy/")
    for pattern in ("deploy/secrets/", "deploy/certs/"):
        idx = gitignore.find(pattern)
        assert idx != -1, f".gitignore must re-ignore {pattern} (operator secret/TLS material)"
        assert idx > carve_out, (
            f"{pattern} re-ignore must come AFTER the !deploy/ carve-out or it has no effect"
        )


# ==========================================================================
# nginx reverse proxy — routing + TLS only; owns NO CSP.
# ==========================================================================

# Security / CSP headers nginx MUST NOT inject — the Next middleware owns the
# full set (a second, un-nonced policy here would fight or weaken it).
_FORBIDDEN_NGINX_HEADERS = (
    "content-security-policy",
    "strict-transport-security",
    "x-frame-options",
    "x-content-type-options",
    "referrer-policy",
    "permissions-policy",
)


def _nginx_add_headers(nginx_conf: str) -> list[str]:
    """Lower-cased header names from every (uncommented) add_header directive."""
    names: list[str] = []
    for raw in nginx_conf.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        match = re.match(r"add_header\s+([A-Za-z0-9-]+)", line)
        if match:
            names.append(match.group(1).lower())
    return names


def test_nginx_conf_exists() -> None:
    assert _NGINX_CONF.is_file(), "deploy/nginx/nginx.conf is missing"


@pytest.mark.security
def test_nginx_sets_no_csp_or_security_headers(nginx_conf: str) -> None:
    emitted = _nginx_add_headers(nginx_conf)
    leaked = sorted(set(emitted) & set(_FORBIDDEN_NGINX_HEADERS))
    assert not leaked, (
        f"nginx injects security/CSP header(s) {leaked} — the Next.js middleware is the "
        "single source of truth for CSP + the security-header set. nginx does routing only."
    )


@pytest.mark.security
def test_nginx_admin_proxy_routes_to_frontend_before_api(nginx_conf: str) -> None:
    # The browser reaches the backend ONLY through the Next admin proxy at
    # /api/admin/*. That route lives on the frontend (it injects the server-held
    # keys), so its location MUST proxy to the frontend AND be declared before the
    # bare /api/ location so longest-prefix-wins keeps it ahead.
    admin_idx = nginx_conf.find("location /api/admin/")
    api_idx = nginx_conf.find("location /api/ ")
    assert admin_idx != -1, "nginx has no `location /api/admin/` — SPA admin proxy unreachable"
    assert api_idx != -1, "nginx has no `location /api/` for direct API consumers"
    assert admin_idx < api_idx, (
        "`location /api/admin/` must precede `location /api/` so the admin proxy wins"
    )
    # The admin-proxy block targets the frontend upstream; the bare /api/ block
    # targets the api upstream.
    admin_block = nginx_conf[admin_idx:api_idx]
    assert "proxy_pass http://frontend" in admin_block, (
        "`location /api/admin/` must proxy to the frontend (the key-injecting Next proxy)"
    )
    api_block = nginx_conf[api_idx : api_idx + 600]
    assert "proxy_pass http://api" in api_block, "`location /api/` must proxy to the api upstream"


@pytest.mark.security
def test_nginx_websocket_upgrade_to_api(nginx_conf: str) -> None:
    ws_idx = nginx_conf.find("location /ws/")
    assert ws_idx != -1, "nginx has no `location /ws/` for the ingest-progress WebSocket"
    ws_block = nginx_conf[ws_idx : ws_idx + 600]
    assert "proxy_pass http://api" in ws_block, "`location /ws/` must proxy to the api upstream"
    assert re.search(r"proxy_set_header\s+Upgrade\s+\$http_upgrade", ws_block), (
        "WebSocket location must forward the Upgrade header"
    )
    assert re.search(r"proxy_set_header\s+Connection\s+\$connection_upgrade", ws_block), (
        "WebSocket location must set Connection: upgrade"
    )
    # The backend's WS authorisation surface is the Origin allow-list — nginx
    # must forward Origin intact for that check to run.
    assert re.search(r"proxy_set_header\s+Origin\s+\$http_origin", ws_block), (
        "WebSocket location must forward the Origin header (backend WS auth surface)"
    )


def test_nginx_terminates_tls(nginx_conf: str) -> None:
    assert "listen 443 ssl" in nginx_conf, "nginx does not terminate TLS on :443"
    assert "ssl_certificate" in nginx_conf, "nginx has no TLS certificate directive"


@pytest.mark.security
def test_nginx_no_baked_secret(nginx_conf: str) -> None:
    lowered = nginx_conf.lower()
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), "nginx.conf contains an sk- API-key token"
    assert "private key" not in lowered, "nginx.conf contains secret-shaped material"


# ==========================================================================
# GCP Cloud Run manifests (deploy/cloud/{api,frontend}-service.yaml,
# demo-reset-job.yaml) — the Cloud Run counterparts of the Compose stack and
# carry the same load-bearing contracts, expressed in Knative annotations:
#
#   - the in-process TaskManager single-instance contract (maxScale=1, no
#     scale-to-zero, no CPU throttling between requests);
#   - Secret Manager indirection for every secret-bearing knob (the Cloud Run
#     analogue of the Compose *_FILE / Docker-secret mounts);
#   - a keyless, GFE-TLS frontend service whose only env knob is the server-side
#     SEC_API_BASE_URL (no NEXT_PUBLIC_*, no secret);
#   - an internal-ingress API the browser can never reach directly.
#
# Like every other locker here, the assertions are on tracked, CI-visible
# files and need no gcloud / network / Docker daemon.
# ===========================================================================

_KNATIVE_SERVICE_KIND = "Service"
_CLOUD_RUN_JOB_KIND = "Job"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _service_container(doc: dict[str, Any]) -> dict[str, Any]:
    """The first container of a Knative ``Service`` revision template."""
    containers = doc["spec"]["template"]["spec"]["containers"]
    assert containers, "Knative Service defines no container"
    return containers[0]


def _job_container(doc: dict[str, Any]) -> dict[str, Any]:
    """The first container of a Cloud Run ``Job`` task template.

    The nesting is Job → ExecutionTemplate (``spec.template``) → TaskTemplate
    (``.spec.template``) → ``.spec.containers``.
    """
    containers = doc["spec"]["template"]["spec"]["template"]["spec"]["containers"]
    assert containers, "Cloud Run Job defines no container"
    return containers[0]


def _template_annotations(doc: dict[str, Any]) -> dict[str, str]:
    return doc["spec"]["template"]["metadata"].get("annotations", {}) or {}


def _env_by_name(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in (container.get("env") or [])}


def _uses_secret_manager(entry: dict[str, Any]) -> bool:
    """True iff the env entry resolves from Secret Manager (no inline value)."""
    if "value" in entry:
        return False
    return "secretKeyRef" in (entry.get("valueFrom") or {})


@pytest.fixture(scope="module")
def cloud_api() -> dict[str, Any]:
    return _load_yaml(_CLOUD_API)


@pytest.fixture(scope="module")
def cloud_frontend() -> dict[str, Any]:
    return _load_yaml(_CLOUD_FRONTEND)


@pytest.fixture(scope="module")
def cloud_job() -> dict[str, Any]:
    return _load_yaml(_CLOUD_JOB)


def test_cloud_artifacts_exist() -> None:
    assert _CLOUD_API.is_file(), "deploy/cloud/api-service.yaml is missing"
    assert _CLOUD_FRONTEND.is_file(), "deploy/cloud/frontend-service.yaml is missing"
    assert _CLOUD_JOB.is_file(), "deploy/cloud/demo-reset-job.yaml is missing"


def test_cloud_services_have_expected_kinds(
    cloud_api: dict[str, Any], cloud_frontend: dict[str, Any], cloud_job: dict[str, Any]
) -> None:
    assert cloud_api["kind"] == _KNATIVE_SERVICE_KIND, "api-service.yaml must be a Knative Service"
    assert cloud_frontend["kind"] == _KNATIVE_SERVICE_KIND, (
        "frontend-service.yaml must be a Knative Service"
    )
    assert cloud_job["kind"] == _CLOUD_RUN_JOB_KIND, "demo-reset-job.yaml must be a Cloud Run Job"


# ---------------------------------------------------------------------------
# In-process TaskManager: exactly one API instance, never scaled to zero.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_cloud_api_is_single_instance(cloud_api: dict[str, Any]) -> None:
    ann = _template_annotations(cloud_api)
    max_scale = ann.get("autoscaling.knative.dev/maxScale")
    assert max_scale == "1", (
        f"api maxScale must be '1' (the in-process TaskManager mints task IDs "
        f"in-process; a second instance orphans them); got {max_scale!r}"
    )


@pytest.mark.security
def test_cloud_api_never_scales_to_zero(cloud_api: dict[str, Any]) -> None:
    # Scale-to-zero or CPU-throttling between requests kills the daemon-thread
    # ingest workers mid-flight (DEPLOYMENT §4.7.quinquies).
    ann = _template_annotations(cloud_api)
    min_scale = ann.get("autoscaling.knative.dev/minScale")
    assert min_scale is not None and int(min_scale) >= 1, (
        f"api minScale must be >= 1 — scale-to-zero kills in-flight ingest "
        f"workers; got {min_scale!r}"
    )
    throttle = ann.get("run.googleapis.com/cpu-throttling")
    assert throttle == "false", (
        f"api cpu-throttling must be 'false' so background worker threads keep "
        f"running between requests; got {throttle!r}"
    )


@pytest.mark.security
def test_cloud_api_ingress_is_internal(cloud_api: dict[str, Any]) -> None:
    # The browser must reach the backend only through the keyless Next admin
    # proxy on the frontend service — never the API directly.
    ingress = cloud_api["metadata"].get("annotations", {}).get("run.googleapis.com/ingress")
    assert ingress == "internal", (
        f"api ingress must be 'internal' (browser reaches it only via the Next "
        f"proxy on the frontend); got {ingress!r}"
    )


# ---------------------------------------------------------------------------
# No baked secret; secret-bearing knobs only via Secret Manager.
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.parametrize("path", [_CLOUD_API, _CLOUD_FRONTEND, _CLOUD_JOB], ids=lambda p: p.name)
def test_cloud_manifest_bakes_no_secret(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), (
        f"{path.name} contains an sk- API-key-shaped token"
    )
    for needle in ("bearer ", "authorization:", "private key", "begin certificate"):
        assert needle not in lowered, f"{path.name} contains secret-shaped material: {needle!r}"
    # No secret-bearing knob may be ASSIGNED a literal `value:`. The Secret
    # Manager indirection (valueFrom.secretKeyRef) carries no literal here, so a
    # `<NAME>` ... `value: <x>` pairing on adjacent lines is the leak shape.
    for name in _SECRET_ENV:
        bad = re.search(rf"name:\s*{name}\b[^\n]*\n\s*value:\s*\S", text)
        assert not bad, (
            f"{path.name} assigns a literal `value:` to {name} — resolve it from "
            "Secret Manager via valueFrom.secretKeyRef, never inline."
        )


@pytest.mark.security
def test_cloud_api_secret_knobs_use_secret_manager(cloud_api: dict[str, Any]) -> None:
    env = _env_by_name(_service_container(cloud_api))
    present_secrets = [name for name in _SECRET_ENV if name in env]
    # Sanity: the cloud API actually configures the encryption key + pepper, so
    # the assertion below is never vacuous.
    assert {"DB_ENCRYPTION_KEY", "API_AUTH_PEPPER"} <= set(present_secrets), (
        f"cloud API must configure DB_ENCRYPTION_KEY + API_AUTH_PEPPER; got {present_secrets}"
    )
    for name in present_secrets:
        assert _uses_secret_manager(env[name]), (
            f"api {name} must resolve from Secret Manager (valueFrom.secretKeyRef), "
            f"never an inline value; got {env[name]!r}"
        )


@pytest.mark.security
def test_cloud_api_persists_data_via_gcsfuse(cloud_api: dict[str, Any]) -> None:
    container = _service_container(cloud_api)
    mounts = {m.get("mountPath") for m in (container.get("volumeMounts") or [])}
    assert "/app/data" in mounts, (
        "api must mount a durable volume at /app/data — task_history is the only "
        "crash-durable ingest record (§4.7.quinquies)"
    )
    volumes = cloud_api["spec"]["template"]["spec"].get("volumes") or []
    drivers = {v.get("csi", {}).get("driver") for v in volumes}
    assert "gcsfuse.run.googleapis.com" in drivers, (
        "api /app/data must be backed by the GCS FUSE CSI driver on Cloud Run "
        f"(no persistent local disk); got volume drivers {drivers}"
    )


def test_cloud_api_hf_cache_is_not_on_the_fuse_volume(cloud_api: dict[str, Any]) -> None:
    # F25: the image default HF_HOME is on /app/data, which on Cloud Run is
    # gcsfuse — memory-mapping ~0.6 GB of weights over FUSE on every cold
    # start. The manifest must pin the cache to container-local disk.
    container = _service_container(cloud_api)
    env = _env_by_name(container)
    assert "HF_HOME" in env and "value" in env["HF_HOME"], (
        "api-service.yaml must set HF_HOME explicitly — the image default "
        "(/app/data/hf) is on the gcsfuse volume"
    )
    hf_home = str(env["HF_HOME"]["value"])
    for mount in container.get("volumeMounts") or []:
        path = str(mount.get("mountPath", "")).rstrip("/")
        assert hf_home != path and not hf_home.startswith(path + "/"), (
            f"HF_HOME={hf_home!r} sits on the {path!r} volume mount (gcsfuse)"
        )


@pytest.mark.security
def test_cloud_api_local_embedder_is_warmed_with_a_secret_managed_token(
    cloud_api: dict[str, Any],
) -> None:
    env = _env_by_name(_service_container(cloud_api))
    if env.get("EMBEDDING_PROVIDER", {}).get("value") != "local":
        pytest.skip("cloud API does not use the local embedder")
    # google/embeddinggemma-300m is gated: no token, no weights.
    assert "HF_TOKEN" in env, "local (gated) embedder configured without HF_TOKEN"
    assert _uses_secret_manager(env["HF_TOKEN"]), (
        "HF_TOKEN must resolve from Secret Manager (valueFrom.secretKeyRef)"
    )
    # Every new instance downloads the weights; warm them before the startup
    # probe passes rather than inside the first user request.
    assert str(env.get("EMBEDDING_WARM_ON_BOOT", {}).get("value", "")).lower() == "true", (
        "api-service.yaml must set EMBEDDING_WARM_ON_BOOT=true (fresh instances "
        "re-download the embedder)"
    )


# ---------------------------------------------------------------------------
# Keyless, GFE-TLS frontend service.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_cloud_frontend_is_keyless(cloud_frontend: dict[str, Any]) -> None:
    container = _service_container(cloud_frontend)
    env = _env_by_name(container)
    for name, entry in env.items():
        assert not name.startswith("NEXT_PUBLIC_"), (
            f"frontend sets {name} — a NEXT_PUBLIC_* var reaches the browser bundle"
        )
        assert name not in _SECRET_ENV, f"frontend env carries a secret-bearing knob {name}"
        assert "valueFrom" not in entry, (
            f"frontend env {name} resolves from a secret store — the SPA image is keyless"
        )
    assert "SEC_API_BASE_URL" in env, "frontend must point at the backend via SEC_API_BASE_URL"
    assert "NEXT_PUBLIC_SEC_API_BASE_URL" not in env, (
        "SEC_API_BASE_URL must stay server-side, never NEXT_PUBLIC_*"
    )


@pytest.mark.security
def test_cloud_frontend_is_public_gfe_tls(cloud_frontend: dict[str, Any]) -> None:
    # GFE terminates TLS and serves a managed certificate for a public service;
    # the manifest defines NO TLS material of its own, and the service is
    # publicly reachable (ingress: all, or unset which defaults to all).
    ingress = (
        cloud_frontend["metadata"].get("annotations", {}).get("run.googleapis.com/ingress", "all")
    )
    assert ingress == "all", (
        f"frontend must be publicly reachable (ingress 'all') so GFE fronts it "
        f"with managed TLS; got {ingress!r}"
    )
    text = _CLOUD_FRONTEND.read_text(encoding="utf-8").lower()
    for needle in ("ssl_certificate", "tls_cert", "443", "begin private key"):
        assert needle not in text, (
            f"frontend manifest defines TLS material ({needle!r}); GFE owns TLS — "
            "the service must not terminate it"
        )


# ---------------------------------------------------------------------------
# Demo-reset Cloud Run Job.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_cloud_demo_reset_is_a_job(cloud_job: dict[str, Any]) -> None:
    # A Job runs to completion per trigger — never a long-lived Service that
    # would hold the destructive `clear` surface open.
    assert cloud_job["kind"] == _CLOUD_RUN_JOB_KIND, "demo-reset must be a Cloud Run Job"
    container = _job_container(cloud_job)
    # It invokes the CLI (which bypasses API_DEMO_MODE) — the only reset path,
    # since the API blocks `clear` under demo mode.
    args = container.get("args") or []
    assert args[:3] == ["sec-rag", "manage", "clear"], (
        f"demo-reset Job must run `sec-rag manage clear` (the CLI reset path that "
        f"bypasses demo mode); got args {args}"
    )
    # `args` only, no `command`: the image ENTRYPOINT (the gosu drop) stays in
    # force, so the reset runs as the unprivileged appuser, not root.
    assert "command" not in container, (
        "demo-reset Job must not override `command` — keep the image ENTRYPOINT "
        "(docker-entrypoint.sh) so the gosu non-root drop still runs"
    )


@pytest.mark.security
def test_cloud_demo_reset_uses_secret_manager(cloud_job: dict[str, Any]) -> None:
    env = _env_by_name(_job_container(cloud_job))
    assert "DB_ENCRYPTION_KEY" in env, (
        "demo-reset Job needs DB_ENCRYPTION_KEY to open the SQLCipher store"
    )
    assert _uses_secret_manager(env["DB_ENCRYPTION_KEY"]), (
        "demo-reset Job DB_ENCRYPTION_KEY must resolve from Secret Manager "
        "(valueFrom.secretKeyRef), never an inline value"
    )


# ===========================================================================
# Cloud Run deploy workflow + Cloud Build config.
# ===========================================================================
# `.github/workflows/deploy.yml` builds the two images via Cloud Build and
# applies the Knative manifests, authenticating to GCP with KEYLESS Workload
# Identity Federation. Like the manifests it deploys, the workflow is an
# operator artefact whose security properties must not silently regress. These
# static lockers pin:
#
#   - SUPPLY CHAIN. Every `uses:` Action is pinned by a 40-hex commit SHA
#     (mirrors ci.yml); the deployed Cloud Run revision references an immutable
#     @sha256 digest, never a floating `:latest`.
#   - KEYLESS AUTH. WIF only — no `credentials_json` (no long-lived SA key);
#     `id-token: write` is granted for the OIDC exchange.
#   - CI GATE. Deploy fires only after the "CI" workflow concludes successfully.
#   - COMMAND-INJECTION GUARD. No `${{ github.event.* }}` interpolation inside a
#     `run:` shell (mirrors ci.yml's stated principle).
#   - CONTAINMENT. ci.yml carries no GCP credential surface whatsoever — every
#     GCP touch lives in deploy.yml.
#   - BUILD HYGIENE. cloudbuild.yaml builds each image with the correct context
#     (api → `.`, frontend → `frontend/`) and pushes commit-tagged refs.
#
# All assertions are on tracked, CI-visible files; nothing here needs gcloud,
# a network, or a Docker daemon.
# ---------------------------------------------------------------------------

_USES_RE = re.compile(r"uses:\s*(\S+)")
# owner/repo[/path…]@<40 hex>  — a fully SHA-pinned Action reference.
_SHA_PINNED_USES_RE = re.compile(r"^[\w.-]+/[\w.-]+(?:/[\w.-]+)*@[0-9a-f]{40}$")
_RUN_BLOCK_RE = re.compile(r"^(\s*)run:\s*[|>]")


def _run_block_bodies(workflow_text: str) -> list[str]:
    """Return the shell body of every ``run: |`` / ``run: >`` block.

    A block is delimited by indentation: lines indented deeper than the ``run:``
    key belong to the script. Used to assert no ``${{ github.event.* }}`` value
    is interpolated straight into a shell command.
    """
    bodies: list[str] = []
    lines = workflow_text.splitlines()
    index = 0
    while index < len(lines):
        match = _RUN_BLOCK_RE.match(lines[index])
        if not match:
            index += 1
            continue
        indent = len(match.group(1))
        index += 1
        body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() == "":
                body.append(line)
                index += 1
                continue
            if (len(line) - len(line.lstrip())) <= indent:
                break
            body.append(line)
            index += 1
        bodies.append("\n".join(body))
    return bodies


def _cloudbuild_step(cloudbuild: dict[str, Any], step_id: str) -> dict[str, Any]:
    for step in cloudbuild.get("steps", []):
        if step.get("id") == step_id:
            return step
    raise AssertionError(f"cloudbuild.yaml has no step id {step_id!r}")


@pytest.fixture(scope="module")
def deploy_workflow() -> str:
    return _DEPLOY_WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ci_workflow() -> str:
    return _CI_WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cloudbuild() -> dict[str, Any]:
    return yaml.safe_load(_CLOUDBUILD.read_text(encoding="utf-8"))


def test_deploy_workflow_artifacts_exist() -> None:
    assert _DEPLOY_WORKFLOW.is_file(), ".github/workflows/deploy.yml is missing"
    assert _CLOUDBUILD.is_file(), "deploy/cloudbuild.yaml is missing"


@pytest.mark.security
def test_deploy_workflow_actions_are_sha_pinned(deploy_workflow: str) -> None:
    refs = _USES_RE.findall(deploy_workflow)
    assert refs, "deploy.yml declares no `uses:` actions"
    for ref in refs:
        assert _SHA_PINNED_USES_RE.match(ref), (
            f"deploy.yml Action {ref!r} is not pinned by a 40-hex commit SHA — a "
            "floating tag is a silent-substitution vector (mirror ci.yml)"
        )


@pytest.mark.security
def test_deploy_workflow_uses_keyless_wif(deploy_workflow: str) -> None:
    assert "google-github-actions/auth@" in deploy_workflow, (
        "deploy.yml must authenticate via google-github-actions/auth"
    )
    assert "workload_identity_provider:" in deploy_workflow, (
        "deploy.yml must use Workload Identity Federation (workload_identity_provider)"
    )
    assert "service_account:" in deploy_workflow, (
        "deploy.yml WIF auth must name the service_account to impersonate"
    )
    # The entire point of WIF is to NOT store a long-lived SA JSON key.
    assert "credentials_json" not in deploy_workflow, (
        "deploy.yml carries a `credentials_json` input — WIF must be keyless; a "
        "long-lived service-account JSON key is exactly what it eliminates"
    )
    assert re.search(r"id-token:\s*write", deploy_workflow), (
        "deploy.yml must grant `id-token: write` for the OIDC → WIF exchange"
    )


@pytest.mark.security
def test_deploy_workflow_is_gated_on_ci_success(deploy_workflow: str) -> None:
    assert "workflow_run:" in deploy_workflow, (
        "deploy.yml must trigger off the CI workflow (workflow_run), not a raw push"
    )
    assert re.search(r'workflows:\s*\[\s*"CI"\s*\]', deploy_workflow), (
        'deploy.yml workflow_run must key on the "CI" workflow name'
    )
    assert "github.event.workflow_run.conclusion == 'success'" in deploy_workflow, (
        "deploy.yml must deploy ONLY when the gating CI run concluded successfully"
    )


@pytest.mark.security
def test_deploy_workflow_pins_image_digests(deploy_workflow: str) -> None:
    # The deployed revision must reference an immutable @sha256 digest resolved at
    # deploy time — never a floating `:latest` (the supply-chain analogue of the
    # SHA-pinned Actions and the digest-pinned Dockerfile bases).
    assert "image_summary.digest" in deploy_workflow, (
        "deploy.yml must resolve each pushed tag to its immutable @sha256 digest"
    )
    assert re.search(r"API_IMAGE=[^\n]*@\$", deploy_workflow), (
        "deploy.yml must build the API image ref as `…@<digest>`, never `:latest`"
    )
    assert re.search(r"FRONTEND_IMAGE=[^\n]*@\$", deploy_workflow), (
        "deploy.yml must build the frontend image ref as `…@<digest>`, never `:latest`"
    )
    assert "gcloud run services replace" in deploy_workflow, (
        "deploy.yml must apply the api/frontend Services via `gcloud run services replace`"
    )
    assert "gcloud run jobs replace" in deploy_workflow, (
        "deploy.yml must apply the demo-reset Job via `gcloud run jobs replace`"
    )


@pytest.mark.security
def test_deploy_workflow_no_event_interpolation_in_run(deploy_workflow: str) -> None:
    for body in _run_block_bodies(deploy_workflow):
        assert "${{ github.event" not in body, (
            "deploy.yml interpolates a `github.event.*` value directly into a "
            "`run:` shell — route it through an env var (command-injection guard)"
        )


@pytest.mark.security
def test_ci_workflow_stays_gcp_credential_free(ci_workflow: str) -> None:
    # ci.yml must stay free of WIF auth, gcloud, and Secret Manager references.
    lowered = ci_workflow.lower()
    for needle in (
        "google-github-actions",
        "workload_identity_provider",
        "credentials_json",
        "id-token",
        "gcloud",
        "secretmanager",
        "secret-manager",
    ):
        assert needle not in lowered, (
            f"ci.yml references {needle!r} — deployment/GCP credentials must live "
            "only in deploy.yml; ci.yml stays GCP-credential-free"
        )


@pytest.mark.security
def test_cloudbuild_builds_both_images_with_correct_context(cloudbuild: dict[str, Any]) -> None:
    api_args = _cloudbuild_step(cloudbuild, "build-api")["args"]
    frontend_args = _cloudbuild_step(cloudbuild, "build-frontend")["args"]
    # API: Dockerfile deploy/Dockerfile.api, context the repo root `.` (the root
    # .dockerignore applies and excludes .env / data / venv / frontend).
    assert "deploy/Dockerfile.api" in api_args, "api build must use deploy/Dockerfile.api"
    assert api_args[-1] == ".", f"api build context must be the repo root `.`; got {api_args[-1]!r}"
    # Frontend: Dockerfile deploy/Dockerfile.frontend, context `frontend/` — the
    # repo-root .dockerignore excludes frontend/, so `.` would be an empty tree.
    assert "deploy/Dockerfile.frontend" in frontend_args, (
        "frontend build must use deploy/Dockerfile.frontend"
    )
    assert frontend_args[-1] == "frontend", (
        f"frontend build context must be `frontend/`, never the repo root; "
        f"got {frontend_args[-1]!r}"
    )


@pytest.mark.security
def test_cloudbuild_pushes_commit_tagged_images(cloudbuild: dict[str, Any]) -> None:
    images = cloudbuild.get("images") or []
    joined = " ".join(images)
    assert "/api:${_COMMIT_SHA}" in joined and "/frontend:${_COMMIT_SHA}" in joined, (
        "cloudbuild.yaml must push api + frontend tagged by ${_COMMIT_SHA} so the "
        f"deploy step can pin an immutable per-commit digest; got {images}"
    )
    # _COMMIT_SHA carries NO default — a missing value must fail the build loudly,
    # never silently push a mutable `latest`-shaped tag.
    assert "_COMMIT_SHA" not in (cloudbuild.get("substitutions") or {}), (
        "cloudbuild.yaml must not default _COMMIT_SHA — require it at submit time"
    )


@pytest.mark.security
def test_cloudbuild_bakes_no_secret() -> None:
    text = _CLOUDBUILD.read_text(encoding="utf-8").lower()
    assert not re.search(r"\bsk-[a-z0-9]{8,}", text), "cloudbuild.yaml contains an sk- token"
    for needle in ("bearer ", "private key", "begin certificate", "credentials_json"):
        assert needle not in text, f"cloudbuild.yaml carries secret-shaped material: {needle!r}"


# ---------------------------------------------------------------------------
# API registry layer cache on Cloud Build (F26)
#
# Cloud Build workers are ephemeral, so the API build imports/exports a BuildKit
# layer cache from Artifact Registry. A cache hit reuses a layer WITHOUT
# re-running the step that made it — including the hash-verified dependency
# install — so the cache itself is a trust input. These lockers keep it:
#
#   - SCOPED. Cache refs live in this project's own Artifact Registry repo
#     (same write boundary as the images), never a third-party registry, and
#     never under the deployable `api` image name.
#   - ROTATED. The cache tag carries ${_CACHE_EPOCH}, which has no default and
#     which deploy.yml derives from the runner's UTC clock (ISO year-week) —
#     never from event data. The first build of each epoch is cold, bounding
#     poisoned-entry persistence and apt/torch staleness to one week.
#   - COMPLETE + NON-BLOCKING. mode=max (the builder stage is not in the final
#     image, so min mode would miss the dependency install); ignore-error=true
#     (a failed export never blocks a deploy); --load (the `images:` push reads
#     the worker daemon).
#   - PINNED EXECUTOR. The BuildKit image that runs every build step is
#     digest-pinned, like the Dockerfile bases.
# ---------------------------------------------------------------------------

_ARTIFACT_REGISTRY_REPO = "${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO}/"


def _step_args(step: dict[str, Any]) -> list[str]:
    return [str(arg) for arg in step.get("args", [])]


def _flag_values(args: list[str], flag: str) -> list[str]:
    """Values of *flag* in an argv list, in both ``--flag v`` and ``--flag=v`` form."""
    values = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == flag]
    values += [arg.partition("=")[2] for arg in args if arg.startswith(f"{flag}=")]
    return values


def _cache_spec(value: str) -> dict[str, str]:
    """``type=registry,ref=…,mode=max`` → ``{"type": "registry", …}``."""
    return dict(part.split("=", 1) for part in value.split(",") if "=" in part)


@pytest.mark.security
def test_cloudbuild_api_layer_cache_is_scoped_rotated_and_non_blocking(
    cloudbuild: dict[str, Any],
) -> None:
    api_args = _step_args(_cloudbuild_step(cloudbuild, "build-api"))
    assert api_args[:2] == ["buildx", "build"], (
        f"build-api must run `docker buildx build` to use the registry cache; got {api_args[:2]}"
    )
    cache_from = [_cache_spec(v) for v in _flag_values(api_args, "--cache-from")]
    cache_to = [_cache_spec(v) for v in _flag_values(api_args, "--cache-to")]
    assert cache_from and len(cache_to) == 1, (
        "build-api must import (--cache-from) and export (one --cache-to) a layer cache"
    )
    for spec in [*cache_from, *cache_to]:
        ref = spec.get("ref", "")
        assert spec.get("type") == "registry", f"cache must be a registry cache: {spec}"
        assert ref.startswith(_ARTIFACT_REGISTRY_REPO), (
            f"cache ref {ref!r} is outside this project's Artifact Registry repo — a "
            "third-party cache would let an outside party feed layers into the image"
        )
        assert not ref.startswith(f"{_ARTIFACT_REGISTRY_REPO}api:"), (
            f"cache ref {ref!r} shares the deployable image's name; keep it separate"
        )
        assert ref.endswith(":${_CACHE_EPOCH}"), (
            f"cache ref {ref!r} is not keyed by ${{_CACHE_EPOCH}} — a never-rotated "
            "cache lets one poisoned or stale layer persist indefinitely"
        )
    (export,) = cache_to
    assert export.get("mode") == "max", (
        "--cache-to must use mode=max: the builder stage (the dependency install) "
        "is not in the final image, so min mode caches nothing worth having"
    )
    assert export.get("ignore-error") == "true", (
        "--cache-to must set ignore-error=true — a cache export failure must never block a deploy"
    )
    assert "--load" in api_args, (
        "build-api must --load the image into the worker daemon; the `images:` push "
        "reads it from there"
    )
    assert "_CACHE_EPOCH" not in (cloudbuild.get("substitutions") or {}), (
        "cloudbuild.yaml must not default _CACHE_EPOCH — a default is a cache that "
        "never rotates; require it at submit time"
    )


@pytest.mark.security
def test_cloudbuild_buildkit_executor_is_digest_pinned(cloudbuild: dict[str, Any]) -> None:
    steps = cloudbuild.get("steps", [])
    api_index = next(i for i, s in enumerate(steps) if s.get("id") == "build-api")
    builders = _flag_values(_step_args(steps[api_index]), "--builder")
    assert len(builders) == 1, "build-api must name its buildx builder explicitly"
    creates = [
        (i, _step_args(step))
        for i, step in enumerate(steps)
        if _step_args(step)[:2] == ["buildx", "create"]
        and builders[0] in _flag_values(_step_args(step), "--name")
    ]
    assert len(creates) == 1, f"no single step creates the {builders[0]!r} builder"
    ((create_index, create_args),) = creates
    assert create_index < api_index, "the buildx builder must be created before build-api"
    assert _flag_values(create_args, "--driver") == ["docker-container"], (
        "the builder must use the docker-container driver (the only one here that "
        "can export a registry cache)"
    )
    images = [
        opt.partition("=")[2]
        for opt in _flag_values(create_args, "--driver-opt")
        if opt.startswith("image=")
    ]
    assert images, (
        "the buildx builder does not pin its BuildKit image — buildx would pull a "
        "floating `moby/buildkit:buildx-stable-1`"
    )
    assert all(_DIGEST_RE.search(image) for image in images), (
        f"BuildKit image not digest-pinned (@sha256:...): {images}. It executes every "
        "build step; a floating tag is a silent-substitution vector"
    )


@pytest.mark.security
def test_deploy_workflow_derives_the_cache_epoch_from_the_clock(deploy_workflow: str) -> None:
    submit = [body for body in _run_block_bodies(deploy_workflow) if "gcloud builds submit" in body]
    assert len(submit) == 1, "deploy.yml must submit exactly one Cloud Build"
    (body,) = submit
    epoch = re.search(r'^\s*(\w+)="\$\(date -u \+%G-w%V\)"\s*$', body, re.MULTILINE)
    assert epoch, (
        "deploy.yml must derive the cache epoch from the UTC ISO year-week "
        '(`epoch="$(date -u +%G-w%V)"`) — a clock-bounded key, never event data'
    )
    assert f"_CACHE_EPOCH=${epoch.group(1)}" in body, (
        "deploy.yml does not pass the clock-derived epoch as _CACHE_EPOCH"
    )


# ===========================================================================
# CI workflow supply-chain checks.
# ===========================================================================
# `deploy.yml` already has its supply-chain posture locked above (SHA-pinned
# Actions, digest-pinned deploy, command-injection guard). `ci.yml` carries the
# *other* half of the supply chain — the dependency/secret scan gates and the
# CPU-torch posture — yet, before this check, only one ci.yml invariant was
# pinned (`test_ci_workflow_stays_gcp_credential_free`). Several lockers above
# claim to "mirror ci.yml" without a test that actually holds ci.yml to it.
# These lockers close that gap and consolidate the CI supply-chain surface:
#
#   - ACTION SHA-PINNING. Every `uses:` in ci.yml is a 40-hex commit SHA — a
#     floating `@v5` tag is a silent-substitution vector (mirrors deploy.yml).
#   - SCAN GATES. The three dependency/secret scanners (pip-audit,
#     detect-secrets, pnpm audit) are non-negotiable PR blockers and must not
#     silently vanish from CI.
#   - CPU-ONLY TORCH. The CUDA torch wheel (~2 GB, GPU runtime) is a Docker
#     build-arg opt-in only (deploy/Dockerfile.api `TORCH_INDEX_URL`, default
#     …/whl/cpu); it must never enter the CI dependency set.
#   - COMMAND-INJECTION GUARD. No `${{ github.event.* }}` interpolation inside a
#     `run:` shell — ci.yml states this design principle in its own header
#     comment; this test enforces it (mirrors the deploy.yml guard).
#
# All assertions are on the tracked `.github/workflows/ci.yml`; no network,
# gcloud, or Docker daemon needed.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_ci_workflow_actions_are_sha_pinned(ci_workflow: str) -> None:
    refs = _USES_RE.findall(ci_workflow)
    assert refs, "ci.yml declares no `uses:` actions"
    for ref in refs:
        assert _SHA_PINNED_USES_RE.match(ref), (
            f"ci.yml Action {ref!r} is not pinned by a 40-hex commit SHA — a "
            "floating tag is a silent-substitution / supply-chain vector"
        )


@pytest.mark.security
def test_ci_workflow_carries_supply_chain_scan_gates(ci_workflow: str) -> None:
    # The three dependency/secret scan gates are load-bearing PR blockers; a
    # silent removal would reopen the known-CVE / committed-secret surface.
    for needle, gate in (
        ("pip_audit", "pip-audit (Python dependency CVE scan)"),
        ("detect-secrets-hook", "detect-secrets (committed-secret scan)"),
        ("audit:ci", "pnpm audit (frontend runtime-dependency CVE scan)"),
    ):
        assert needle in ci_workflow, (
            f"ci.yml no longer runs the {gate} gate ({needle!r} missing) — the "
            "supply-chain scan gates are non-negotiable PR blockers"
        )


@pytest.mark.security
def test_ci_workflow_torch_is_cpu_only(ci_workflow: str) -> None:
    # A CUDA torch wheel index has no place in CI: the test job never exercises a
    # real model load, and the GPU build is a deploy/Dockerfile.api build-arg
    # opt-in (`TORCH_INDEX_URL`, default …/whl/cpu). Pin CPU-only directly.
    assert "download.pytorch.org/whl/cu" not in ci_workflow, (
        "ci.yml references a CUDA torch wheel index (…/whl/cu…) — CI must stay "
        "CPU-only; the GPU wheel is a Docker build-arg opt-in, never a CI dependency"
    )
    # Forward guard: plain `pip install torch` on Linux resolves the default-index
    # CUDA build. If a future change installs the heavy [local-embeddings] extra
    # in CI, it MUST force the explicit CPU wheel index.
    if "local-embeddings" in ci_workflow:
        assert "download.pytorch.org/whl/cpu" in ci_workflow, (
            "ci.yml installs the [local-embeddings] extra (torch) without forcing "
            "the CPU wheel index (…/whl/cpu) — the default PyPI torch wheel is the "
            "CUDA build; pin --index-url …/whl/cpu to keep CI CPU-only"
        )


def test_ci_workflow_caches_the_tiktoken_encodings(ci_workflow: str) -> None:
    # F25/F31: without a cache every CI run downloads cl100k_base + o200k_base
    # (~5 MB, ~15 s of the suite). The pytest step's TIKTOKEN_CACHE_DIR must be
    # exactly the path an actions/cache step restores.
    steps = yaml.safe_load(ci_workflow)["jobs"]["test"]["steps"]
    cached = {
        str(step.get("with", {}).get("path", "")).strip()
        for step in steps
        if str(step.get("uses", "")).startswith("actions/cache@")
    }
    pytest_steps = [s for s in steps if "pytest" in str(s.get("run", ""))]
    assert pytest_steps, "ci.yml test job has no pytest step"
    cache_dirs = {str((s.get("env") or {}).get("TIKTOKEN_CACHE_DIR", "")) for s in pytest_steps}
    assert "" not in cache_dirs, "the pytest step does not set TIKTOKEN_CACHE_DIR"
    assert cache_dirs <= cached, (
        f"pytest TIKTOKEN_CACHE_DIR {sorted(cache_dirs)} is not restored by an "
        f"actions/cache step (cached paths: {sorted(cached)})"
    )


@pytest.mark.security
def test_ci_workflow_no_event_interpolation_in_run(ci_workflow: str) -> None:
    for body in _run_block_bodies(ci_workflow):
        assert "${{ github.event" not in body, (
            "ci.yml interpolates a `github.event.*` value directly into a `run:` "
            "shell — route it through an env var (command-injection guard)"
        )


# ---------------------------------------------------------------------------
# Core quality gates are wired as blocking CI checks
#
# The supply-chain *scan* gates (pip-audit / detect-secrets / pnpm audit) are
# already pinned above. These comments pin the four core *quality* gates so
# none can silently vanish from CI and let a regression merge:
#
#   - PRESENCE. Each gate is invoked as a real `run:` step in ci.yml (parsed
#     from the workflow, not matched against a comment).
#   - TOOL ANCHORING. The frontend `pnpm test` / `pnpm build` indirection is
#     anchored to its actual tooling in frontend/package.json: `test` MUST run
#     Vitest in non-watch mode (`vitest run` — a watch-mode invocation would
#     hang CI and never gate), `build` MUST run `next build`.
#   - BLOCKING. No job or step is marked `continue-on-error: true` (which turns
#     a gate advisory), and the gates run on `pull_request` so they can be
#     required status checks.
#
# All assertions are on the tracked workflow + package.json; no network, no
# pnpm, no Docker daemon.
# ---------------------------------------------------------------------------


def _workflow_run_commands(workflow_text: str) -> list[str]:
    """Every step's ``run:`` script across all jobs in a parsed workflow.

    Captures both single-line (``run: ruff check .``) and block
    (``run: |`` …) step bodies, which ``_run_block_bodies`` alone would miss.
    """
    workflow = yaml.safe_load(workflow_text)
    commands: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run")
            if isinstance(run, str):
                commands.append(run)
    return commands


def _workflow_triggers(workflow_text: str) -> Any:
    """The ``on:`` trigger mapping, tolerating YAML 1.1 ``on`` → ``True``."""
    workflow = yaml.safe_load(workflow_text)
    triggers = workflow.get(True)
    if triggers is None:
        triggers = workflow.get("on")
    return triggers


@pytest.fixture(scope="module")
def frontend_package_scripts() -> dict[str, str]:
    data = json.loads(_FRONTEND_PACKAGE_JSON.read_text(encoding="utf-8"))
    return data.get("scripts", {})


@pytest.mark.security
def test_ci_workflow_carries_core_quality_gates(ci_workflow: str) -> None:
    # The four quality gates are non-negotiable PR blockers; a silent
    # removal would let a lint break, a failing test, a type/Vitest regression,
    # or a broken production build merge to main.
    commands = _workflow_run_commands(ci_workflow)

    def _runs(predicate: str) -> bool:
        return any(predicate in command for command in commands)

    for predicate, gate in (
        ("ruff check", "Ruff lint"),
        ("ruff format --check", "Ruff format check"),
        ("pytest", "pytest backend suite"),
        ("pnpm test", "Vitest frontend suite (pnpm test)"),
        ("pnpm build", "Next.js production build (pnpm build)"),
    ):
        assert _runs(predicate), (
            f"ci.yml no longer invokes the {gate} gate (no `run:` step contains "
            f"{predicate!r}) — the core quality gates are non-negotiable PR blockers"
        )


@pytest.mark.security
def test_frontend_scripts_back_core_gates_with_vitest_and_next_build(
    frontend_package_scripts: dict[str, str],
) -> None:
    # ci.yml gates the frontend through the `pnpm test` / `pnpm build` script
    # indirection. Anchor that indirection to the expected tools so the
    # gate cannot be hollowed out by rewriting the script body.
    test_script = frontend_package_scripts.get("test", "")
    assert "vitest" in test_script, (
        "frontend `test` script no longer runs Vitest — `pnpm test` is the CI "
        f"Vitest gate (got {test_script!r})"
    )
    assert "run" in test_script.split(), (
        "frontend `test` script runs Vitest in watch mode — CI needs `vitest run` "
        f"(non-watch); a watch invocation hangs the gate forever (got {test_script!r})"
    )
    build_script = frontend_package_scripts.get("build", "")
    assert "next build" in build_script, (
        "frontend `build` script no longer runs `next build` — `pnpm build` is the "
        f"CI Next.js production-build gate (got {build_script!r})"
    )


@pytest.mark.security
def test_ci_workflow_quality_gates_are_blocking(ci_workflow: str) -> None:
    # A gate marked `continue-on-error: true` reports green even when it fails —
    # advisory, not blocking. None of the CI jobs/steps may carry it.
    workflow = yaml.safe_load(ci_workflow)
    for job_name, job in workflow.get("jobs", {}).items():
        assert job.get("continue-on-error") is not True, (
            f"ci.yml job {job_name!r} is `continue-on-error: true` — a failing gate "
            "would still report success; CI gates must block the PR"
        )
        for step in job.get("steps", []):
            if step.get("continue-on-error") is True:
                label = step.get("name") or step.get("uses") or step.get("run")
                raise AssertionError(
                    f"ci.yml job {job_name!r} step {label!r} is "
                    "`continue-on-error: true` — a failing gate would still report "
                    "success; CI gates must block the PR"
                )

    # The gates must run on pull_request so they can be required status checks
    # on `main` — a push-only workflow never gates a PR before merge.
    triggers = _workflow_triggers(ci_workflow)
    assert isinstance(triggers, dict) and "pull_request" in triggers, (
        "ci.yml does not trigger on `pull_request` — the gates cannot block a PR "
        "before merge unless they run on pull_request events"
    )


# ---------------------------------------------------------------------------
# ChromaDB attack-surface lockers
# ---------------------------------------------------------------------------
#
# The pip-audit job suppresses four chromadb advisories (CVE-2026-45829 /
# -45830 / -45831 / -45833). Every one of them lives in the Chroma **server**:
# the FastAPI HTTP surface, its tenant model, and its RBAC authorization
# provider. The suppressions are therefore only sound while this project keeps
# embedding ChromaDB as a local ``chromadb.PersistentClient`` and never serves,
# calls, or authorizes a Chroma HTTP endpoint.
#
# That precondition used to live in a workflow comment. These two lockers make
# it executable: the first fails the moment a server-surface symbol appears in
# ``src/``, the second fails the moment the ignore list drifts from the
# reviewed set. Neither needs Docker or a network.

# Client constructors that speak to a Chroma **server** rather than a local
# file. ``PersistentClient`` (the sanctioned one) is deliberately absent.
_CHROMA_SERVER_CLIENTS = frozenset({"HttpClient", "AsyncHttpClient", "CloudClient", "Client"})

# The keyword that turns a collection's embedding-function config into remote
# code execution (CVE-2026-45829 / -45833).
_CHROMA_RCE_KEYWORD = "trust_remote_code"

# Substrings identifying Chroma's authn/authz provider stack (CVE-2026-45831).
_CHROMA_AUTHZ_MARKERS = ("AuthorizationProvider", "AuthenticationProvider", "SimpleRBAC")

_EXPECTED_PIP_AUDIT_IGNORES = frozenset(
    {
        "CVE-2026-45829",
        "CVE-2026-45830",
        "CVE-2026-45831",
        "CVE-2026-45833",
    }
)

_IGNORE_VULN_RE = re.compile(r"--ignore-vuln\s+(\S+)")


def _dotted_name(node: ast.AST) -> str:
    """Render an attribute/name chain as a dotted string (``a.b.c``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _chroma_server_surface_violations(source: str, label: str) -> list[str]:
    """Return every Chroma **server**-surface use in *source*.

    AST-based on purpose: a substring grep over the tree trips on the
    word ``PersistentClient`` inside a docstring (``database/reindex.py``
    has exactly that), and would equally miss a symbol reached through an
    ``import ... as`` alias.
    """
    violations: list[str] = []
    tree = ast.parse(source)

    # Resolve module aliases first: ``import chromadb as db`` makes
    # ``db.HttpClient(...)`` the same call as ``chromadb.HttpClient(...)``,
    # and matching the literal base name alone would sail straight past it.
    chroma_module_names = {"chromadb"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "chromadb" or alias.name.startswith("chromadb."):
                    chroma_module_names.add(alias.asname or alias.name.split(".")[0])

    for node in ast.walk(tree):
        # ``chromadb.HttpClient(...)`` / ``db.Client(...)`` for any binding
        # of the chromadb module.
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _CHROMA_SERVER_CLIENTS
            and _dotted_name(node.value).split(".")[-1] in chroma_module_names
        ):
            violations.append(f"{label}:{node.lineno}: chromadb.{node.attr} (server client)")

        # ``from chromadb import HttpClient`` / ``from chromadb.auth import
        # SimpleRBACAuthorizationProvider`` (aliased or not). An import alias
        # is neither a Name nor an Attribute, so the marker sweep below can
        # never see it — both checks have to happen here.
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("chromadb"):
                for alias in node.names:
                    if alias.name in _CHROMA_SERVER_CLIENTS:
                        violations.append(
                            f"{label}:{node.lineno}: from {module} import "
                            f"{alias.name} (server client)"
                        )
            for alias in node.names:
                if any(marker in alias.name for marker in _CHROMA_AUTHZ_MARKERS):
                    violations.append(
                        f"{label}:{node.lineno}: from {module} import "
                        f"{alias.name} (Chroma authz provider)"
                    )

        # ``import chromadb.auth`` / ``import chromadb.auth as ...``
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("chromadb.auth"):
                    violations.append(
                        f"{label}:{node.lineno}: import {alias.name} (Chroma authz stack)"
                    )

        # ``trust_remote_code=True`` anywhere at all.
        if isinstance(node, ast.keyword) and node.arg == _CHROMA_RCE_KEYWORD:
            violations.append(f"{label}:{node.lineno}: {_CHROMA_RCE_KEYWORD}= (remote-code config)")

        # Any reference to Chroma's authn/authz provider stack.
        if isinstance(node, ast.Name | ast.Attribute):
            rendered = node.id if isinstance(node, ast.Name) else node.attr
            if any(marker in rendered for marker in _CHROMA_AUTHZ_MARKERS):
                violations.append(f"{label}:{node.lineno}: {rendered} (Chroma authz provider)")

    return violations


@pytest.mark.security
def test_chromadb_usage_stays_embedded_only() -> None:
    """``src/`` must never reach the Chroma server surface.

    This is the executable half of the pip-audit suppressions documented
    in ``ci.yml``. All four ignored advisories are server-side; the
    moment this project opens an ``HttpClient``, enables
    ``trust_remote_code``, or configures an authorization provider, they
    stop being unreachable and the matching ``--ignore-vuln`` MUST be
    removed in the same change.

    Scoped to ``src/`` deliberately: ``tests/`` constructs a real
    ``PersistentClient`` in many places, and widening the scan would
    force exclusions that hollow it out.
    """
    violations: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        violations.extend(
            _chroma_server_surface_violations(
                path.read_text(encoding="utf-8"),
                str(path.relative_to(_REPO_ROOT)),
            )
        )

    assert not violations, (
        "src/ reaches the ChromaDB *server* attack surface, which the pip-audit "
        "suppressions in .github/workflows/ci.yml assume is unreachable "
        "(CVE-2026-45829/-45830/-45831/-45833 are all server-side). Either revert "
        "the change or remove the matching --ignore-vuln entries in the same "
        "commit:\n  " + "\n  ".join(violations)
    )


@pytest.mark.security
def test_the_chromadb_scan_actually_finds_planted_violations() -> None:
    """Non-vacuity guard for the scan above.

    ``test_chromadb_usage_stays_embedded_only`` passes on a clean tree,
    so on its own it cannot distinguish "no violations" from "scanner is
    broken". Each planted sample below must be caught, and the
    sanctioned embedded usage must NOT be.
    """
    planted = (
        "import chromadb\nclient = chromadb.HttpClient(host='chroma')\n",
        "import chromadb as db\nclient = db.HttpClient(host='chroma')\n",
        "import chromadb.auth as ca\nprovider = ca.SimpleRBACAuthorizationProvider()\n",
        "import chromadb\nclient = chromadb.Client()\n",
        "from chromadb import HttpClient\n",
        "from chromadb import HttpClient as Remote\n",
        "collection.modify(configuration={'trust_remote_code': True})\n"
        "fn(trust_remote_code=True)\n",
        "from chromadb.auth import SimpleRBACAuthorizationProvider\n",
        "from chromadb.auth import SimpleRBACAuthorizationProvider as P\n",
        "import chromadb.auth\n",
        "provider = chromadb.auth.SimpleRBACAuthorizationProvider()\n",
    )
    for sample in planted:
        assert _chroma_server_surface_violations(sample, "<planted>"), (
            f"the ChromaDB server-surface scan missed a planted violation:\n{sample}"
        )

    # The sanctioned embedded client must stay clean — including the
    # docstring mention that a substring grep would false-positive on.
    sanctioned = (
        '"""Opens a chromadb.PersistentClient and an HttpClient-free path."""\n'
        "import chromadb\n"
        "import chromadb as db\n"
        "client = chromadb.PersistentClient(path='/app/data/chroma_db')\n"
        "other = db.PersistentClient(path='/app/data/chroma_db')\n"
    )
    assert not _chroma_server_surface_violations(sanctioned, "<sanctioned>"), (
        "the ChromaDB server-surface scan false-positives on the sanctioned "
        "embedded PersistentClient usage"
    )


@pytest.mark.security
def test_ci_pip_audit_ignore_set_is_exactly_the_reviewed_cves(ci_workflow: str) -> None:
    """Pin the ``--ignore-vuln`` set so a fifth suppression is a reviewed edit.

    Suppressing a CVE is a security decision with an expiry condition
    (a fixed release ships, or the reachability argument stops holding).
    Appending one quietly is exactly the failure mode this pins shut —
    the same posture as the route-inventory allow-lists.
    """
    found = set(_IGNORE_VULN_RE.findall(ci_workflow))

    unreviewed = found - _EXPECTED_PIP_AUDIT_IGNORES
    assert not unreviewed, (
        f"ci.yml suppresses CVEs that are not in the reviewed set: {sorted(unreviewed)}. "
        "Adding a --ignore-vuln entry is a deliberate security decision: document why "
        "the advisory is unreachable AND that no fixed release exists, then add it to "
        "_EXPECTED_PIP_AUDIT_IGNORES in the same commit"
    )

    stale = _EXPECTED_PIP_AUDIT_IGNORES - found
    assert not stale, (
        f"ci.yml no longer suppresses {sorted(stale)}, but the locker still expects it. "
        "If a fixed chromadb release shipped and the ignore was dropped, drop it from "
        "_EXPECTED_PIP_AUDIT_IGNORES too (and re-audit the rest)"
    )


@pytest.mark.security
def test_ci_workflow_audits_the_api_image_lock(ci_workflow: str) -> None:
    """The release image's exact dependency set is CVE-audited on every PR (F26).

    The resolved-environment audit above never contains the SQLCipher or
    on-device-embedder extras (CI installs neither), so without this step the
    packages the API image actually ships — pysqlcipher3, sentence-transformers,
    transformers and their trees — would go unaudited. It is also the
    compensating control for pinning: a stale lock carrying a newly-advised
    version turns CI red instead of shipping quietly.
    """
    steps = yaml.safe_load(ci_workflow)["jobs"]["dependency-scan"]["steps"]
    audits = [
        str(step.get("run", ""))
        for step in steps
        if "pip_audit" in str(step.get("run", ""))
        and _API_LOCK_IN_CONTEXT in str(step.get("run", ""))
    ]
    assert len(audits) == 1, (
        f"the dependency-scan job must run exactly one pip-audit over {_API_LOCK_IN_CONTEXT}"
    )
    (command,) = audits
    assert "--disable-pip" in command, (
        "the lock audit must be a pure pinned-version lookup (--disable-pip): resolving "
        "it would pull the default-index CUDA torch and compile pysqlcipher3 in CI"
    )
    assert set(_IGNORE_VULN_RE.findall(command)) == _EXPECTED_PIP_AUDIT_IGNORES, (
        "the lock audit must carry exactly the reviewed chromadb ignore set"
    )
