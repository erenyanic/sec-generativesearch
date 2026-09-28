"""Tests for the CLI's shared helpers — :mod:`sec_generative_search.cli._errors`
and :mod:`sec_generative_search.cli._common` (OPTIMIZATIONS F32).

The provider ladder decides whether an operator is told to rotate a key, so
it is security-relevant: it used to be written out six times across
``cli/rag.py`` and ``cli/provider.py``.  It must stay one table, keep its
subclass-before-base ordering, never put exception *text* into ``details``,
and keep saying what the API says for the same failure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import typer

from sec_generative_search.cli._common import resolve_stamp, validate_date
from sec_generative_search.cli._errors import LOCAL_ENDPOINT_HINT, classify_provider_error
from sec_generative_search.core.exceptions import (
    DatabaseError,
    GenerationError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SearchError,
)

_CLI = Path(__file__).resolve().parents[2] / "src" / "sec_generative_search" / "cli"
_SECRET_TEXT = "https://api.example.invalid/v1?key=sk-SHOULD-NOT-LEAK"  # pragma: allowlist secret


def _classify(exc: BaseException, **kwargs):
    return classify_provider_error(exc, provider_name="openai", phase="generation", **kwargs)


@pytest.mark.security
class TestProviderLadder:
    @pytest.mark.parametrize(
        ("exc", "code", "label"),
        [
            (ProviderAuthError("x"), "provider_unauthorized", "Provider unauthorised"),
            (ProviderRateLimitError("x"), "provider_unavailable", "Provider unavailable"),
            (ProviderTimeoutError("x"), "provider_unavailable", "Provider unavailable"),
            (ProviderConnectionError("x"), "provider_unavailable", "Provider unavailable"),
            (ProviderContentFilterError("x"), "provider_error", "Provider error"),
            (ProviderError("x"), "provider_error", "Provider error"),
            (GenerationError("x"), "generation_failed", "Generation failed"),
        ],
    )
    def test_each_failure_lands_on_its_rung(self, exc, code, label) -> None:
        classified = _classify(exc)
        assert classified is not None
        assert (classified.error_code, classified.label) == (code, label)

    def test_only_the_auth_rung_suggests_rotating_the_key(self) -> None:
        """Telling an operator to rotate a working key on a transient or
        non-auth failure burns quota and invites a needless rotation."""
        for exc in (
            ProviderRateLimitError("x"),
            ProviderTimeoutError("x"),
            ProviderConnectionError("x"),
            ProviderError("x"),
            GenerationError("x"),
        ):
            hint = _classify(exc).hint  # type: ignore[union-attr]
            assert "rotate the provider key" not in hint, type(exc).__name__
        auth_hint = _classify(ProviderAuthError("x")).hint  # type: ignore[union-attr]
        assert "rotate the provider key for 'openai'" in auth_hint

    def test_connection_rung_is_distinct_except_for_validation(self) -> None:
        exc = ProviderConnectionError("refused")
        assert _classify(exc).hint == LOCAL_ENDPOINT_HINT  # type: ignore[union-attr]
        collapsed = _classify(exc, connection_is_distinct=False)
        assert collapsed is not None and collapsed.error_code == "provider_error"

    def test_phase_names_the_step(self) -> None:
        classified = classify_provider_error(
            ProviderError("x"), provider_name="openai", phase="query understanding"
        )
        assert classified is not None
        assert classified.message.endswith("during query understanding.")

    def test_details_carry_the_type_never_the_text(self) -> None:
        for exc in (
            ProviderAuthError(_SECRET_TEXT, details=_SECRET_TEXT),
            ProviderRateLimitError(_SECRET_TEXT, details=_SECRET_TEXT),
            ProviderConnectionError(_SECRET_TEXT, details=_SECRET_TEXT),
            ProviderError(_SECRET_TEXT, details=_SECRET_TEXT),
            GenerationError(_SECRET_TEXT, details=_SECRET_TEXT),
        ):
            classified = _classify(exc)
            assert classified is not None
            rendered = " ".join(
                filter(None, [classified.message, classified.hint, classified.details])
            )
            assert "SHOULD-NOT-LEAK" not in rendered
            assert classified.details in (None, type(exc).__name__)

    @pytest.mark.parametrize(
        "exc", [SearchError("x"), DatabaseError("x"), ValueError("x"), RuntimeError("x")]
    )
    def test_off_ladder_failures_are_left_to_the_caller(self, exc) -> None:
        assert _classify(exc) is None


@pytest.mark.security
class TestLadderMirrorsTheApi:
    """The CLI mirrors — never imports — ``api.routes.rag._PROVIDER_ERROR_LADDER``.
    Same failure, same message: checked here, where importing both is fine."""

    @pytest.mark.parametrize(
        "exc",
        [
            ProviderAuthError("x"),
            ProviderRateLimitError("x"),
            ProviderTimeoutError("x"),
            ProviderConnectionError("x"),
            ProviderError("x"),
            GenerationError("x"),
        ],
    )
    def test_message_matches_the_api_rung(self, exc) -> None:
        from sec_generative_search.api.routes.rag import _match_provider_error

        api = _match_provider_error(exc)
        cli = _classify(exc)
        assert api is not None and cli is not None
        assert cli.message == api.message.format(phase="generation")
        if not isinstance(exc, ProviderAuthError | ProviderConnectionError):
            # The auth hint names the CLI's key source; the endpoint hint
            # adds the CLI-specific `ollama serve` / LOCAL_LLM_BASE_URL.
            assert cli.hint == api.hint


@pytest.mark.security
class TestOneCopyOfEachCliHelper:
    """Ten error renderers, four date validators, eight stamp resolutions and
    six provider ladders became one each; a re-fork re-opens the drift."""

    @staticmethod
    def _modules() -> list[tuple[str, ast.Module]]:
        return [
            (path.name, ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(_CLI.glob("*.py"))
        ]

    def test_helpers_are_defined_only_in_common(self) -> None:
        forbidden = {"_print_error", "print_error", "_validate_date", "validate_date"}
        offenders = [
            f"{name}:{node.name}"
            for name, tree in self._modules()
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in forbidden and name != "_common.py"
        ]
        assert offenders == []

    def test_provider_subclasses_are_classified_only_in_errors(self) -> None:
        ladder_types = {
            "ProviderAuthError",
            "ProviderRateLimitError",
            "ProviderTimeoutError",
            "ProviderConnectionError",
        }
        offenders = [
            f"{name}:{node.lineno}"
            for name, tree in self._modules()
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in ladder_types and name != "_errors.py"
        ]
        assert offenders == []

    def test_stamp_error_is_built_only_in_common(self) -> None:
        offenders = [
            f"{name}:{node.lineno}"
            for name, tree in self._modules()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and node.value == "Embedder configuration invalid"
            and name != "_common.py"
        ]
        assert offenders == []


class TestCommonHelpers:
    def test_validate_date(self) -> None:
        assert validate_date(None, "--since") is None
        assert validate_date("2024-02-29", "--since") == "2024-02-29"
        with pytest.raises(typer.BadParameter, match="--since"):
            validate_date("29-02-2024", "--since")

    def test_resolve_stamp(self) -> None:
        stamp = resolve_stamp("openai", "text-embedding-3-small")
        assert (stamp.provider, stamp.model, stamp.dimension) == (
            "openai",
            "text-embedding-3-small",
            1536,
        )
        with pytest.raises(typer.Exit):
            resolve_stamp("openai", "not-a-model")
