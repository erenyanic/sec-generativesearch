"""F28 — start-up must not import what the command does not use.

Each scenario runs in a **fresh interpreter** (imports accumulate within a
process, so an in-process check would see whatever earlier tests loaded).
Before F28, ``sec-rag --help`` took ~3.4 s and ~260 MB: every CLI module was
imported eagerly and dragged in the three vendor SDKs (``openai``,
``anthropic``, ``google.genai`` — ~1.75 s), ``edgar`` (~0.8 s) and
``chromadb`` (~0.6 s).  The demo-reset Job and every Cloud Run cold start
paid the same.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_HEAVY = ("openai", "anthropic", "google.genai", "edgar", "chromadb")


def _heavy_modules_loaded(code: str) -> list[str]:
    probe = (
        "import json, sys\n"
        f"{code}\n"
        f"print(json.dumps(sorted(m for m in {_HEAVY!r} if m in sys.modules)))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "FORCE_COLOR"}
    result = subprocess.run(  # noqa: S603 — fixed argv (this interpreter), no shell
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_cli_import_and_help_load_no_sdk_edgar_or_chromadb() -> None:
    code = (
        "import sec_generative_search.cli.main as cli_main\n"
        "sys.argv = ['sec-rag', '--help']\n"
        "try:\n"
        "    cli_main.main()\n"
        "except SystemExit:\n"
        "    pass\n"
    )
    assert _heavy_modules_loaded(code) == []


@pytest.mark.parametrize("extra", [[], ["--output", "json"]], ids=["table", "json"])
def test_provider_list_loads_no_sdk_edgar_or_chromadb(extra: list[str]) -> None:
    code = (
        "import contextlib, io\n"
        "import sec_generative_search.cli.main as cli_main\n"
        f"sys.argv = ['sec-rag', 'provider', 'list', *{extra!r}]\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    try:\n"
        "        cli_main.main()\n"
        "    except SystemExit:\n"
        "        pass\n"
    )
    assert _heavy_modules_loaded(code) == []


def test_settings_and_registry_probes_load_no_sdk() -> None:
    """The ``EMBEDDING_PROVIDER`` validator, listings and the LLM capability
    probe are metadata reads — they must not import an adapter's SDK."""
    code = (
        "from sec_generative_search.config.settings import get_settings\n"
        "get_settings()\n"
        "from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface\n"
        "ProviderRegistry.list_providers(ProviderSurface.LLM)\n"
        "ProviderRegistry.list_providers(ProviderSurface.EMBEDDING)\n"
        "ProviderRegistry.get_capability('openai', ProviderSurface.LLM)\n"
        "ProviderRegistry.get_capability('anthropic', ProviderSurface.LLM, 'claude-haiku-4-5')\n"
        "ProviderRegistry.list_models('gemini', ProviderSurface.LLM)\n"
        "import sec_generative_search.providers\n"
        "import sec_generative_search.pipeline\n"
        "import sec_generative_search.database\n"
    )
    assert _heavy_modules_loaded(code) == []


def test_package_exports_still_resolve_on_access() -> None:
    """The lazy ``providers`` package still exports every name it did."""
    import sec_generative_search.providers as providers

    for name in providers.__all__:
        assert getattr(providers, name) is not None, name
    with pytest.raises(AttributeError):
        _ = providers.NotAProvider  # type: ignore[attr-defined]
