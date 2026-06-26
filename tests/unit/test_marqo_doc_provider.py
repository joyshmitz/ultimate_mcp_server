"""Unit tests for Marqo doc-generation provider resolution (gh #16).

The provider used to be hard-coded to ``Provider.GEMINI`` in
``marqo_fused_search._call_llm_for_doc_generation``, so dynamic docstring
augmentation failed entirely when Gemini was not configured. These tests pin
the new configurable + fallback resolution order.
"""

# Import the gateway first so the tools import graph is warmed in the right
# order (mirrors tests/conftest.py). ``marqo_fused_search`` participates in a
# package-level import cycle (completion <-> filesystem) that only resolves once
# ``core.server`` is loaded.
import ultimate_mcp_server.core.server  # noqa: F401  (import-for-side-effect)
from ultimate_mcp_server.constants import Provider
from ultimate_mcp_server.tools.marqo_fused_search import (
    _resolve_doc_generation_providers,
)


def test_override_env_takes_precedence_and_is_lowercased(monkeypatch):
    monkeypatch.setenv("MARQO_DOC_GENERATION_PROVIDER", "Anthropic")
    providers = _resolve_doc_generation_providers()
    assert providers[0] == "anthropic"
    # No duplicates, order preserved.
    assert len(providers) == len(set(providers))


def test_fallback_chain_always_present(monkeypatch):
    monkeypatch.delenv("MARQO_DOC_GENERATION_PROVIDER", raising=False)
    providers = _resolve_doc_generation_providers()
    for expected in (
        Provider.OPENAI.value,
        Provider.ANTHROPIC.value,
        Provider.GEMINI.value,
        Provider.DEEPSEEK.value,
        Provider.OPENROUTER.value,
    ):
        assert expected in providers


def test_no_longer_hardcoded_to_gemini(monkeypatch):
    # With an explicit non-gemini override, gemini must not lead the list
    # (regression guard for the old hard-coded Provider.GEMINI.value behavior).
    monkeypatch.setenv("MARQO_DOC_GENERATION_PROVIDER", "openai")
    providers = _resolve_doc_generation_providers()
    assert providers[0] == "openai"
    assert providers.index("openai") < providers.index("gemini")
