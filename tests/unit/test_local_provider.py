"""Tests for the generic local (OpenAI-compatible) provider.

These tests do NOT require a running local LLM server -- the OpenAI-compatible client
is mocked. They cover registration, config/base_url/api_key/model resolution,
zero-cost accounting, and a mocked completion + streaming call.
"""

from typing import Any, Dict

import pytest
from pytest import MonkeyPatch

from ultimate_mcp_server.constants import DEFAULT_MODELS, Provider
from ultimate_mcp_server.core.providers import PROVIDER_REGISTRY
from ultimate_mcp_server.core.providers.base import ModelResponse
from ultimate_mcp_server.core.providers.local import (
    DEFAULT_LOCAL_BASE_URL,
    DUMMY_API_KEY,
    LocalProvider,
)
from ultimate_mcp_server.utils import get_logger

logger = get_logger("test.local_provider")

pytestmark = pytest.mark.asyncio(loop_scope="function")


# --- Mock OpenAI-compatible client -----------------------------------------------------


class _MockUsage:
    def __init__(self):
        self.prompt_tokens = 12
        self.completion_tokens = 8
        self.total_tokens = 20


class _MockMessage:
    def __init__(self):
        self.content = "Mock local response"


class _MockChoice:
    def __init__(self):
        self.message = _MockMessage()
        self.finish_reason = "stop"


class _MockCompletion:
    def __init__(self):
        self.id = "mock-local-completion"
        self.model = "llama3.1:8b"
        self.choices = [_MockChoice()]
        self.usage = _MockUsage()


class _MockStreamChunkDelta:
    def __init__(self, content):
        self.content = content


class _MockStreamChoice:
    def __init__(self, content, finish_reason=None):
        self.delta = _MockStreamChunkDelta(content)
        self.finish_reason = finish_reason


class _MockStreamChunk:
    def __init__(self, content, finish_reason=None):
        self.model = "llama3.1:8b"
        self.choices = [_MockStreamChoice(content, finish_reason)]


class _MockModelEntry:
    def __init__(self, model_id):
        self.id = model_id


class _MockModelsResponse:
    def __init__(self):
        self.data = [_MockModelEntry("llama3.1:8b"), _MockModelEntry("qwen2.5:7b")]


class _MockCompletions:
    def __init__(self, fail=False):
        self._fail = fail

    async def create(self, **kwargs):
        if self._fail:
            raise ConnectionError("connection refused")
        if kwargs.get("stream"):
            async def _gen():
                yield _MockStreamChunk("Mock ")
                yield _MockStreamChunk("local ")
                yield _MockStreamChunk("stream", finish_reason="stop")
            return _gen()
        return _MockCompletion()


class _MockChat:
    def __init__(self, fail=False):
        self.completions = _MockCompletions(fail=fail)


class _MockModels:
    async def list(self):
        return _MockModelsResponse()


class MockAsyncOpenAI:
    def __init__(self, fail=False, **kwargs):
        self.kwargs = kwargs
        self.chat = _MockChat(fail=fail)
        self.models = _MockModels()


def _make_provider(monkeypatch: MonkeyPatch, fail: bool = False, **init_kwargs) -> LocalProvider:
    """Build a LocalProvider with its OpenAI client replaced by a mock."""
    provider = LocalProvider(**init_kwargs)
    provider.client = MockAsyncOpenAI(fail=fail)
    return provider


# --- Registration & config -------------------------------------------------------------


def test_registered_in_registry():
    """The local provider is registered in the provider registry under 'local'."""
    assert Provider.LOCAL.value == "local"
    assert Provider.LOCAL.value in PROVIDER_REGISTRY
    assert PROVIDER_REGISTRY[Provider.LOCAL.value] is LocalProvider


def test_default_base_url_and_model():
    """Defaults to Ollama's OpenAI-compatible endpoint and the constant default model."""
    provider = LocalProvider()
    assert provider.base_url == DEFAULT_LOCAL_BASE_URL == "http://localhost:11434/v1"
    assert provider.default_model == DEFAULT_MODELS.get(Provider.LOCAL)
    assert provider.provider_name == "local"


def test_dummy_api_key_when_unset():
    """When no API key is provided, a harmless dummy is used (OpenAI client needs one)."""
    provider = LocalProvider()
    assert provider.api_key == DUMMY_API_KEY


def test_api_key_override():
    """An explicitly provided API key is respected."""
    provider = LocalProvider(api_key="sk-custom")
    assert provider.api_key == "sk-custom"


def test_base_url_override():
    """base_url kwarg overrides the default (e.g. for LM Studio / vLLM / llama.cpp)."""
    provider = LocalProvider(base_url="http://localhost:1234/v1")
    assert provider.base_url == "http://localhost:1234/v1"


# --- Zero-cost accounting --------------------------------------------------------------


def test_get_cost_is_zero():
    """Local inference is free: get_cost always returns 0.0."""
    provider = LocalProvider()
    assert provider.get_cost("any-model", 1_000_000, 1_000_000) == 0.0


async def test_completion_cost_is_zero(monkeypatch: MonkeyPatch):
    """A completed ModelResponse from the local provider reports zero cost."""
    provider = _make_provider(monkeypatch)
    result = await provider.generate_completion(prompt="Hello", model="llama3.1:8b")
    assert isinstance(result, ModelResponse)
    assert result.cost == 0.0
    # It really used tokens, proving cost is forced to zero, not merely zero tokens.
    assert result.input_tokens == 12
    assert result.output_tokens == 8


# --- Completion ------------------------------------------------------------------------


async def test_completion(monkeypatch: MonkeyPatch):
    """Mocked completion returns a normalized ModelResponse."""
    provider = _make_provider(monkeypatch)
    result = await provider.generate_completion(prompt="Hello", model="llama3.1:8b")
    assert result.text == "Mock local response"
    assert result.provider == Provider.LOCAL.value
    assert result.model == "local/llama3.1:8b"
    assert result.total_tokens == 20
    assert result.message == {"role": "assistant", "content": "Mock local response"}


async def test_completion_uses_default_model(monkeypatch: MonkeyPatch):
    """When no model is passed, the configured default model is used."""
    provider = _make_provider(monkeypatch)
    result = await provider.generate_completion(prompt="Hi")
    assert result.model == f"local/{provider.default_model}"


async def test_completion_strips_provider_prefix(monkeypatch: MonkeyPatch):
    """A 'local:' or 'local/' prefix on the model is stripped before the API call."""
    provider = _make_provider(monkeypatch)
    result = await provider.generate_completion(prompt="Hi", model="local:llama3.1:8b")
    assert result.model == "local/llama3.1:8b"


async def test_completion_requires_prompt_or_messages(monkeypatch: MonkeyPatch):
    """Passing neither prompt nor messages raises ValueError."""
    provider = _make_provider(monkeypatch)
    with pytest.raises(ValueError):
        await provider.generate_completion()


async def test_unreachable_server_raises_clear_error(monkeypatch: MonkeyPatch):
    """An unreachable local server yields a clear RuntimeError, not a raw crash."""
    provider = _make_provider(monkeypatch, fail=True)
    with pytest.raises(RuntimeError) as exc_info:
        await provider.generate_completion(prompt="Hello", model="llama3.1:8b")
    assert "server" in str(exc_info.value).lower()


# --- Streaming -------------------------------------------------------------------------


async def test_streaming(monkeypatch: MonkeyPatch):
    """Streaming yields chunks, each reporting zero cost, plus a final sentinel."""
    provider = _make_provider(monkeypatch)
    chunks = []
    async for text, meta in provider.generate_completion_stream(
        prompt="Hello", model="llama3.1:8b"
    ):
        chunks.append((text, meta))

    # Every chunk reports zero cost.
    assert all(meta["cost"] == 0.0 for _, meta in chunks)
    assert all(meta["provider"] == Provider.LOCAL.value for _, meta in chunks)
    text_joined = "".join(text for text, _ in chunks)
    assert "Mock local stream" in text_joined
    # Final sentinel chunk reports a stop finish reason.
    assert chunks[-1][1]["finish_reason"] == "stop"


async def test_streaming_error_yields_error_metadata(monkeypatch: MonkeyPatch):
    """A streaming error is surfaced as an error metadata chunk (not a crash)."""
    provider = _make_provider(monkeypatch, fail=True)
    chunks = []
    async for text, meta in provider.generate_completion_stream(
        prompt="Hello", model="llama3.1:8b"
    ):
        chunks.append((text, meta))
    assert chunks[-1][1]["finish_reason"] == "error"
    assert "error" in chunks[-1][1]
    assert chunks[-1][1]["cost"] == 0.0


# --- Model listing ---------------------------------------------------------------------


async def test_list_models_from_endpoint(monkeypatch: MonkeyPatch):
    """list_models queries the /v1/models endpoint when available."""
    provider = _make_provider(monkeypatch)
    provider.available_models = []  # force a fresh fetch
    models = await provider.list_models()
    ids = [m["id"] for m in models]
    assert "llama3.1:8b" in ids
    assert "qwen2.5:7b" in ids
    assert all(m["provider"] == Provider.LOCAL.value for m in models)


async def test_check_api_key_always_true():
    """Local servers need no API key, so check_api_key is always True."""
    provider = LocalProvider()
    assert await provider.check_api_key() is True


def test_get_default_model():
    """get_default_model returns the configured default."""
    provider = LocalProvider()
    assert provider.get_default_model() == DEFAULT_MODELS.get(Provider.LOCAL)


# --- Fixtures placeholder (kept for parity with other provider test modules) ------------


@pytest.fixture
def mock_local_responses() -> Dict[str, Any]:
    return {"completion": _MockCompletion(), "models": _MockModelsResponse()}
