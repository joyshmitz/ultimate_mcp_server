"""Generic local (OpenAI-compatible) LLM provider implementation.

This single, configurable provider targets any local inference server that exposes
an OpenAI-compatible ``/v1/chat/completions`` API. That covers the common local
runtimes used to avoid paid API calls, including:

- **Ollama** (``http://localhost:11434/v1``) -- the default endpoint.
- **llama.cpp** server (``llama-server``).
- **mistral.rs** (``--port`` OpenAI server mode).
- **vLLM** (OpenAI-compatible server).
- **LM Studio** (``http://localhost:1234/v1``).

Because all of these speak the same wire protocol, one provider with a configurable
``base_url`` covers every case, minimizing the maintenance surface versus bespoke
clients per backend.

Cost is always reported as **$0.00**: local inference runs on hardware you already
own, so there is no per-token charge. This is the whole point for the cost-saving /
delegation strategy -- the cost optimizer will always prefer a local model when one
is available and capable of the task.
"""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from ultimate_mcp_server.constants import DEFAULT_MODELS, Provider
from ultimate_mcp_server.core.providers.base import BaseProvider, ModelResponse
from ultimate_mcp_server.utils import get_logger

# Use the same naming scheme everywhere: logger at module level
logger = get_logger("ultimate_mcp_server.providers.local")

# Default base URL points at Ollama's OpenAI-compatible endpoint, which is the most
# common local runtime. Override with LOCAL_LLM_BASE_URL (or config) for llama.cpp,
# mistral.rs, vLLM, LM Studio, etc.
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"

# Most local servers ignore the API key entirely, but the OpenAI client requires a
# non-empty value. Use a harmless sentinel when none is configured.
DUMMY_API_KEY = "local-no-key-required"


class LocalProvider(BaseProvider):
    """Provider for any local OpenAI-compatible LLM server (Ollama, llama.cpp, mistral.rs, vLLM, LM Studio).

    A single configurable provider rather than one client per backend: point ``base_url``
    at whichever local server you run. All inference is treated as **free** ($0 cost) so
    the cost optimizer prefers it for delegated / cost-sensitive work.
    """

    provider_name = Provider.LOCAL.value

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the local provider.

        Args:
            api_key: Optional API key. Most local servers ignore this; if omitted a
                harmless dummy value is used because the OpenAI client requires one.
            **kwargs: Additional options:
                - base_url (str): Override the local server base URL
                  (default: ``http://localhost:11434/v1`` for Ollama).
        """
        config = self._get_provider_config()

        super().__init__(api_key=api_key, **kwargs)
        self.name = "local"

        # Resolve base_url: explicit kwarg wins, then config, then the Ollama default.
        self.base_url = (
            kwargs.get("base_url")
            or (config.base_url if config else None)
            or DEFAULT_LOCAL_BASE_URL
        )

        # Resolve default model: config first, then constants. May be None if the user
        # has not configured one (they can always pass `model=` explicitly per call).
        self.default_model = (config.default_model if config else None) or DEFAULT_MODELS.get(
            Provider.LOCAL
        )

        # API key is optional for local servers. Fall back to a dummy so the OpenAI
        # client (which forbids an empty key) can be constructed.
        if not self.api_key:
            self.api_key = DUMMY_API_KEY

        self.client = None
        self.available_models: List[Dict[str, Any]] = []

    @staticmethod
    def _get_provider_config():
        """Safely fetch this provider's config block, returning None if unavailable."""
        try:
            from ultimate_mcp_server.config import get_config

            providers_config = getattr(get_config(), "providers", None)
            return getattr(providers_config, Provider.LOCAL.value, None) if providers_config else None
        except Exception:  # pragma: no cover - config is optional at construction time
            return None

    async def initialize(self) -> bool:
        """Initialize the OpenAI-compatible client pointed at the local server.

        Returns:
            bool: True if the client was constructed successfully. Note that this does
            NOT require the local server to be reachable -- connection failures are
            surfaced clearly at call time so a missing/asleep server degrades gracefully
            instead of crashing provider discovery.
        """
        try:
            config = self._get_provider_config()
            timeout = (config.timeout if config and config.timeout else None) or 30.0

            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=timeout,
            )

            # Best-effort model pre-fetch. A local server that is not running must not
            # break initialization; we just log and fall back to configured models.
            try:
                self.available_models = await self.list_models()
                logger.info(
                    f"Loaded {len(self.available_models)} models from local server at {self.base_url}"
                )
            except Exception as model_err:
                logger.warning(
                    f"Could not list models from local server at {self.base_url}: {model_err}. "
                    "Provider remains available; specify a model explicitly per request."
                )
                self.available_models = self._get_fallback_models()

            logger.success(
                f"Local (OpenAI-compatible) provider initialized at {self.base_url}",
                emoji_key="provider",
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize local provider at {self.base_url}: {str(e)}",
                emoji_key="error",
            )
            return False

    def _strip_provider_prefix(self, model: Optional[str]) -> Optional[str]:
        """Strip a leading ``local:`` / ``local/`` prefix from a model name if present."""
        if not model:
            return model
        for sep in (":", "/"):
            prefix = f"{self.provider_name}{sep}"
            if model.startswith(prefix):
                return model[len(prefix) :]
        return model

    async def generate_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        json_mode: bool = False,
        **kwargs,
    ) -> ModelResponse:
        """Generate a completion from the local server.

        Args:
            prompt: Text prompt (optional if ``messages`` provided).
            messages: List of chat messages (optional if ``prompt`` provided).
            model: Model name served by the local backend (e.g. ``llama3.1:8b``).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            json_mode: Request JSON-formatted output (if the backend supports it).
            **kwargs: Additional OpenAI-compatible parameters.

        Returns:
            ModelResponse whose ``cost`` is always 0.0 (local inference is free).

        Raises:
            ValueError: If neither prompt nor messages is provided, or no model resolved.
            RuntimeError: If the local server is unreachable or the request fails.
        """
        if not self.client:
            if not await self.initialize():
                raise RuntimeError(f"{self.provider_name} provider failed to initialize.")

        if prompt is None and not messages:
            raise ValueError("Either prompt or messages must be provided")

        model = self._strip_provider_prefix(model) or self.default_model
        if model is None:
            raise ValueError(
                "No model specified and no default model configured for the local provider. "
                "Set LOCAL_LLM_DEFAULT_MODEL or pass model= explicitly."
            )

        chat_messages = messages or [{"role": "user", "content": prompt}]

        params: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if json_mode:
            params["response_format"] = {"type": "json_object"}
            self.logger.debug("Setting response_format to JSON mode for local provider")

        # Pass through any remaining standard OpenAI-compatible kwargs.
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        prompt_length = len(prompt) if prompt else sum(len(m.get("content", "")) for m in chat_messages)
        self.logger.info(
            f"Generating completion with local model {model}",
            emoji_key=self.provider_name,
            prompt_length=prompt_length,
            json_mode=json_mode,
        )

        try:
            response, processing_time = await self.process_with_timer(
                self.client.chat.completions.create, **params
            )
        except Exception as e:
            self.logger.error(
                f"Local completion failed for model {model} at {self.base_url}: {str(e)}",
                emoji_key="error",
                model=model,
            )
            raise RuntimeError(
                f"Local LLM request failed (is the server running at {self.base_url}?): {e}"
            ) from e

        completion_text = response.choices[0].message.content or ""

        # Local servers may omit usage; default to 0 so cost stays 0 regardless.
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) or (
            input_tokens + output_tokens
        )

        result = ModelResponse(
            text=completion_text,
            model=f"{self.provider_name}/{model}",
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            processing_time=processing_time,
            raw_response=response,
        )

        # Local inference is free: force zero cost so the optimizer always prefers it.
        result.cost = 0.0

        # Add message for compatibility with chat_completion
        result.message = {"role": "assistant", "content": completion_text}

        self.logger.success(
            "Local completion successful",
            emoji_key="success",
            model=model,
            tokens={"input": result.input_tokens, "output": result.output_tokens},
            cost=result.cost,
            time=result.processing_time,
        )

        return result

    async def generate_completion_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        json_mode: bool = False,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Stream a completion from the local server.

        Yields:
            Tuples of ``(text_chunk, metadata)``. Every metadata dict reports ``cost: 0.0``.

        Raises:
            ValueError: If neither prompt nor messages is provided, or no model resolved.
            RuntimeError: If the local server is unreachable or the request fails.
        """
        if not self.client:
            if not await self.initialize():
                raise RuntimeError(f"{self.provider_name} provider failed to initialize.")

        if prompt is None and not messages:
            raise ValueError("Either prompt or messages must be provided")

        model = self._strip_provider_prefix(model) or self.default_model
        if model is None:
            raise ValueError(
                "No model specified and no default model configured for the local provider."
            )

        chat_messages = messages or [{"role": "user", "content": prompt}]

        params: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        for key, value in kwargs.items():
            if key not in params and key != "stream":
                params[key] = value

        self.logger.info(
            f"Generating streaming completion with local model {model}",
            emoji_key=self.provider_name,
        )

        start_time = time.time()
        total_chunks = 0
        final_model_name = f"{self.provider_name}/{model}"

        try:
            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                total_chunks += 1
                delta = chunk.choices[0].delta
                content = delta.content or ""

                metadata = {
                    "model": final_model_name,
                    "provider": self.provider_name,
                    "chunk_index": total_chunks,
                    "finish_reason": chunk.choices[0].finish_reason,
                    "cost": 0.0,  # Local inference is free
                }

                yield content, metadata

            processing_time = time.time() - start_time
            self.logger.success(
                "Local streaming completion successful",
                emoji_key="success",
                model=model,
                chunks=total_chunks,
                time=processing_time,
            )

            # Final sentinel chunk
            yield "", {
                "model": final_model_name,
                "provider": self.provider_name,
                "chunk_index": total_chunks + 1,
                "processing_time": processing_time,
                "finish_reason": "stop",
                "cost": 0.0,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"Local streaming completion failed for model {model} at {self.base_url}: {str(e)}",
                emoji_key="error",
                model=model,
            )
            yield "", {
                "model": final_model_name,
                "provider": self.provider_name,
                "chunk_index": total_chunks + 1,
                "error": f"{type(e).__name__}: {str(e)}",
                "processing_time": processing_time,
                "finish_reason": "error",
                "cost": 0.0,
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List models advertised by the local server's ``/v1/models`` endpoint.

        Falls back to configured/fallback models if the endpoint is unavailable.

        Returns:
            List of model info dictionaries with ``id``, ``provider`` and ``description``.
        """
        if self.available_models:
            return self.available_models

        if not self.client:
            # Build a transient client just for listing if not initialized yet.
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        models: List[Dict[str, Any]] = []
        try:
            response = await self.client.models.list()
            for m in response.data:
                model_id = getattr(m, "id", None)
                if model_id:
                    models.append(
                        {
                            "id": model_id,
                            "provider": self.provider_name,
                            "description": f"Local model served at {self.base_url}",
                        }
                    )
        except Exception as e:
            logger.debug(f"Local /v1/models listing failed: {e}; using fallback models.")

        if not models:
            models = self._get_fallback_models()

        self.available_models = models
        return models

    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Return the configured default model (if any) as the fallback model list."""
        if self.default_model:
            return [
                {
                    "id": self.default_model,
                    "provider": self.provider_name,
                    "description": f"Configured local model at {self.base_url}",
                }
            ]
        return []

    def get_default_model(self) -> Optional[str]:
        """Get the default local model.

        Returns:
            The configured default model name, or None if none is configured.
        """
        return self.default_model

    def get_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Local inference is free.

        Returns:
            Always 0.0 -- this is what makes the cost optimizer prefer local models.
        """
        return 0.0

    async def check_api_key(self) -> bool:
        """Local servers generally need no API key, so this is always considered valid."""
        return True


# Make available via discovery
__all__ = ["LocalProvider"]
