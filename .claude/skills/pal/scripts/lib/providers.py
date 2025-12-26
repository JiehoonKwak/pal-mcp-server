"""
Multi-Provider AI Client Abstraction for PAL Skills.

Supports:
- Google Gemini (with thinking modes)
- OpenAI (GPT-4, O1, O3, O4)
- X.AI (Grok)
- OpenRouter (50+ models)
- Custom/Ollama (local models)
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

# Model capabilities registry
MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    # Gemini models
    "gemini-2.5-flash": {
        "context_window": 1048576,
        "thinking_modes": ["minimal", "low", "medium", "high", "max"],
        "supports_images": True,
        "supports_json_mode": True,
        "max_thinking_tokens": 24576,
    },
    "gemini-2.5-pro": {
        "context_window": 2097152,
        "thinking_modes": ["minimal", "low", "medium", "high", "max"],
        "supports_images": True,
        "supports_json_mode": True,
        "max_thinking_tokens": 24576,
    },
    "gemini-2.0-flash": {
        "context_window": 1048576,
        "thinking_modes": [],
        "supports_images": True,
        "supports_json_mode": True,
    },
    # OpenAI models
    "gpt-4o": {
        "context_window": 128000,
        "thinking_modes": [],
        "supports_images": True,
        "supports_json_mode": True,
    },
    "gpt-4o-mini": {
        "context_window": 128000,
        "thinking_modes": [],
        "supports_images": True,
        "supports_json_mode": True,
    },
    "o1": {
        "context_window": 200000,
        "thinking_modes": [],
        "supports_images": False,
        "supports_json_mode": True,
        "fixed_temperature": True,
    },
    "o1-mini": {
        "context_window": 128000,
        "thinking_modes": [],
        "supports_images": False,
        "supports_json_mode": True,
        "fixed_temperature": True,
    },
    "o3-mini": {
        "context_window": 200000,
        "thinking_modes": ["low", "medium", "high"],
        "supports_images": False,
        "supports_json_mode": True,
        "fixed_temperature": True,
    },
    # Claude models (via OpenRouter)
    "claude-sonnet-4-20250514": {
        "context_window": 200000,
        "thinking_modes": [],
        "supports_images": True,
        "supports_json_mode": True,
    },
    "claude-3.5-sonnet": {
        "context_window": 200000,
        "thinking_modes": [],
        "supports_images": True,
        "supports_json_mode": True,
    },
    # Grok models
    "grok-3": {
        "context_window": 131072,
        "thinking_modes": [],
        "supports_images": False,
        "supports_json_mode": True,
    },
    "grok-2": {
        "context_window": 131072,
        "thinking_modes": [],
        "supports_images": False,
        "supports_json_mode": True,
    },
}

# Default capabilities for unknown models
DEFAULT_CAPABILITIES: dict[str, Any] = {
    "context_window": 128000,
    "thinking_modes": [],
    "supports_images": False,
    "supports_json_mode": True,
}


def get_model_capabilities(model_name: str) -> dict[str, Any]:
    """
    Get capabilities for a model.

    Args:
        model_name: The model name

    Returns:
        Dictionary of model capabilities
    """
    # Check exact match
    if model_name in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model_name]

    # Check prefix match for versioned models
    model_lower = model_name.lower()
    for name, caps in MODEL_CAPABILITIES.items():
        # Match base name (e.g., "gpt-4o" matches "gpt-4o-2024-05-13")
        if model_lower.startswith(name.lower()):
            return caps
        # Match model family (e.g., "gemini" matches any gemini model)
        base_name = name.split("-")[0]
        if model_lower.startswith(base_name):
            return caps

    # Return default capabilities
    return DEFAULT_CAPABILITIES.copy()


def supports_thinking_mode(model_name: str, mode: str) -> bool:
    """Check if a model supports a specific thinking mode."""
    caps = get_model_capabilities(model_name)
    return mode in caps.get("thinking_modes", [])


def get_context_window(model_name: str) -> int:
    """Get the context window size for a model."""
    caps = get_model_capabilities(model_name)
    return caps.get("context_window", 128000)


class BaseProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        thinking_mode: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt
            model: Model name
            system_prompt: Optional system prompt
            temperature: Temperature (0.0-2.0)
            thinking_mode: Thinking budget (minimal/low/medium/high/max)
            images: Optional list of image paths or data URLs

        Returns:
            Dictionary with 'content', 'usage', and optional 'metadata'
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


class GeminiProvider(BaseProvider):
    """Google Gemini API provider with thinking mode support."""

    THINKING_BUDGETS = {
        "minimal": 0.005,  # 0.5% of max
        "low": 0.08,  # 8% of max
        "medium": 0.33,  # 33% of max
        "high": 0.67,  # 67% of max
        "max": 1.0,  # 100% of max
    }

    def __init__(self, api_key: str):
        try:
            from google import genai

            self.genai = genai
            self.client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

    @property
    def provider_name(self) -> str:
        return "google"

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        thinking_mode: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        from google.genai import types

        # Build full prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Prepare content parts
        parts = [{"text": full_prompt}]

        # Add images if provided
        if images:
            for image_path in images:
                image_part = self._process_image(image_path)
                if image_part:
                    parts.append(image_part)

        contents = [{"parts": parts}]

        # Build generation config
        config = types.GenerateContentConfig(
            temperature=temperature,
            candidate_count=1,
        )

        # Add thinking config for supported models
        if thinking_mode and thinking_mode in self.THINKING_BUDGETS:
            # Check if model supports thinking (2.5 series and newer)
            if "2.5" in model or "3" in model:
                max_thinking_tokens = 24576  # Default for Gemini 2.5
                budget = int(max_thinking_tokens * self.THINKING_BUDGETS[thinking_mode])
                config.thinking_config = types.ThinkingConfig(thinking_budget=budget)

        # Execute request
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # Extract usage
        usage = {}
        try:
            if response.usage_metadata:
                usage["input_tokens"] = response.usage_metadata.prompt_token_count
                usage["output_tokens"] = response.usage_metadata.candidates_token_count
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        except (AttributeError, TypeError):
            pass

        return {
            "content": response.text or "",
            "usage": usage,
            "metadata": {
                "model": model,
                "provider": self.provider_name,
                "thinking_mode": thinking_mode,
            },
        }

    def _process_image(self, image_path: str) -> Optional[dict]:
        """Process an image for the Gemini API."""
        import base64
        from pathlib import Path

        try:
            if image_path.startswith("data:"):
                # Data URL
                _, data = image_path.split(",", 1)
                mime_type = image_path.split(";")[0].split(":")[1]
                return {"inline_data": {"mime_type": mime_type, "data": data}}
            else:
                # File path
                path = Path(image_path)
                if not path.exists():
                    return None

                # Determine MIME type
                suffix = path.suffix.lower()
                mime_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                mime_type = mime_types.get(suffix, "image/png")

                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()

                return {"inline_data": {"mime_type": mime_type, "data": data}}
        except Exception:
            return None


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        thinking_mode: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message with optional images
        if images:
            content = [{"type": "text", "text": prompt}]
            for image_path in images:
                image_content = self._process_image(image_path)
                if image_content:
                    content.append(image_content)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        # O1/O3/O4 models have fixed temperature
        effective_temp = temperature
        if any(x in model.lower() for x in ["o1", "o3", "o4"]):
            effective_temp = 1.0

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=effective_temp,
        )

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "content": response.choices[0].message.content or "",
            "usage": usage,
            "metadata": {
                "model": model,
                "provider": self.provider_name,
            },
        }

    def _process_image(self, image_path: str) -> Optional[dict]:
        """Process an image for the OpenAI API."""
        import base64
        from pathlib import Path

        try:
            if image_path.startswith("data:"):
                return {"type": "image_url", "image_url": {"url": image_path}}
            else:
                path = Path(image_path)
                if not path.exists():
                    return None

                suffix = path.suffix.lower()
                mime_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                mime_type = mime_types.get(suffix, "image/png")

                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()

                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{data}"},
                }
        except Exception:
            return None


class XAIProvider(BaseProvider):
    """X.AI (Grok) API provider."""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    @property
    def provider_name(self) -> str:
        return "xai"

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        thinking_mode: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "content": response.choices[0].message.content or "",
            "usage": usage,
            "metadata": {
                "model": model,
                "provider": self.provider_name,
            },
        }


class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider for 50+ models."""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        thinking_mode: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "content": response.choices[0].message.content or "",
            "usage": usage,
            "metadata": {
                "model": model,
                "provider": self.provider_name,
            },
        }


class CustomProvider(BaseProvider):
    """Custom/Ollama API provider for local models."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key or "ollama",
                base_url=base_url.rstrip("/") + "/v1",
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    @property
    def provider_name(self) -> str:
        return "custom"

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        thinking_mode: Optional[str] = None,
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "content": response.choices[0].message.content or "",
            "usage": usage,
            "metadata": {
                "model": model,
                "provider": self.provider_name,
            },
        }


def is_model_allowed(model: str, provider_type: str, config: dict) -> bool:
    """
    Check if a model is allowed based on restrictions config.

    Args:
        model: Model name to check
        provider_type: Provider type (google, openai, xai, openrouter)
        config: Configuration dictionary

    Returns:
        True if allowed, False if restricted
    """
    restrictions = config.get("restrictions", {})
    key = f"{provider_type}_allowed_models"
    allowed_list = restrictions.get(key, [])

    # Empty list = no restrictions (all allowed)
    if not allowed_list:
        return True

    # Check if model matches any allowed pattern (case-insensitive, partial match)
    model_lower = model.lower()
    for allowed in allowed_list:
        allowed_lower = allowed.lower().strip()
        if allowed_lower in model_lower or model_lower in allowed_lower:
            return True

    return False


def get_fallback_providers(config: dict) -> list[tuple[str, str, type]]:
    """
    Get ordered list of available providers for fallback.

    Returns:
        List of (api_key, default_model, ProviderClass) tuples in priority order
    """
    api_keys = config.get("api_keys", {})
    providers = []

    # Priority: OpenRouter > Gemini > OpenAI > XAI > Custom
    openrouter_default = config.get("defaults", {}).get("openrouter_model", "google/gemini-2.5-flash")
    if api_keys.get("openrouter") and is_model_allowed(openrouter_default, "openrouter", config):
        providers.append((api_keys["openrouter"], openrouter_default, OpenRouterProvider))

    if api_keys.get("gemini") and is_model_allowed("gemini-2.5-flash", "google", config):
        providers.append((api_keys["gemini"], "gemini-2.5-flash", GeminiProvider))

    if api_keys.get("openai") and is_model_allowed("gpt-4o", "openai", config):
        providers.append((api_keys["openai"], "gpt-4o", OpenAIProvider))

    if api_keys.get("xai") and is_model_allowed("grok-3", "xai", config):
        providers.append((api_keys["xai"], "grok-3", XAIProvider))

    if api_keys.get("custom_url"):
        providers.append((api_keys["custom_url"], "llama3.2", CustomProvider))

    return providers


def get_provider(model: str, config: dict) -> tuple[BaseProvider, str]:
    """
    Get appropriate provider for a model.

    Args:
        model: Model name or "auto"
        config: Configuration dictionary

    Returns:
        Tuple of (provider_instance, resolved_model_name)

    Raises:
        ValueError: If model is not allowed by restrictions
    """
    api_keys = config.get("api_keys", {})

    # Handle 'auto' mode
    if model.lower() == "auto":
        # Priority: OpenRouter > Gemini > OpenAI > XAI > Custom
        # Respect restrictions when selecting auto model
        openrouter_default = config.get("defaults", {}).get("openrouter_model", "google/gemini-2.5-flash")
        if api_keys.get("openrouter") and is_model_allowed(openrouter_default, "openrouter", config):
            return OpenRouterProvider(api_keys["openrouter"]), openrouter_default
        elif api_keys.get("gemini") and is_model_allowed("gemini-2.5-flash", "google", config):
            return GeminiProvider(api_keys["gemini"]), "gemini-2.5-flash"
        elif api_keys.get("openai") and is_model_allowed("gpt-4o", "openai", config):
            return OpenAIProvider(api_keys["openai"]), "gpt-4o"
        elif api_keys.get("xai") and is_model_allowed("grok-3", "xai", config):
            return XAIProvider(api_keys["xai"]), "grok-3"
        elif api_keys.get("custom_url"):
            return CustomProvider(api_keys["custom_url"]), "llama3.2"
        else:
            raise ValueError("No API keys configured or all models restricted.")

    # Route by model name
    model_lower = model.lower()

    # Check OpenRouter format first (provider/model)
    if "/" in model:
        if not is_model_allowed(model, "openrouter", config):
            allowed = config.get("restrictions", {}).get("openrouter_allowed_models", [])
            raise ValueError(f"Model '{model}' not in allowed list: {allowed}")
        key = api_keys.get("openrouter") or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not configured")
        return OpenRouterProvider(key), model

    elif any(x in model_lower for x in ["gpt", "o1", "o3", "o4", "chatgpt"]):
        if not is_model_allowed(model, "openai", config):
            allowed = config.get("restrictions", {}).get("openai_allowed_models", [])
            raise ValueError(f"Model '{model}' not in allowed list: {allowed}")
        key = api_keys.get("openai") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not configured")
        return OpenAIProvider(key), model

    elif "grok" in model_lower:
        if not is_model_allowed(model, "xai", config):
            allowed = config.get("restrictions", {}).get("xai_allowed_models", [])
            raise ValueError(f"Model '{model}' not in allowed list: {allowed}")
        key = api_keys.get("xai") or os.environ.get("XAI_API_KEY")
        if not key:
            raise ValueError("XAI_API_KEY not configured")
        return XAIProvider(key), model

    elif "gemini" in model_lower:
        if not is_model_allowed(model, "google", config):
            allowed = config.get("restrictions", {}).get("google_allowed_models", [])
            raise ValueError(f"Model '{model}' not in allowed list: {allowed}")
        key = api_keys.get("gemini") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not configured")
        return GeminiProvider(key), model

    else:
        # Try custom/local (no restrictions for custom models)
        url = api_keys.get("custom_url") or os.environ.get("CUSTOM_API_URL")
        if url:
            return CustomProvider(url), model

        raise ValueError(f"Unknown model '{model}' and no custom endpoint configured")


def execute_request(
    provider: BaseProvider,
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    thinking_mode: Optional[str] = None,
    images: Optional[list[str]] = None,
    max_retries: int = 4,
    config: Optional[dict] = None,
    **kwargs,
) -> dict:
    """
    Execute a request with retry logic and automatic provider fallback.

    Args:
        provider: Provider instance
        prompt: User prompt
        model: Model name
        system_prompt: Optional system prompt
        temperature: Temperature
        thinking_mode: Thinking mode for supported models
        images: Optional images
        max_retries: Maximum retry attempts per provider
        config: Configuration dict (enables automatic fallback to other providers)

    Returns:
        Response dictionary
    """
    delays = [1, 3, 5, 8]
    last_error = None

    # Try primary provider
    for attempt in range(max_retries):
        try:
            return provider.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                thinking_mode=thinking_mode,
                images=images,
                **kwargs,
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = delays[min(attempt, len(delays) - 1)]
                time.sleep(delay)

    # If config provided, try fallback providers
    if config:
        fallback_providers = get_fallback_providers(config)
        for api_key, fallback_model, ProviderClass in fallback_providers:
            # Skip same provider type
            if isinstance(provider, ProviderClass):
                continue

            try:
                fallback_provider = ProviderClass(api_key)
                result = fallback_provider.generate(
                    prompt=prompt,
                    model=fallback_model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    thinking_mode=thinking_mode,
                    images=images,
                    **kwargs,
                )
                # Add fallback info to metadata
                result["metadata"]["fallback"] = True
                result["metadata"]["original_model"] = model
                result["metadata"]["original_provider"] = provider.provider_name
                return result
            except Exception:
                continue  # Try next fallback

    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_error}")
