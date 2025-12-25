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


def get_provider(model: str, config: dict) -> tuple[BaseProvider, str]:
    """
    Get appropriate provider for a model.

    Args:
        model: Model name or "auto"
        config: Configuration dictionary

    Returns:
        Tuple of (provider_instance, resolved_model_name)
    """
    api_keys = config.get("api_keys", {})

    # Handle 'auto' mode
    if model.lower() == "auto":
        # Priority: Gemini > OpenAI > XAI > OpenRouter > Custom
        if api_keys.get("gemini"):
            return GeminiProvider(api_keys["gemini"]), "gemini-2.5-flash"
        elif api_keys.get("openai"):
            return OpenAIProvider(api_keys["openai"]), "gpt-4o"
        elif api_keys.get("xai"):
            return XAIProvider(api_keys["xai"]), "grok-3"
        elif api_keys.get("openrouter"):
            return OpenRouterProvider(api_keys["openrouter"]), "google/gemini-2.5-flash"
        elif api_keys.get("custom_url"):
            return CustomProvider(api_keys["custom_url"]), "llama3.2"
        else:
            raise ValueError("No API keys configured. Set at least one provider API key.")

    # Route by model name
    model_lower = model.lower()

    if "gemini" in model_lower:
        key = api_keys.get("gemini") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not configured")
        return GeminiProvider(key), model

    elif any(x in model_lower for x in ["gpt", "o1", "o3", "o4", "chatgpt"]):
        key = api_keys.get("openai") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not configured")
        return OpenAIProvider(key), model

    elif "grok" in model_lower:
        key = api_keys.get("xai") or os.environ.get("XAI_API_KEY")
        if not key:
            raise ValueError("XAI_API_KEY not configured")
        return XAIProvider(key), model

    elif "/" in model:  # OpenRouter format: provider/model
        key = api_keys.get("openrouter") or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not configured")
        return OpenRouterProvider(key), model

    else:
        # Try custom/local
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
    **kwargs,
) -> dict:
    """
    Execute a request with retry logic.

    Args:
        provider: Provider instance
        prompt: User prompt
        model: Model name
        system_prompt: Optional system prompt
        temperature: Temperature
        thinking_mode: Thinking mode for supported models
        images: Optional images
        max_retries: Maximum retry attempts

    Returns:
        Response dictionary
    """
    delays = [1, 3, 5, 8]
    last_error = None

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

    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_error}")
