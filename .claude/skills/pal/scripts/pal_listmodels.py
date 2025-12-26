#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///
"""
PAL ListModels - List available AI models and their configuration status.

Run with uv (recommended):
    uv run scripts/pal_listmodels.py [options]

Usage:
    python pal_listmodels.py [options]

Options:
    --json    Output as JSON

Examples:
    # List all available models
    python pal_listmodels.py

    # Output as JSON
    python pal_listmodels.py --json
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response  # noqa: E402

from config import load_config  # noqa: E402
from providers import MODEL_CAPABILITIES  # noqa: E402

# Provider configurations with their env vars and known models
PROVIDER_INFO = {
    "gemini": {
        "display_name": "Google Gemini",
        "env_var": "GEMINI_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    },
    "openai": {
        "display_name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"],
    },
    "xai": {
        "display_name": "X.AI (Grok)",
        "env_var": "XAI_API_KEY",
        "models": ["grok-3", "grok-2"],
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "models": ["google/gemini-3-flash-preview", "anthropic/claude-sonnet-4", "openai/gpt-4o"],
    },
    "custom_url": {
        "display_name": "Custom/Ollama",
        "env_var": "CUSTOM_API_URL",
        "models": ["(local models via OpenAI-compatible API)"],
    },
}


def is_provider_configured(key: str, config: dict) -> bool:
    """Check if a provider has a valid API key configured."""
    api_keys = config.get("api_keys", {})
    value = api_keys.get(key, "")
    # Check it's not empty and not a placeholder
    return bool(value) and not value.startswith("${") and not value.startswith("your_")


def get_allowed_models(provider_key: str, config: dict) -> list[str]:
    """Get allowed models for a provider based on restrictions."""
    restrictions = config.get("restrictions", {})
    # Map provider keys to restriction keys
    restriction_key_map = {
        "gemini": "google_allowed_models",
        "openai": "openai_allowed_models",
        "xai": "xai_allowed_models",
        "openrouter": "openrouter_allowed_models",
    }
    key = restriction_key_map.get(provider_key)
    if key:
        return restrictions.get(key, [])
    return []


def format_model_info(model: str) -> str:
    """Format model name with capabilities info."""
    caps = MODEL_CAPABILITIES.get(model, {})
    if not caps:
        return f"  - {model}"

    features = []
    if caps.get("thinking_modes"):
        features.append("thinking")
    if caps.get("supports_images"):
        features.append("vision")

    context = caps.get("context_window", 0)
    if context >= 1000000:
        context_str = f"{context // 1000000}M"
    elif context >= 1000:
        context_str = f"{context // 1000}K"
    else:
        context_str = str(context)

    if features:
        return f"  - {model} ({context_str} ctx, {', '.join(features)})"
    else:
        return f"  - {model} ({context_str} ctx)"


def build_markdown_output(config: dict) -> str:
    """Build markdown-formatted output of available models."""
    lines = ["# Available AI Models", ""]

    for provider_key, info in PROVIDER_INFO.items():
        display_name = info["display_name"]
        is_configured = is_provider_configured(provider_key, config)

        if is_configured:
            lines.append(f"## {display_name} [checkmark]")

            # Get allowed models (if restrictions exist)
            allowed = get_allowed_models(provider_key, config)
            if allowed:
                lines.append(f"*Restricted to: {', '.join(allowed)}*")
                lines.append("")

            # List models
            for model in info["models"]:
                lines.append(format_model_info(model))
        else:
            lines.append(f"## {display_name} [x]")
            lines.append(f"Not configured (set {info['env_var']})")

        lines.append("")

    # Add default model info
    defaults = config.get("defaults", {})
    lines.append("## Current Defaults")
    lines.append(f"- Model: {defaults.get('model', 'auto')}")
    lines.append(f"- Temperature: {defaults.get('temperature', 1.0)}")
    lines.append(f"- Thinking mode: {defaults.get('thinking_mode', 'medium')}")
    if defaults.get("openrouter_model"):
        lines.append(f"- OpenRouter default: {defaults.get('openrouter_model')}")

    return "\n".join(lines)


def build_json_output(config: dict) -> dict:
    """Build JSON output of available models."""
    providers = {}

    for provider_key, info in PROVIDER_INFO.items():
        is_configured = is_provider_configured(provider_key, config)
        allowed = get_allowed_models(provider_key, config)

        models_with_caps = []
        for model in info["models"]:
            caps = MODEL_CAPABILITIES.get(model, {})
            models_with_caps.append(
                {
                    "name": model,
                    "context_window": caps.get("context_window"),
                    "thinking_modes": caps.get("thinking_modes", []),
                    "supports_images": caps.get("supports_images", False),
                }
            )

        providers[provider_key] = {
            "display_name": info["display_name"],
            "configured": is_configured,
            "env_var": info["env_var"],
            "models": models_with_caps,
            "restrictions": allowed if allowed else None,
        }

    return {
        "providers": providers,
        "defaults": config.get("defaults", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="PAL ListModels - List available AI models")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config()

        # Build output
        if args.json:
            models_data = build_json_output(config)

            # Build agentic response
            result = build_agentic_response(
                tool_name="listmodels",
                status="success",
                content=json.dumps(models_data, indent=2),
                continuation_id="",
                model="",
                provider="",
                usage={},
                confidence="certain",
            )
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            output = build_markdown_output(config)
            print(output)

    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        if args.json:
            print(json.dumps(error_result, indent=2))
            sys.exit(1)
        else:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
