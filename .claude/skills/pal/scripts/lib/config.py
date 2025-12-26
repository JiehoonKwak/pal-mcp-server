"""
Configuration loading for PAL Skills.

Supports:
- YAML configuration files
- Environment variable substitution
- Local overrides
- CLI client configurations
"""

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml


def _get_skill_root() -> Path:
    """Get the root directory of the PAL skill."""
    return Path(__file__).parent.parent.parent


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${ENV_VAR} patterns with environment values."""
    if isinstance(value, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(var_name, default)

        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load PAL configuration from YAML file.

    Priority (highest to lowest):
    1. Environment variables (always override)
    2. config/config.local.yaml (project-specific)
    3. config/config.yaml (main config)
    4. Built-in defaults

    Args:
        config_path: Optional explicit config path

    Returns:
        Merged configuration dictionary
    """
    skill_root = _get_skill_root()
    config_dir = skill_root / "config"

    # Start with defaults
    config = {
        "version": "1.0",
        "api_keys": {
            "gemini": os.environ.get("GEMINI_API_KEY", ""),
            "openai": os.environ.get("OPENAI_API_KEY", ""),
            "xai": os.environ.get("XAI_API_KEY", ""),
            "openrouter": os.environ.get("OPENROUTER_API_KEY", ""),
            "custom_url": os.environ.get("CUSTOM_API_URL", ""),
        },
        "defaults": {
            "model": "auto",
            "temperature": 1.0,
            "thinking_mode": "medium",
        },
        "conversation": {
            "max_turns": 50,
            "timeout_hours": 3,
            "storage": "memory",
        },
        "restrictions": {
            "google_allowed_models": [],
            "openai_allowed_models": [],
        },
    }

    # Load main config
    main_config_path = config_path or (config_dir / "config.yaml")
    if main_config_path.exists():
        try:
            with open(main_config_path) as f:
                file_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, file_config)
        except Exception as e:
            print(f"Warning: Failed to load {main_config_path}: {e}")

    # Load local overrides
    local_config_path = config_dir / "config.local.yaml"
    if local_config_path.exists():
        try:
            with open(local_config_path) as f:
                local_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, local_config)
        except Exception as e:
            print(f"Warning: Failed to load {local_config_path}: {e}")

    # Substitute environment variables
    config = _substitute_env_vars(config)

    # Ensure API keys from environment override file values
    env_keys = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "custom_url": "CUSTOM_API_URL",
    }
    for key, env_var in env_keys.items():
        env_value = os.environ.get(env_var)
        if env_value:
            config["api_keys"][key] = env_value

    return config


def load_cli_client(client_name: str) -> dict:
    """
    Load CLI client configuration.

    Args:
        client_name: Name of the CLI client (e.g., "gemini", "claude")

    Returns:
        Client configuration dictionary
    """
    skill_root = _get_skill_root()
    client_path = skill_root / "config" / "cli_clients" / f"{client_name}.yaml"

    if not client_path.exists():
        raise FileNotFoundError(f"CLI client config not found: {client_path}")

    with open(client_path) as f:
        client_config = yaml.safe_load(f) or {}

    return _substitute_env_vars(client_config)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_available_api_keys(config: dict) -> list[str]:
    """Get list of providers with valid API keys."""
    api_keys = config.get("api_keys", {})
    available = []

    key_names = {
        "gemini": "Google Gemini",
        "openai": "OpenAI",
        "xai": "X.AI",
        "openrouter": "OpenRouter",
        "custom_url": "Custom/Ollama",
    }

    for key, name in key_names.items():
        value = api_keys.get(key, "")
        if value and not value.startswith("your_") and value != "":
            available.append(name)

    return available
