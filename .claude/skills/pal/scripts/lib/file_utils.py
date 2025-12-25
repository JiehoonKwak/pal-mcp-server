"""
File utilities for PAL Skills.

Provides:
- File reading with line numbers
- Token estimation
- Path expansion and validation
- Directory traversal
"""

import os
from pathlib import Path
from typing import Optional

# Excluded directories (common patterns to skip)
EXCLUDED_DIRS = {
    "node_modules",
    ".git",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".eggs",
    "*.egg-info",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "coverage",
    ".coverage",
    ".nyc_output",
}

# Binary file extensions to skip
BINARY_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".obj",
    ".o",
    ".a",
    ".lib",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wav",
    ".flac",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
}


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token on average.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4


def read_file_content(
    file_path: str,
    include_line_numbers: bool = True,
    max_lines: Optional[int] = None,
) -> tuple[str, int]:
    """
    Read file content with optional line numbers.

    Args:
        file_path: Path to the file
        include_line_numbers: Whether to add line number markers
        max_lines: Optional maximum lines to read

    Returns:
        Tuple of (formatted_content, token_estimate)
    """
    path = Path(file_path).resolve()

    if not path.exists():
        return f"\n--- ERROR: File not found: {file_path} ---\n", 0

    if not path.is_file():
        return f"\n--- ERROR: Not a file: {file_path} ---\n", 0

    # Skip binary files
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return f"\n--- SKIPPED: Binary file: {file_path} ---\n", 0

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if max_lines:
            lines = lines[:max_lines]

        if include_line_numbers:
            # Format with line numbers: "LINE│ content"
            formatted_lines = []
            for i, line in enumerate(lines, 1):
                # Remove trailing newline for consistent formatting
                line_content = line.rstrip("\n\r")
                formatted_lines.append(f"{i:4d}│ {line_content}")
            content = "\n".join(formatted_lines)
        else:
            content = "".join(lines)

        # Wrap in file markers
        formatted = f"\n--- BEGIN FILE: {file_path} ---\n{content}\n--- END FILE: {file_path} ---\n"

        return formatted, estimate_tokens(formatted)

    except Exception as e:
        return f"\n--- ERROR reading {file_path}: {e} ---\n", 0


def expand_paths(paths: list[str]) -> list[str]:
    """
    Expand paths, handling directories recursively.

    Args:
        paths: List of file/directory paths

    Returns:
        List of individual file paths
    """
    expanded = []

    for path_str in paths:
        path = Path(path_str).resolve()

        if path.is_file():
            expanded.append(str(path))
        elif path.is_dir():
            # Recursively find files
            for file_path in _walk_directory(path):
                expanded.append(str(file_path))

    return expanded


def _walk_directory(directory: Path, max_depth: int = 10) -> list[Path]:
    """
    Walk directory recursively, respecting exclusions.

    Args:
        directory: Directory to walk
        max_depth: Maximum recursion depth

    Returns:
        List of file paths
    """
    files = []

    def _walk(current: Path, depth: int):
        if depth > max_depth:
            return

        try:
            for entry in current.iterdir():
                # Skip excluded directories
                if entry.is_dir():
                    if entry.name in EXCLUDED_DIRS:
                        continue
                    _walk(entry, depth + 1)
                elif entry.is_file():
                    # Skip binary files
                    if entry.suffix.lower() not in BINARY_EXTENSIONS:
                        files.append(entry)
        except PermissionError:
            pass

    _walk(directory, 0)
    return sorted(files)


def read_files(
    paths: list[str],
    include_line_numbers: bool = True,
    max_tokens: int = 100000,
    reserve_tokens: int = 1000,
) -> str:
    """
    Read multiple files with token budget management.

    Args:
        paths: List of file/directory paths
        include_line_numbers: Whether to add line numbers
        max_tokens: Maximum tokens for all files
        reserve_tokens: Tokens to reserve

    Returns:
        Combined formatted file content
    """
    if not paths:
        return ""

    expanded_paths = expand_paths(paths)

    if not expanded_paths:
        return ""

    parts = []
    total_tokens = 0
    effective_limit = max_tokens - reserve_tokens

    for file_path in expanded_paths:
        content, tokens = read_file_content(file_path, include_line_numbers)

        if total_tokens + tokens > effective_limit:
            # Add note about skipped files
            remaining = len(expanded_paths) - len(parts)
            if remaining > 0:
                parts.append(
                    f"\n[NOTE: {remaining} file(s) omitted due to token limit. "
                    f"Used {total_tokens:,} of {effective_limit:,} tokens.]\n"
                )
            break

        parts.append(content)
        total_tokens += tokens

    return "".join(parts)


def validate_path(path: str, must_exist: bool = True) -> Optional[str]:
    """
    Validate a file path.

    Args:
        path: Path to validate
        must_exist: Whether the path must exist

    Returns:
        Error message if invalid, None if valid
    """
    if not path:
        return "Path cannot be empty"

    # Expand user home
    expanded = os.path.expanduser(path)

    if not os.path.isabs(expanded):
        return f"Path must be absolute: {path}"

    if must_exist and not os.path.exists(expanded):
        return f"Path does not exist: {path}"

    return None
