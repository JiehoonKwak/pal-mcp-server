"""
Conversation Memory for PAL Skills.

Provides multi-turn conversation persistence with:
- UUID-based threading
- Cross-tool continuation
- Newest-first file deduplication
- SQLite or in-memory storage
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class InMemoryStorage:
    """Simple in-memory storage with TTL support."""

    def __init__(self):
        self._data: dict[str, tuple[dict, float]] = {}

    def set(self, key: str, value: dict, ttl: int = 10800) -> None:
        """Store value with TTL (default 3 hours)."""
        expires_at = datetime.now(timezone.utc).timestamp() + ttl
        self._data[key] = (value, expires_at)

    def get(self, key: str) -> Optional[dict]:
        """Get value if not expired."""
        if key not in self._data:
            return None

        value, expires_at = self._data[key]
        if datetime.now(timezone.utc).timestamp() > expires_at:
            del self._data[key]
            return None

        return value

    def delete(self, key: str) -> None:
        """Delete a key."""
        self._data.pop(key, None)


class SQLiteStorage:
    """SQLite-based persistent storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON conversations(expires_at)")

    def set(self, key: str, value: dict, ttl: int = 10800) -> None:
        """Store value with TTL."""
        expires_at = datetime.now(timezone.utc).timestamp() + ttl
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO conversations (key, value, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), expires_at),
            )

    def get(self, key: str) -> Optional[dict]:
        """Get value if not expired."""
        now = datetime.now(timezone.utc).timestamp()
        with sqlite3.connect(self.db_path) as conn:
            # Clean expired entries
            conn.execute("DELETE FROM conversations WHERE expires_at < ?", (now,))

            cursor = conn.execute(
                "SELECT value FROM conversations WHERE key = ? AND expires_at > ?",
                (key, now),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def delete(self, key: str) -> None:
        """Delete a key."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversations WHERE key = ?", (key,))


class ConversationMemory:
    """
    Multi-turn conversation memory with cross-tool continuation.

    Features:
    - UUID-based threading for unique conversation identification
    - Cross-tool continuation (chat → codereview → debug)
    - Newest-first file deduplication
    - Configurable storage (memory or SQLite)
    """

    def __init__(self, config: dict):
        """
        Initialize conversation memory.

        Args:
            config: Configuration dictionary with conversation settings
        """
        conv_config = config.get("conversation", {})
        self.max_turns = conv_config.get("max_turns", 50)
        self.timeout_hours = conv_config.get("timeout_hours", 3)
        self.timeout_seconds = self.timeout_hours * 3600

        storage_type = conv_config.get("storage", "memory")
        if storage_type == "sqlite":
            db_path = Path.home() / ".pal" / "conversations.db"
            self.storage = SQLiteStorage(db_path)
        else:
            self.storage = InMemoryStorage()

    def create_thread(self, tool_name: str, initial_context: dict) -> dict:
        """
        Create a new conversation thread.

        Args:
            tool_name: Name of the tool creating the thread
            initial_context: Initial request context

        Returns:
            Thread dictionary with thread_id
        """
        thread_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Filter out non-serializable parameters
        filtered_context = {
            k: v
            for k, v in initial_context.items()
            if k not in ["temperature", "thinking_mode", "model", "continuation_id"]
        }

        # Extract parent_thread_id if continuing from another thread
        parent_thread_id = initial_context.get("parent_thread_id", "")

        thread = {
            "thread_id": thread_id,
            "parent_thread_id": parent_thread_id,  # For thread chaining
            "created_at": now,
            "last_updated_at": now,
            "tool_name": tool_name,
            "turns": [],
            "initial_context": filtered_context,
        }

        self.storage.set(thread_id, thread, ttl=self.timeout_seconds)
        return thread

    def get_thread(self, thread_id: str) -> Optional[dict]:
        """
        Get thread by ID.

        Args:
            thread_id: UUID of the thread

        Returns:
            Thread dictionary or None if not found/expired
        """
        if not thread_id or not self._is_valid_uuid(thread_id):
            return None
        return self.storage.get(thread_id)

    def add_turn(
        self,
        thread_id: str,
        role: str,
        content: str,
        files: Optional[list[str]] = None,
        images: Optional[list[str]] = None,
        tool_name: Optional[str] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_metadata: Optional[dict] = None,
    ) -> bool:
        """
        Add a turn to the conversation.

        Args:
            thread_id: Thread UUID
            role: "user" or "assistant"
            content: Turn content
            files: Optional list of files referenced
            images: Optional list of images referenced
            tool_name: Tool that generated this turn
            model_name: Model used (e.g., "gemini-2.5-flash")
            model_provider: Provider used (e.g., "google")
            model_metadata: Optional metadata about the model response (tokens, thinking, etc.)

        Returns:
            True if successful, False otherwise
        """
        thread = self.get_thread(thread_id)
        if not thread:
            return False

        if len(thread["turns"]) >= self.max_turns:
            return False

        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "images": images,
            "tool_name": tool_name,
            "model_name": model_name,
            "model_provider": model_provider,
            "model_metadata": model_metadata,
        }

        thread["turns"].append(turn)
        thread["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.storage.set(thread_id, thread, ttl=self.timeout_seconds)
        return True

    def build_history(self, thread: dict, max_tokens: int = 100000) -> str:
        """
        Build formatted conversation history.

        Uses newest-first prioritization for files and token-aware turn selection.

        Args:
            thread: Thread dictionary
            max_tokens: Maximum tokens for history

        Returns:
            Formatted conversation history string
        """
        if not thread or not thread.get("turns"):
            return ""

        turns = thread["turns"]

        # Collect files with newest-first deduplication
        seen_files = set()
        all_files = []
        for turn in reversed(turns):
            for f in turn.get("files") or []:
                if f not in seen_files:
                    seen_files.add(f)
                    all_files.append(f)

        parts = [
            "=== CONVERSATION HISTORY (CONTINUATION) ===",
            f"Thread: {thread['thread_id']}",
            f"Tool: {thread['tool_name']}",
            f"Turn {len(turns)}/{self.max_turns}",
            "You are continuing this conversation thread from where it left off.",
            "",
        ]

        # Add file references if any
        if all_files:
            parts.append("Files referenced in this conversation:")
            for f in all_files:
                parts.append(f"  - {f}")
            parts.append("")

        # Add previous turns
        parts.append("Previous conversation turns:")
        for i, turn in enumerate(turns, 1):
            if turn["role"] == "user":
                role_label = "Agent"
            else:
                role_label = turn.get("model_name") or "Assistant"

            header = f"\n--- Turn {i} ({role_label}"
            if turn.get("tool_name"):
                header += f" using {turn['tool_name']}"
            if turn.get("model_provider"):
                header += f" via {turn['model_provider']}"
            header += ") ---"
            parts.append(header)

            if turn.get("files"):
                parts.append(f"Files: {', '.join(turn['files'])}")

            parts.append(turn["content"])

        parts.extend(
            [
                "",
                "=== END CONVERSATION HISTORY ===",
                "",
                "IMPORTANT: You are continuing an existing conversation thread.",
                "Build upon the previous exchanges shown above.",
                "DO NOT repeat or summarize previous analysis - provide only new insights.",
                f"This is turn {len(turns) + 1} of the conversation.",
            ]
        )

        return "\n".join(parts)

    def get_conversation_files(self, thread: dict) -> list[str]:
        """
        Get all unique files from conversation with newest-first prioritization.

        Args:
            thread: Thread dictionary

        Returns:
            List of unique file paths, newest references first
        """
        if not thread or not thread.get("turns"):
            return []

        seen_files = set()
        file_list = []

        # Walk backwards (newest first)
        for turn in reversed(thread["turns"]):
            for f in turn.get("files") or []:
                if f not in seen_files:
                    seen_files.add(f)
                    file_list.append(f)

        return file_list

    def get_thread_chain(self, thread_id: str, max_depth: int = 20) -> list[dict]:
        """
        Traverse parent chain to get all threads in sequence.

        This allows following the lineage of conversations across tool switches.

        Args:
            thread_id: Starting thread UUID
            max_depth: Maximum chain depth to prevent infinite loops

        Returns:
            List of thread dictionaries in chronological order (oldest first)
        """
        chain = []
        current_id = thread_id
        seen_ids: set[str] = set()

        while current_id and len(chain) < max_depth:
            # Circular reference protection
            if current_id in seen_ids:
                break
            seen_ids.add(current_id)

            context = self.get_thread(current_id)
            if not context:
                break

            chain.append(context)
            current_id = context.get("parent_thread_id", "")

        # Return in chronological order (oldest first)
        chain.reverse()
        return chain

    def is_thread_expired(self, thread: dict) -> bool:
        """
        Check if a thread has expired based on timeout_hours.

        Args:
            thread: Thread dictionary

        Returns:
            True if thread has expired
        """
        if not thread or not thread.get("created_at"):
            return True

        try:
            created = datetime.fromisoformat(thread["created_at"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_hours = (now - created).total_seconds() / 3600
            return age_hours > self.timeout_hours
        except (ValueError, TypeError):
            return True

    def create_child_thread(self, parent_thread_id: str, tool_name: str, initial_context: dict) -> Optional[dict]:
        """
        Create a new thread that chains from a parent thread.

        This is used when switching tools but continuing the conversation.

        Args:
            parent_thread_id: UUID of the parent thread
            tool_name: Name of the new tool
            initial_context: Initial context for the new thread

        Returns:
            New thread dictionary or None if parent not found
        """
        # Verify parent exists
        parent = self.get_thread(parent_thread_id)
        if not parent:
            return None

        # Create new thread with parent reference
        context = dict(initial_context)
        context["parent_thread_id"] = parent_thread_id
        return self.create_thread(tool_name, context)

    @staticmethod
    def _is_valid_uuid(val: str) -> bool:
        """Validate UUID format."""
        try:
            uuid.UUID(val)
            return True
        except ValueError:
            return False
