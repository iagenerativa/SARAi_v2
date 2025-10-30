"""Tone memory buffer for Layer 2.

This module persistently stores tone events produced by Layer 1 so that
subsequent layers (memory, reasoning, fillers) can access historical context.
Entries are appended to a JSONL file and retained in-memory using a bounded
deque for quick access.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional


class ToneMemoryBuffer:
    """Persistent buffer that stores tone analysis events."""

    def __init__(
        self,
        storage_path: Path | str = Path("state/layer2_tone_memory.jsonl"),
        max_entries: int = 256,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.max_entries = max_entries
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._buffer: Deque[Dict] = deque(maxlen=max_entries)

        if self.storage_path.exists():
            self._prime_from_disk()

    def _prime_from_disk(self) -> None:
        """Load the most recent entries from disk into the in-memory buffer."""
        try:
            with self.storage_path.open("r", encoding="utf-8") as fh:
                tail: List[str] = fh.readlines()[-self.max_entries :]
        except Exception:
            return

        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._buffer.append(payload)

    def append(self, entry: Dict) -> None:
        """Append a tone entry and persist it to disk."""
        payload = dict(entry)
        payload.setdefault("timestamp", time.time())
        with self._lock:
            self._buffer.append(payload)
            with self.storage_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 10) -> List[Dict]:
        with self._lock:
            return list(self._buffer)[-limit:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def iter_recent(self, limit: Optional[int] = None) -> Iterable[Dict]:
        data = self.recent(limit or self.max_entries)
        for item in data:
            yield item

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            if self.storage_path.exists():
                self.storage_path.unlink()