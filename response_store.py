from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class StoredResponseRecord:
    id: str
    created_at: int
    model: str
    metadata: dict[str, Any] | None
    request_options: dict[str, Any]
    instructions: str | None
    input_items: list[dict[str, Any]]
    previous_response_id: str | None
    transcript: list[dict[str, Any]]
    output: list[dict[str, Any]]
    usage: dict[str, Any]
    status: str
    store: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "model": self.model,
            "metadata": self.metadata,
            "request_options": self.request_options,
            "instructions": self.instructions,
            "input_items": self.input_items,
            "previous_response_id": self.previous_response_id,
            "transcript": self.transcript,
            "output": self.output,
            "usage": self.usage,
            "status": self.status,
            "store": self.store,
        }


class ResponseStore:
    def __init__(self, max_items: int = 256, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._records: OrderedDict[str, tuple[StoredResponseRecord, int]] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    @staticmethod
    def _estimate_size(record: StoredResponseRecord) -> int:
        payload = json.dumps(record.as_dict(), ensure_ascii=False, separators=(",", ":"))
        return len(payload.encode("utf-8"))

    def get(self, response_id: str) -> StoredResponseRecord | None:
        with self._lock:
            entry = self._records.get(response_id)
            if entry is None:
                return None
            record, size = entry
            self._records.move_to_end(response_id)
            return record

    def put(self, record: StoredResponseRecord) -> None:
        size = self._estimate_size(record)
        if size > self.max_bytes:
            return

        with self._lock:
            existing = self._records.pop(record.id, None)
            if existing is not None:
                _, existing_size = existing
                self._bytes -= existing_size

            self._records[record.id] = (record, size)
            self._bytes += size
            self._evict_if_needed()

    def delete(self, response_id: str) -> bool:
        with self._lock:
            existing = self._records.pop(response_id, None)
            if existing is None:
                return False
            _, size = existing
            self._bytes -= size
            return True

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "items": len(self._records),
                "bytes": self._bytes,
                "max_items": self.max_items,
                "max_bytes": self.max_bytes,
            }

    def _evict_if_needed(self) -> None:
        while self._records and (
            len(self._records) > self.max_items or self._bytes > self.max_bytes
        ):
            _, (_, size) = self._records.popitem(last=False)
            self._bytes -= size
