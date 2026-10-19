"""Uploaded files on local disk. The database stores only the relative ``storage_key``."""

from __future__ import annotations

import os
from pathlib import Path


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(self, doc_id: str, ext: str, data: bytes) -> str:
        key = f"uploads/{doc_id[:2]}/{doc_id}{ext}"
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)  # never leave a half-written file under the final name
        return key

    def path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"storage key escapes the storage root: {key!r}")
        return path

    def read(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)
