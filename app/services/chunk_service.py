from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.utils.hashing import sha256_text


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    collection: str
    chunk_index: int
    text: str
    text_hash: str
    token_count: int
    metadata: dict[str, Any]


class ChunkService:
    def chunk_document(
        self,
        *,
        collection: str,
        document_id: str,
        title: str,
        text: str,
        metadata: dict[str, Any],
        strategy: str,
    ) -> list[Chunk]:
        if strategy == "json_fields":
            parts = self._json_field_parts(text)
        elif strategy == "markdown_semantic":
            parts = self._markdown_parts(text)
        else:
            parts = self._plain_text_parts(text)
        chunks = []
        base_metadata = {**metadata, "title": title}
        for index, part in enumerate(parts):
            part_hash = sha256_text(part)
            chunk_id = f"{document_id}:chunk:{index:04d}:{part_hash[:12]}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    collection=collection,
                    chunk_index=index,
                    text=part,
                    text_hash=part_hash,
                    token_count=len(part.split()),
                    metadata={**base_metadata, "chunk_index": index},
                )
            )
        return chunks

    def _plain_text_parts(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
        normalized = text.strip()
        if len(normalized) <= chunk_size:
            return [normalized]
        parts: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + chunk_size)
            parts.append(normalized[start:end].strip())
            if end == len(normalized):
                break
            start = max(0, end - overlap)
        return [part for part in parts if part]

    def _markdown_parts(self, text: str) -> list[str]:
        sections: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith(("# ", "## ", "### ")) and current:
                sections.extend(self._plain_text_parts("\n".join(current).strip()))
                current = [line]
            else:
                current.append(line)
        if current:
            sections.extend(self._plain_text_parts("\n".join(current).strip()))
        return sections or [text.strip()]

    def _json_field_parts(self, text: str) -> list[str]:
        parts = [part.strip() for part in text.split("\n\n") if part.strip()]
        return parts or [text.strip()]
